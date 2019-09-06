'''
This file contains methods responsible for collecting data from
NOAA predictions and collected Planet cloud mask data.

In hindsight, I found out that Planet_common has methods for doing a lot of this. Should've just used those.

author: Kyle Cochran
'''
import urllib3
import os
import json
import requests
from requests.auth import HTTPBasicAuth
import shutil
from datetime import datetime
from datetime import timedelta
import time
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from multiprocessing.dummy import Pool as ThreadPool
from retrying import retry
from osgeo import gdal
from itertools import compress
from io import open as iopen
from cgi import parse_header

# Do some setup for stuff we'll use a lot
auth_key=os.environ["PLANET_API_TOKEN"] + ':'
base_url = "https://api.planet.com/data"
api_version = "v1"
versioned_url = base_url + '/' + api_version + '/'
auth_header = HTTPBasicAuth(auth_key, '')

"""
    HELPER FUNCTIONS=================================================
"""

def activeprint(text):
    '''
        prints then flushes the stream. Useful for printing inside of a loop
    '''
    sys.stdout.write(str(text))
    sys.stdout.flush()

def jsonfile(data: dict):
    '''
        saves a dict to a debug file
    '''
    with open('debug.json', 'w') as f:
        json.dump(data, f)

def ISO8601Z_to_datetime(time: str):
    return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")

def datetime_to_ISO8601Z(time: datetime):
    return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def geodatefilter(geojson: dict, tstart: datetime, tend: datetime):
    geo_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson
    }

    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": tstart.isoformat() + 'Z',
        "lte": tend.isoformat() + 'Z'
      }
    }

    filter = {
        "type":"AndFilter",
        "config": [geo_filter, date_range_filter]
    }

    return filter

def geojsonisin(innergeojson: dict, outergeojson: dict) -> bool:
    '''
    Returns whether geometry of innergeojson is held completely within innergeojson

    The assumption being of course that outergeojson is convex
    '''

    pts1 = innergeojson["coordinates"][0]
    pts2 = outergeojson["coordinates"][0]

    #uncomment for visual confirmation
    '''from matplotlib import pyplot as plt
    for pt in pts1: plt.plot(pt[0], pt[1], "bo")
    for pt in pts2: plt.plot(pt[0], pt[1], "r+")
    plt.show()'''

    outerpoly = Polygon([(latlon[0], latlon[1]) for latlon in pts2])

    isin = True
    for point in pts1: isin = isin and outerpoly.contains(Point(point))

    return isin

def boxfromgeojson(geojson: dict):
    ''' Takes a geojson and pretends it's a box (whether it is or not)
    and returns the min/man lat and long
    '''

    coords = geojson["coordinates"][0]
    minlon = minlat = 500
    maxlon = maxlat = -500

    for point in coords:
        minlon = point[0] if point[0] < minlon else minlon
        maxlon = point[0] if point[0] > maxlon else maxlon

        minlat = point[1] if point[1] < minlat else minlat
        maxlat = point[1] if point[1] > maxlat else maxlat

    return minlon, minlat, maxlon, maxlat

def addcloudstofilenames(filenames: list):
    '''
        adds "AOI" to filenames to represent files cropped to the area of interest
    '''
    return [
    ".".join(str(name).split(".")[:-1] + ["clouds.tiff"]) for name in filenames
    ]

def addaoitofilenames(filenames: list):
    '''
        adds "clouds" to filenames to represent cloud layers extracted from the UDM2
    '''
    return [
    ".".join(str(name).split(".")[:-1] + ["aoi.tiff"]) for name in filenames
    ]
"""
======================================================================
"""

def nextcoverset(geoAOI:dict, features: list, tolpercent = 0.7) -> int:
    '''
    returns how many of the next images to take such that the area of interest is covered completely. Theoretically, these should not span more than a day. (based on the fact that we "should" have all areas of the earth covered once per day.) but of course, we often have strips of nothing.
    '''

    covered = False
    numcov = 0

    def geoj2poly (geoj):
        '''Converts a geojson dict to a shapely.Polygon

        Note: the assumption for now is that the geometries of images,
        may have multiple polygons, but will never have negative space.
        i.e. no polygons with "holes" in them'''

        if geoj["type"] == "Polygon":
            return Polygon(
            [(latlon[0], latlon[1]) for latlon in geoj["coordinates"][0]]
            )
        elif geoj["type"] == "MultiPolygon":
            return cascaded_union([
            Polygon([(latlon[0], latlon[1]) for latlon in p[0]])
            for p in geoj["coordinates"]
            ])

    aoipoly = geoj2poly(geoAOI)
    targetarea = aoipoly.area * tolpercent
    covpoly = Polygon([])

    while not covered:
        if numcov >= len(features): return 0
        featurepoly = geoj2poly(features[numcov]["geometry"])
        covpoly = covpoly.union(featurepoly.intersection(aoipoly))
        numcov +=1
        if covpoly.area > targetarea: covered = True

    return numcov

def get_stats(filter: dict, item_types: list):
    '''
    Contact the stats db
    '''
    # Stats API request object
    stats_request = {
    "interval": "day",
    "item_types": item_types,
    "filter": filter
    }

    # fire off the POST request
    result = \
    requests.post(
    versioned_url + 'stats', #API url
    auth=auth_header,
    json=stats_request)

    return result

def search(filter: dict, item_types: list):
    '''
    Searches the planet database for certain images
    '''
    # Search API request object
    search_request = {
      "item_types": item_types,
      "filter": filter
    }

    result = \
      requests.post(
        versioned_url + 'quick-search',
        auth=auth_header,
        json=search_request)
    return result

def nextpage(currentpage: dict):
    '''
    Gets the next page of search results
    '''
    link = currentpage["_links"]["_next"]
    return requests.get(link, auth=HTTPBasicAuth(auth_key, ''))

def getassetslisting(item_id, item_type):
    '''
    Gets the assets available for a certain item
    '''
    return requests.get(
        versioned_url + 'item-types/' + item_type + '/items/' + item_id + '/assets/',
        auth=HTTPBasicAuth(auth_key, '')).json()

# "Wait 2^x * 1000 milliseconds between each retry, up to 10
# seconds, then 10 seconds afterwards"
@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000)
def activate(itemid, itemtype, assettype):
    '''
    Calls the planet API to activate the assets we want
    '''
    # setup auth
    session = requests.Session()
    session.auth = (auth_key, '')

    # request an item
    item = \
      session.get(
        ("https://api.planet.com/data/v1/item-types/" +
        "{}/items/{}/assets/").format(itemtype, itemid))

    # extract the activation url from the item for the desired asset
    item_activation_url = item.json()[assettype]["_links"]["activate"]

    # request activation until it works
    active_status = session.post(item_activation_url).status_code
    while(active_status != 204):
        # raise an exception to trigger the rate limited retry
        if active_status == 429:
            raise Exception("rate limit error")
        active_status = session.post(item_activation_url).status_code
        #activeprint(active_status)

    return active_status

def multiactivate(itemids: list, itemtypes: list, assettype: str, parallelism=5):
    '''
    Calls activate in parallel to ready multiple assets

    '''
    print("activating " + str(len(itemids)) + " items")
    activations = [(id, type, assettype) for id,type in zip(itemids,itemtypes)]

    with ThreadPool(parallelism) as pool:
        pool.starmap(activate, activations)

    print("Done!")

# "Wait 2^x * 1000 milliseconds between each retry, up to 10
# seconds, then 10 seconds afterwards"
@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000)
def downloadfile(url: str, filepath="./data", filename = "noname") -> str:
    '''
        Takes a url of a file that is assumed to be activated.

        TODO: put in error check for activated

        Downloads the file and attempts to use it's given name (from the data headers)
    '''

    #make the request
    result = requests.get(url, auth=auth_header)
    if (result.status_code == 429):
        raise Exception("rate limit error")
    #try to get filename from the header
    if "Content-Disposition" in result.headers.keys():
        params = parse_header(result.headers["Content-Disposition"])[1]
    else:
        params = {"Content-Disposition": "filename=" + str(filename)}
    if "filename" in params.keys():
        filename = params["filename"]

    #save the image in the data folder
    path = filepath + '/' + filename
    if result.status_code == 200:
        with iopen(path, 'wb') as f:
            f.write(result.content)

    return filename

def downloadasset(itemid: str, itemtype: str, assettype: str):
    # setup auth
    session = requests.Session()
    session.auth = (auth_key, '')

    # request an item
    item = \
      session.get(
        ("https://api.planet.com/data/v1/item-types/" +
        "{}/items/{}/assets/").format(itemtype, itemid))

    # extract the activation url from the item for the desired asset
    if "location" in item.json()[assettype].keys():
        download_url = item.json()[assettype]["location"]
    else:
        raise Exception("tried to download an item that was not activated")

    filename = downloadfile(download_url)

    return filename

def multidownload(itemids: list, itemtypes: list, assettype: str, parallelism=5):
    '''Downloads a list of assets in parallel in parallel
    '''

    print("downloading " + str(len(itemids)) + " items")
    downloads = [(id, type, assettype) for id,type in zip(itemids,itemtypes)]
    with ThreadPool(parallelism) as pool:
        filenames = pool.starmap(downloadasset, downloads)

    # QUICKFIX!!!!!!!!!
    # if download fails, we ignore it.
    filenames = [f for f in filenames if f is not "noname"]
    print("Done!")
    return filenames

def getudmseries(geoAOI: dict, timestart: datetime, timeend: datetime, getVisual = False):
    '''Downloads all udm files between timestart and timeend that contain completely the area described in geojson.

    Args:
    geojson: (geojson dict) describing some area of interest
    timestart: (datetime) beginning of time of interest
    timestart: (datetime) end of time of interest
    '''

    '''
        STEP 1: Search for items that contain parts of AOI
    '''
    filters = geodatefilter(geoAOI, timestart, timeend)

    '''
        !!!
        TODO: only using 4band because those are the only ones guaranteed to have UDM2 assets. Could potentially use 3band but would have to filter for ones that have UDM2 assets.
        !!!
    '''
    item_types = [
         "PSScene4Band",#"REScene", "SkySatScene", "SkySatCollect"#, "Landsat8L1G", "Sentinel2L1C"#,"PSScene3Band", "REScene", "SkySatScene", "SkySatCollect", "PSOrthoTile", "REOrthoTile"
        ]

    resultspage = search(filters, item_types).json()
    features = resultspage["features"]

    # gather all search results
    while resultspage["_links"]["_next"] != None:
        resultspage = nextpage(resultspage).json()
        # sometimes the last page of results has nothing on it
        if len(resultspage["features"]) > 0:
            features.append(resultspage["features"])

    '''
        !!!
        BIG TODO: STEP 2: bin the results chronologically such that the images in each bin cover the area of interest to some tolerance

        Right now the number of images is capped at 100 and we make no effort to ensure that each image fully captures the region we care about.

        One solution is to grab as many images as we need to fully cover the area, merge those, and then call that 1 image.

        That is the point of the `nextcoverset` function (which should work)
        !!!
    '''
    covnum = 100#nextcoverset(geoAOI, features) #TODO

    covnum = min(covnum, len(features))

    covfeatures = features[:covnum]

    for i in range(0,covnum-1): features.pop(0)

    '''
        STEP 3: activate UDM2 assets
    '''
    itemids = []
    itemtypes = []
    timestamps = []
    itemtypes_short = []

    for feat in covfeatures:
        itemids.append(feat["id"])
        type = feat["properties"]["item_type"]
        itemtypes.append(type)
        timestamps.append(feat["properties"]["acquired"])

        if type == "PSScene4Band": itemtypes_short.append("3B")
        elif type == "PSScene3Band": itemtypes_short.append("3B")
        else: itemtypes_short.append("3B")


    # work out what the filenames should be
    udmfilenames = [item + "_" + itemtypes_short[i] + "_udm2" + ".tif" for i,item in enumerate(itemids)]

    # make a binary array of which ones to download
    '''
        !!!
        TODO: the below line should make us only download ones that we haven't already, It was making the NOAA data getter act weird, so I commented it out temporarily
        !!!
    '''
    to_download = [True for file in udmfilenames ]#[not os.path.exists("./data/" + file) for file in udmfilenames]

    # activate the ones we intend to download
    multiactivate(list(compress(itemids, to_download)), list(compress(itemtypes, to_download)), "udm2")

    '''
        STEP 4: download UDMs to data folder
    '''
    multidownload(list(compress(itemids, to_download)), list(compress(itemtypes, to_download)), "udm2")

    # Get the visual images for comparison if desired
    if getVisual:
        multiactivate(list(compress(itemids, to_download)), list(compress(itemtypes, to_download)), "analytic")
        multidownload(list(compress(itemids, to_download)), list(compress(itemtypes, to_download)), "analytic")

    # vvvv uncomment for visual confirmation of coverage vvvvv
    # This just checks the geometry of collected images againts the area of interest
    '''pts1 = geoAOI["coordinates"][0]
    from matplotlib import pyplot as plt
    for pt in pts1: plt.plot(pt[0], pt[1], "bo")

    for f in covfeatures:
        for poly in f["geometry"]["coordinates"]:
            pts2 = poly[0]
            if len(pts2) < 3: continue
            for pt in pts2: plt.plot(pt[0], pt[1], "r+")

    plt.show()'''
    return itemids, timestamps, udmfilenames

def extractcloudlayers(filenames: list, dir="./data"):
    '''
        UDM2 assets have several layers for different types of
        unusable data. Extract the one that is for clouds (layer 6)
    '''
    newfiles = addcloudstofilenames(filenames)

    #layer 6 is cloud, and we want geotiff format
    translateoptions = gdal.TranslateOptions(bandList=[6], format="GTiff")

    for name, newname in zip(filenames,newfiles):
        name = dir + "/" + name
        newname = dir + "/" + newname
        gdal.Translate(newname, name, options=translateoptions)

    return newfiles

def croptoAOI(filenames: list, geoAOI: dict, dir="./data", binfactor=None):
    '''
        Takes a list of geoTiff files and crops them to fit in a geojson.
    '''
    xdim = ydim = 0
    newfiles = addaoitofilenames(filenames)

    # use, instead, the smallest box containing geoAOI
    minlon, minlat, maxlon, maxlat = boxfromgeojson(geoAOI)

    # the output pixel dimensions are sometimes slightly mismatched. So just use dimensions of the first one
    warpoptions = gdal.WarpOptions(outputBounds=(minlon, minlat, maxlon, maxlat), dstSRS="EPSG:4296")
    gdal.Warp(dir + "/" + newfiles[0], dir + "/" + filenames[0], options=warpoptions)

    raster = gdal.Open( dir + "/" + newfiles[0])
    xpix = raster.RasterXSize
    ypix = raster.RasterYSize

    # if we plan on binning the data, round the dimensions up to be a multiple of the binning factor
    # NOTE: this might crash if we try crazy huge bin factors
    if binfactor is not None:
        xpix += binfactor - (xpix % binfactor)
        ypix += binfactor - (ypix % binfactor)

    # redefine warpoptions now with a specified width/height
    warpoptions = gdal.WarpOptions(outputBounds=(minlon, minlat, maxlon, maxlat), dstSRS="EPSG:4296", width=xpix, height=ypix)

    # crop all images (redo the first one in case binning changed it)
    for name, newname in zip(filenames,newfiles):
        name = dir + "/" + name
        newname = dir + "/" + newname
        gdal.Warp(newname, name, options=warpoptions)

    return newfiles

def testtargeted():
    '''
        Test for gtetting a single picture
    '''
    geoAOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [
              -90.2914810180664,
              35.54256307316442
            ],
            [
              -90.19054412841797,
              35.54256307316442
            ],
            [
              -90.19054412841797,
              35.60176471569719
            ],
            [
              -90.2914810180664,
              35.60176471569719
            ],
            [
              -90.2914810180664,
              35.54256307316442
            ]
        ]
      ]
    }

    dtstart = datetime(2019, 8, 3)
    dtend = datetime(2019, 8, 5)

    # these are how you actually fetch the files and ids
    #itemids, timestamps, filenames = getudmseries(geoAOI, dtstart, dtend)
    #extractcloudlayers(filenames)
    filenames=[
    "20190803_151610_1050_3B_AnalyticMS.tif", "20190804_161356_0e16_3B_AnalyticMS.tif",
    "20190803_162048_1004_3B_AnalyticMS.tif", "20190804_161357_0e16_3B_AnalyticMS.tif", "20190804_151702_0f36_3B_AnalyticMS.tif", "20190804_163752_61_1059_3B_AnalyticMS.tif",
    "20190804_151703_0f36_3B_AnalyticMS.tif"
    ]

    croptoAOI(filenames, geoAOI)

    timestamps = ['2019-08-04T16:37:52.617715Z', '2019-08-04T16:37:52.617715Z']
    itemids = ['2576796_1555525_2019-08-04_1059', '20190804_163752_61_1059']
    #filenames = ['20190804_163752_61_1059_3B_udm2.clouds.tiff',  '2576796_1555525_2019-08-04_1059_udm2.clouds.tiff']


    return itemids, timestamps, filenames

def timeseriescov(lookback_s: int, geojson: dict) -> list:
    '''
        Gets the cloud coverage percent for `lookback_s` seconds in the past
    '''
    now = datetime.utcnow()
    past = now - timedelta(seconds=lookback_s)

    geo_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geojson
    }

    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": past.isoformat() + 'Z',
        "lte": now.isoformat() + 'Z'
      }
    }

    item_types = ["PSScene3Band"]

    filter = {
        "type":"AndFilter",
        "config": [geo_filter, date_range_filter]
    }

    resultsjson = search(filter, item_types).json()

    itemtypes = []
    itemids = []
    assettypes = []
    cloudcov = []

    for feature in resultsjson["features"]:
        #TODO for getting UDMs later
        #itemtypes.push(feature["properties"]["item_type"])
        #itemids.push(feature["id"])
        #assettypes.push()
        props = feature["properties"]
        cloudcov.append(
            tuple([props["cloud_cover"], ISO8601Z_to_datetime(props["acquired"])])
        )

    # DEBUG save the json as debug.json
    jsonfile(resultsjson)

    # sort by date (second entry)
    cloudcov.sort(key=lambda tup: tup[1])


    return cloudcov

'''
    gets the cloud coverage percent for the last three days
'''
def threedaycov(geojson: dict) -> list:

    day2s = 24*60*60
    return timeseriescov(3*day2s, geojson)

def test3d():
    geojson = {
    "type": "Polygon",
    "coordinates": [
        [
            [
              -90.2914810180664,
              35.54256307316442
            ],
            [
              -90.19054412841797,
              35.54256307316442
            ],
            [
              -90.19054412841797,
              35.60176471569719
            ],
            [
              -90.2914810180664,
              35.60176471569719
            ],
            [
              -90.2914810180664,
              35.54256307316442
            ]
        ]
      ]
    }

    print(threedaycov(geojson))

def getlatestudm(geojson:dict):
    '''
        downloads the latest udm available for some area
    '''
    day2s = 24*60*60
    lookback_s = 10*day2s #"enough" time to find an image

    now = datetime.utcnow()
    past = now - timedelta(seconds=lookback_s)

    # filter for location and time
    filter = geodatefilter(geojson, past, now)
    item_types = ["PSScene4Band"]

    resultjson = search(filter, item_types).json()

    item_id = resultjson["features"][0]["id"]
    item_type = resultjson["features"][0]["properties"]["item_type"]
    asset_type = "udm2"

    timestamp = resultjson["features"][0]["properties"]["acquired"]

    # activate the asset and download
    activate(item_id, item_type, asset_type)
    filepath = downloadasset(item_id, item_type, asset_type)

    return filepath, timestamp
