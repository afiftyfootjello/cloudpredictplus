#!/usr/bin/python3

'''
    Main file for cloud cover predictions leveraging planet data.

    See `using_cloud_predict.ipynb` for examples

    @author: Kyle Cochran
'''

from PDEFIND import PDE_FIND as pdef
import numpy as np
import time
import datetime
import planet.missions.feasibility.managers.clouds as noaai
import planetingress as plani
from osgeo import gdal
from itertools import product


'''
    ======================== HELPER FUNCTIONS ============================
'''

def geocenter(geojson: dict):
    '''
        gets the geometric center of a geojson area as a lon, lat tuple. Just an average.
    '''
    coords = geojson["coordinates"][0]
    avg = [0,0]
    for lonlat in coords:
        avg[0] += lonlat[0]/len(coords)
        avg[1] += lonlat[1]/len(coords)
    return tuple(avg)


def rebin(arr, new_shape):
    """
        Performs a binning operation on arr such that it acquires shape new_shape
    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def mat_layer_average(matrix3d):
    """
        Averages layers of a 3d matrix, where layers are along the 3rd axis. Returns a list of each layer average
    """
    n_times = matrix3d.shape[2]
    avgs = []
    for t in range(n_times):
        slice = matrix3d[:,:,t]

        avg = np.mean(np.asarray(
            [
                np.mean(arr) for arr in slice
            ]
        ))
        avgs.append(avg)

    return avgs

def printPDE(c, description, ut = 'cov_t'):
    # Simple access wrapper
    pdef.print_pde(c, description, ut = ut)

'''
    =================== NUMERICAL FUNCTIONS =======================
'''


def findiff(dataset: np.ndarray, tgrid: np.ndarray) -> np.ndarray:
    '''
        wrapper function for Numpy gradient.

        Central differencing to obtain time gradient info.

        in: numpy array where col1 is time and col2 is val
        out: numpy array where col1 is time and col2 is gradient
    '''
    f = np.gradient(dataset, tgrid)
    return f

def findPDE(geoaoi: dict, dtstart: datetime, dtend: datetime):
    '''
        This is the meat of the PDE finding method. Does a few things:

        - retrieves planet data
        - processes it into a matrix
        - takes all of the first order derivatives
        - Gets the NOAA data
        - Arranges all data into PDEFIND format
        - Calls PDEFIND
        - Saves found PDE
    '''
    # find relevant images and download them
    itemids, ISOtimes, filenames = plani.getudmseries(geoaoi, dtstart, dtend)

    #extract cloud layers, clip them, and produce a raster stack
    cov3d = processplanetdata(geoaoi, itemids, ISOtimes, filenames, binfactor=32)

    # time grid is normalized unix stamps
    dts = [plani.ISO8601Z_to_datetime(t) for t in ISOtimes]
    timestamps = [plani.ISO8601Z_to_datetime(t).timestamp() for t in ISOtimes]
    maxt = max(timestamps)

    # make the grid on which to evaluate derivatives
    nx = cov3d.shape[1]
    ny = cov3d.shape[0]
    xgrid = np.asarray(range(0, nx))
    ygrid = np.asarray(range(0, ny))
    tgrid = np.asarray([t/maxt for t in timestamps])

    ddx, ddy, ddt = evaluatederivatives(cov3d, xgrid, ygrid, tgrid=tgrid)

    # Build vector of NOAA data
    # Assumption here is that the predict is the same for every pixel
    lon, lat = geocenter(geoaoi)
    #noaacov = getnoaadata([lat], [lon], dts)
    Q = []
    #for i in range(0, nx*ny): Q+=noaacov
    Q = [5]*nx*ny
    Q = np.asarray(Q)
    Q.shape += (1,)

    '''
        !!
        TODO: NOAA data is not being included in the PDE model at the moment because the NOAA data retrieval code wasn't working too well.

        To include NOAA data, put it in the array Q and hstack it onto U. See PDEFIND documentation for more info on this.
        !!
    '''
    # Stack 3d data into 1d
    U = stackdata(cov3d)
    U.reshape((1, U.size))
    U.shape += (1,)

    Ux = stackdata(ddx)
    Uy = stackdata(ddy)
    Ut = stackdata(ddt)
    Ux.shape += (1,)
    Uy.shape += (1,)
    Ut.shape += (1,)

    derivs = np.hstack([np.ones((U.size, 1)), Ux, Uy])
#    UQ = np.hstack([U,Q])

    dependentvars = ["U", "P"]
    independentvars = ["x", "y", "t"]
    deriv_descrip = ['', "Ux", "Uy"]

    maxpolyorder = 3

    Theta, description = pdef.build_Theta(U, derivs, deriv_descrip, maxpolyorder, data_description = ["U"])

    print('Candidate terms for PDE')
    print(['1']+description[1:])

    lam = 10**-5
    d_tol = 10
    a,b = Theta.shape
    c = pdef.TrainSTRidge(Theta,Ut,lam,d_tol, normalize=0)

    savePDE(c, dependentvars, independentvars, description, min(dts), max(dts), 0, 0, 90, 90)

    return c, description

def getinitialconditions(geoaoi):
    '''
        This function produces initial conditions for a PDE simulation by pulling down the latest cloud coverage data possible and generating derivatives from that.

        TODO: This is still a prototype and has not really been tested.
    '''
    # get the latest data possible so we can derive some initial conditions
    latestfile, timestamp = plani.getlatestudm(geoaoi)
    latestfile = [latestfile]
    #latestfile = ['20190811_183630_0f22_3B_udm2.tif']
    cloudlayer = plani.extractcloudlayers(latestfile)
    reduxcloudlayer = plani.croptoAOI(cloudlayer, geoaoi)
    pathname = ["./data/"+file for file in reduxcloudlayer]

    IC = np.asarray(noaai.CloudForecastGFS.stack_rasters(pathname), dtype="float")
    xgrid = np.asarray(range(0,IC.shape[1]))
    ygrid = np.asarray(range(0,IC.shape[0]))

    ddxIC, ddyIC, _ = evaluatederivatives(IC, xgrid, ygrid)
    print(timestamp)
    return IC, ddxIC, ddyIC, timestamp

def predictPDE(geoaoi: dict, prediction_time, model_location):
    '''
    Uses generated PDE to predict the future (weather)
    '''

    # pull down the very latest cover info to use as init conditions
    #IC, ddxIC, ddyIC, time = getinitialconditions(geoaoi)
    pass

def savePDE(coeffs, depvars, indepvars, descrips, gentimestart, gentimeend, lonmin, latmin, lonmax, latmax):
    '''
        Saves a generated PDE into a json file

        Args:
            coeffs: (list of coefficients) to terms in descrips
            descrips: (list of strings) all possible variable combinations

        TODO: (in the far future) use lonlat information to put searchable location information in the filename. So if we want to figure out if we have a model for some location, we can search it.
    '''

    # What we've called dependent and independent variables in the system
    #depvars = ['U', 'Q']
    #indepvars = ['x', 'y', 't']

    #descrips = ["UxxQyUxyt", "UUQt", "UtUtUtQx", "QQ", ""]
    #coeffs = [2.5, 3.5, 4, 0, 5.5]

    minlon = 0
    minlat = 0
    maxlon = 90
    maxlat = 90

    #gentimestart = datetime.datetime(2019, 8, 3)
    #gentimeend = datetime.datetime(2019, 8, 4)

    # This chunk is to parse carat (^) signs and turn them into multiplication
    clean_descrips = []
    for termstr in descrips:
        #loop through each PDE term
        newtermstr = ""
        varstr = ""
        i = 0
        while i < len(termstr):
            # Loop through each symbol
            c = termstr[i]
            if c is "^":
                # if we find a carat, look for numbers after it and add that
                # many of the previous symbol to the term string
                numstr = ""
                i += 1
                c = termstr[i]
                # look for all digits following ^
                while c.isdigit():
                    numstr += c
                    i += 1
                    if i >= len(termstr):break
                    c = termstr[i]
                power = int(numstr)
                # we already added the symbol once, add what's left
                for j in range(0, power - 1):
                    newtermstr += varstr
            elif c.islower():
                # derivative symbol attached to a variable
                varstr += c
                i+=1
            elif c.isupper():
                # start of a new variable
                newtermstr += varstr
                varstr = c
                i+=1
            elif c.isdigit():
                # we made it to a digit that doesn't follow a ^. Must be a const
                newtermstr += c
                i+=1
            else:
                #junk?
                i+=1

        newtermstr += varstr
        clean_descrips.append(newtermstr)

    # how many entries we need to fully describe each variable
    varlen = len(depvars) + len(indepvars)

    pde = {}
    terms = []
    import IPython as ip
    nonzerocoeffs = [c for c in coeffs if c > 0]
    # loops through possible terms
    for i, termstr in enumerate(clean_descrips):
        if coeffs[i] > 0:

            # new term
            terms.append([])

            # if empty string
            if termstr.isdigit(): terms[-1].append([0,]*varlen)

            # Otherwise run through the letters
            for var in termstr:
                if var.isupper():

                    # new variable
                    terms[-1].append([0,]*varlen)
                    # add Dependent Variable to the new var tuple
                    terms[-1][-1][depvars.index(var)] += 1
                elif var.islower():
                    # add Independent Variable to the new var tuple
                    terms[-1][-1][len(depvars) + indepvars.index(var)] += 1

    pde["rhs_terms"] = terms
    pde["coeffs"] = nonzerocoeffs
    pde["legend"] = {}
    pde["legend"]["dependent_variables"] = depvars
    pde["legend"]["independent_variables"] = indepvars
    pde["location"] = {}

    pde["location"]["minlon"] = minlon
    pde["location"]["minlat"] = minlat
    pde["location"]["maxlon"] = maxlon
    pde["location"]["maxlat"] = maxlat

    pde["time"] = {}
    pde["time"]["updated"] = datetime.datetime.utcnow().isoformat() + "Z"
    pde["time"]["based_on_begin"] = gentimestart.isoformat() + "Z"
    pde["time"]["based_on_end"] = gentimeend.isoformat() + "Z"

    gt = gentimeend
    sep = "_"

    '''
        TODO: use lonlat information to put searchable location information in the filename. So if we want to figure out if we have a model for some location, we can search it.
    '''
    filename = ("{}" + sep + "{}" + sep + "{}" + sep + "{}" + sep + "sf.json").format(gt.year, gt.month, gt.day, gt.hour)

    filepath = "./data/" + filename
    import json
    with open(filepath, 'w') as f:
        json.dump(pde, f)

def loadPDE(geojson: dict):
    '''
        Given some AOI, see if we have a weather model that covers it and load it.

        Meant to be called inside of predictPDE
    '''
    pass

def getplanetdata(lookbackd: float, geojson: dict):
    '''
        --DEPRECATED--: getudmseries is used instead

    Gets the data we need from Planet and formats it into a numpy array
    '''
    day2s = 24*60*60
    #gets a sorted list of tuples
    covseries = plani.timeseriescov(lookbackd*day2s, geojson)

    #convert times from UTC string to unixstamp
    #unixstamp = lambda str: datetime.datetime.strptime(str, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()

    # build data array
    data = np.zeros((len(covseries), 2))
    for i,datum in enumerate(covseries):
        data[i, 0] = datum[1].timestamp() # unix stamp
        data[i, 1] = datum[0] # cov percent

    return data

def processplanetdata(geoAOI, itemids, timestamps, filenames, binfactor=None):
    '''
        Given a list of downloaded UDM2 files, produced a cloud cover timeseries
        array as a 3d numpy array of ones and zeros.

        Args:
            geoAOI: (a dict) some geojson of the area we care about. NOTE: only one polygon supported currently

            filenames: (a list) of filenames. Should always be UDM2 files

            binfactor: (an integer) x,y binning of each image
    '''


    # pull out cloud layer from UDMs
    cloudlayers = plani.extractcloudlayers(filenames)

    # Crop the images to only include what we care about
    reduxcloudlayers = plani.croptoAOI(cloudlayers, geoAOI, binfactor=binfactor)

    # Process the reduced cloud cover files into a stacked array (of presumably ones and zeros for cover/no cover)
    pathnames = ["./data/"+file for file in reduxcloudlayers]
    stacked_rasters = np.asarray(noaai.CloudForecastGFS.stack_rasters(pathnames), dtype="float")

    # Do some binning to make the matrix less unweildy
    if binfactor is not None:
        # original dimensions
        ny, nx, nz=stacked_rasters.shape

        # new dimensions after binning
        bnx, bny = nx//binfactor, ny//binfactor
        covdata = np.zeros((bny, bnx, nz))

        # go through each time layer and apply binning
        for i in range(0, nz):
            covdata[:,:,i] = rebin(stacked_rasters[:,:,i], (bny, bnx))
    else:
        covdata = stacked_rasters

    return covdata

def stackdata(U3d):
    '''
        Takes a 3d numpy array and turns it into a 1d array useful for PDE-FIND.

        Arranges data in stack-major, and then row-major form

        x = which column of pixel
        y = which row of pixel
        z = which frame in stack

        s.t. [[[1,2], [3,4]], [[5,6], [7,8]]]
        becomes [1, 2, 5, 6, 3, 4, 7, 8]
    '''
    U3d = np.asarray(U3d)
    U = []

    for x in range(U3d.shape[1]):
        for y in range(U3d.shape[0]):
            for z in range(U3d.shape[2]):
                U.append(U3d[y, x, z])


    return np.asarray(U)

def evaluatederivatives(data, xgrid, ygrid, tgrid=None):
    '''
        takes derivatives of 3d data (e.g. a raster stack) in time and space.

        This is used to generate derivative vector fields for the PDE finding method.

        Args:
        data: (3d ndarray) containing the data source (coverage data)
        xgrid: (1d ndarray) containing x discretization, spatial
        ygrid: (1d ndarray) containing y discretization, spatial
        tgrind: (1d ndarray) containing t discretization, temporal

        Out:
        ddx: 1st derivative of data with respect to x
        ddy: 1st derivative of data with respect to y
        ddt: 1st derivative of data with respect to t
    '''

    ny, nx, nt = data.shape

    # only take time derivatives if a grid is supplied
    ddt = np.zeros(data.shape)
    if tgrid is not None:
        for ix, iy in product(range(0,nx), range(0,ny)):
            ddt[iy, ix, :] = findiff(data[iy, ix, :], tgrid)


    # y derivatives
    ddy = np.zeros(data.shape)
    for ix, it in product(range(0,nx), range(0,nt)):
        ddy[:, ix, it] = findiff(data[:, ix, it], ygrid)


    # x derivatives
    ddx = np.zeros(data.shape, dtype=float)

    for it, iy in product(range(0, nt), range(0, ny)):
        ddx[iy, :, it] = findiff(data[iy, :, it], xgrid)

    return ddx, ddy, ddt

def getnoaadatum(lat:float, lon:float, time, preferred_lead_time=datetime.timedelta(hours=12)):
    '''
        Gets a single NOAA prediction using the clouds.py functions in the Planet feasibility code.

        Args:
            lat: float, the latitude
            lon: float, the longitude
            time: datetime, the time that you're making a prediction for
            preferred_lead_time: timedelta, from when you want to make the prediction

        Out:
            percent: float, the predicted cloudiness percent
            lead_time: float, the _actual_ lead time of the prediction
    '''

    '''
        !!!!!!
        TODO: The only way I was able to get this code to work was to enforce a minimum 12 hour lead time.... This should not be necessary because NOAA provides new predicts every 3 hours. Fix.
        !!!!!!
    '''

    # number of hours rounded up, 100 max
    dur = min(100, int((preferred_lead_time.total_seconds() // 3600) + 1))

    # Which forecasting file to go back and get
    run_time = time-datetime.timedelta(hours=dur)

    # we only pick up prediction files every three hours. So we only try to get ones generated more than three hours ago
    if run_time > datetime.datetime.now() - datetime.timedelta(hours=3):
        run_time = datetime.datetime.now() - datetime.timedelta(hours=3)
        # adjust duration
        dur = int((time - run_time).total_seconds() // 3600)

    # round up to the nearest multiple of 3
    dur += dur % 3 + 12

    #test_run_time = datetime.datetime(2019, 9, 3, 17, 20, 14, 310405)
    #test_run_time = datetime.datetime(2019, 9, 3,  6, 35, 16, 508236)
    #test_dur = 20
    #test_time     = datetime.datetime(2019, 9, 3, 18, 35, 16, 508236)

    # call the predicter from feasibility repo (in clouds.py)
    dailypredicter = noaai.CloudForecastGFS(forecast_run_time=run_time, forecast_duration_hr=dur)

    # Find the forecast at the actual desired time
    percent, lead_time = dailypredicter.compute_cloud_forecast([lat], [lon], [time])

    # convert to [0,1]
    percent /= 100

    return percent, lead_time

def predict_linear(where: dict, when: datetime.datetime, history_days = 5) -> float:
    '''
        Uses a first-order corrective approximation to predict into the future.
        That means two things:
            - looks at change in bias of a noaa prediction over time for an entire region and linearly extrapolates that into the future
            - looks at bias of certain subareas of a region and linearly interpolates that into the future


        Args:
            where: a (dict) holding a geojson
            when: a (datetime) of the desired prediction time
            history_days: (int) how many days in the past to use for trend finding
    '''

    '''
        PART 1: wholesale correction of NOAA predict
    '''
    # find two places with Planet imagery. One with the latest imagery and one from history_days in the past.

    dtend = datetime.datetime.now()

    if when < dtend: raise Exception("ERROR: the date to predict must be in the future")
    dtstart = dtend - datetime.timedelta(days=history_days)

    # Retrieve the Planet data
    print("getting planet data")
    itemids, ISOtimes, filenames = plani.getudmseries(where, dtstart, dtend)

    print("Evaluating NOAA bias drift")
    # we only want two of the images
    dt_times = sorted([plani.ISO8601Z_to_datetime(time) for time in ISOtimes])
    filenames = [ x for _,x in sorted(zip(dt_times, filenames))]
    itemids = [ x for _,x in sorted(zip(dt_times, itemids))]
    ISOtimes = [ x for _,x in sorted(zip(dt_times, ISOtimes))]

    # Turn planet data into a matrix, reduce it a bit via image binning
    cov3d = processplanetdata(where, [itemids[0], itemids[-1]], [ISOtimes[0], ISOtimes[-1]], [filenames[0],filenames[-1]],binfactor=64)

    # the data we get from planet can be considered the truth model
    truth_cov = mat_layer_average(cov3d)

    # get the corresponding NOAA data
    # ------------------------------------------
    lon, lat = geocenter(where) # average over the polygon

    noaa_predict_init, lead_time_init = getnoaadatum(lat, lon, dt_times[-1], preferred_lead_time = dt_times[-1] - dt_times[0])

    noaa_predict_final, lead_time_final = getnoaadatum(lat, lon, dt_times[-1])

    # 1st order model of NOAA correction
    init_noaa_error = noaa_predict_init - truth_cov[0]
    final_noaa_error = noaa_predict_final - truth_cov[1]
    change_in_predict_time = lead_time_init - lead_time_final

    noaa_correction_rate = (final_noaa_error - init_noaa_error)/change_in_predict_time

    # ask NOAA for a prediction of the point in the future that we actually care about
    noaa_predict, lead_time = getnoaadatum(lat, lon, when)


    corrected_predict = noaa_predict + lead_time * noaa_correction_rate

    # clamp between 0 and 1
    corrected_predict = min(1, max(0, corrected_predict))

    '''
        PART 2: regional biases of NOAA data
    '''
    print("evaluating local biases and creating map")
    # this time process _all_ planet images we pulled (not just first and last)
    bias = processplanetdata(where, itemids, ISOtimes, filenames, binfactor=256)

    # get a NOAA predict to match every point of Planet data that we have
    noaa_predicts = [
        getnoaadatum(lat,lon,t)[0] for t in dt_times
    ]

    # Figure out how incorrect the NOAA predict was for each pixel
    for i in range(bias.shape[2]):
        bias[:,:,i] -= noaa_predicts[i]

    # average over time, gives a 2d matrix
    avg_bias = np.mean(bias, axis=2)

    # make a map of prediction over the region
    predict_map = corrected_predict + avg_bias

    # these prints should be commented out later
    print("original predict", noaa_predict)

    #clamp it
    for i,a in enumerate(predict_map):
        for j,b in enumerate(a):
            predict_map[i][j] = min(1, max(0, b))

    # round it off a bit so it's easier to look at
    corrected_predict = np.around(corrected_predict, decimals=3)
    predict_map = np.around(predict_map, decimals=3)

    return corrected_predict, predict_map

def main():
    '''
        Method used for testing
    '''

    # Where do we want to predict?
    # SF bay area. 10 km square
    geoaoi = {
            "type": "Polygon",
            "coordinates": [
              [
                [
                  -122.48485565185547,
                  37.74004179435127
                ],
                [
                  -122.43232727050781,
                  37.74004179435127
                ],
                [
                  -122.43232727050781,
                  37.77695634643178
                ],
                [
                  -122.48485565185547,
                  37.77695634643178
                ],
                [
                  -122.48485565185547,
                  37.74004179435127
                ]
              ]
            ]
          }

    # When do we want to predict?
    prediction_time = datetime.datetime(2019, 9, 8)

    '''
        Linear prediction method
    '''

    # lookback for linear prediction
    hist_days = 5

    # do prediction
    predict, predict_map = predict_linear(geoaoi, prediction_time, history_days = hist_days)

    # Save the prediction map
    p_time_str = plani.datetime_to_ISO8601Z(prediction_time)
    now = plani.datetime_to_ISO8601Z(datetime.datetime.now())
    filename = p_time_str + "_" + now + "_" + str(hist_days)

    testfolder = "./tests/"
    np.save(testfolder + filename, predict_map)

    print("corrected total predict: ", predict)
    print("predict map with local corrections saved in:", testfolder, "with the format: <predict_time>_<run_time>_<hist_days>.npy")

    print("use the plotmat.py `plot` function to show the map.")

    '''
        Uncomment to run the PDE finding method
    '''
    #dtstart = datetime.datetime(2019, 8, 11)
    #dtend = datetime.datetime(2019, 8, 18)
    #c, description = findPDE(geoaoi, dtstart, dtend)
    #pdef.print_pde(c, description, ut = 'cov_t')
    '''
        ------------------------------------------
    '''

