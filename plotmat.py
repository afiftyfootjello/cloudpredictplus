
import matplotlib.pyplot as plt
import numpy as np

def plot(file):
    arr = np.load(file)
    fig, ax = plt.subplots()

    ax.matshow(arr, cmap=plt.cm.Blues)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            c = arr[i,j]
            ax.text(j, i, str(c), va='center', ha='center')

    fig.show()
