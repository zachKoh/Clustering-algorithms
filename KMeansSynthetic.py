import numpy as np
from KMeans import *

def createSyntheticDataset(data):
    numDataPoints = len(data)
    numFeatures = data.shape[1]

    means = np.zeros(numFeatures)
    standardDeviations = np.ones(numFeatures)

    synthetic_data = np.random.normal(loc=means, scale=standardDeviations, size=(numDataPoints, numFeatures))

    return synthetic_data

def main():
    data = loadDataset()
    synthetic_dataset = createSyntheticDataset(data)
    plotSilhouette(synthetic_dataset,maxIter=50)

main()