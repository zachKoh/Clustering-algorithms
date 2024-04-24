import numpy as np
from KMeans import *


#Description: Creates a synthetic dataset with the same shape as the input data.

#Input
#data: numpy array, original dataset used to determine the shape of the synthetic dataset
#seed: int, seed value for random number generation
#
#Output
#synthetic_data: numpy array, synthetic dataset with the same shape as the input data
def createSyntheticDataset(data, seed):
    numDataPoints = len(data)
    numFeatures = data.shape[1]

    means = np.zeros(numFeatures)
    standardDeviations = np.ones(numFeatures)

    np.random.seed(seed)
    synthetic_data = np.random.normal(loc=means, scale=standardDeviations, size=(numDataPoints, numFeatures))

    return synthetic_data


#Description: Main function to generate a synthetic dataset and plot Silhouette coefficients.
def main():
    data = loadDataset()
    synthetic_dataset = createSyntheticDataset(data, 2169)
    plotSilhouette(synthetic_dataset,maxIter=50)

main()