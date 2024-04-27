import numpy as np
from KMeans import *


#Description: Creates a synthetic dataset with the same shape as the input data.

#Input
#data: numpy array, original dataset used to determine the shape of the synthetic dataset
#seed: int, seed value for random number generation
#
#Output
#synthetic_data: numpy array, synthetic dataset with the same shape as the input data

def getSyntheticData(dataset, seed):
    std = np.std(dataset, axis = 0)
    mean = np.mean(dataset, axis = 0)
    np.random.seed(seed)
    syntheticData = np.random.normal(mean, std, size = dataset.shape)
    return syntheticData


#Description: Main function to generate a synthetic dataset and plot Silhouette coefficients.
def main():
    data = loadDataset()
    synthetic_dataset = getSyntheticData(data,42)
    plotSilhouette(synthetic_dataset,maxIter=10)

main()