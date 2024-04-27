import numpy as np
from KMeans import *


#Input
#data: numpy array, dataset containing data points
#k: int, number of centroids (clusters)
#seed: int, seed value for random number generation

#Output
#centroids: numpy array, initialized centroids

#Description: Initializes centroids using the K-means++ algorithm.

def kmeansPlusPlusInit(data, k, seed):
    np.random.seed(seed)
    # Choose the first centroid randomly from data points
    centroids = [data[np.random.choice(len(data))]]

    # Choose the remaining centroids using K-means++ initialization
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumProbabilities = probabilities.cumsum()
        randVal = np.random.rand()
        newCentroidIndex = np.searchsorted(cumProbabilities, randVal)
        centroids.append(data[newCentroidIndex])

    return np.array(centroids)



#Input
#dataset: numpy array, dataset containing data points
#k: int, number of centroids (clusters)
#maxIter: int, maximum number of iterations

#Output
#clusters: numpy array, cluster assignments for each data point

#Description: Performs the k-means algorithm using K-means++ initialization.

def kMeansPlusPlus(dataset, k, maxIter):
    centroids = kmeansPlusPlusInit(dataset, k, seed=21)
    prevCentroids = None
    iter = 0
    while not np.array_equal(centroids, prevCentroids) and iter < maxIter:
        clusters = assignClusterIDs(dataset, k, centroids)
        # Update centroids
        prevCentroids = centroids
        centroids = computeClusterRepresentatives(dataset, clusters, k)
        iter += 1
    
    return clusters



#Input:
#data: numpy array, dataset containing data points
#maxIter: int, maximum number of iterations for kMeansPlusPlus algorithm

#Output:
#None

#Description: Plots the Silhouette coefficients for varied values of k using K-means++ initialization.

def plotSilhouettePlusPlus(data,maxIter):
    kValues = range(1, 10)
    silhouetteCoefficients = []

    for k in kValues:
        clusters = kMeansPlusPlus(data,k,maxIter)
        silhouetteCoefficients.append(computeSilhouette(clusters,data))  
    
    #print(silhouetteCoefficients)

    plt.plot(kValues, silhouetteCoefficients, marker='o')
    plt.title('Silhouette Coefficients for Varied Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(kValues)
    plt.grid(True)
    plt.show()
    return


#Description: Main function to generate a synthetic dataset and plot Silhouette coefficients using K-means++ initialization.
def main():
    data = loadDataset()
    plotSilhouettePlusPlus(data,maxIter=50)


main()