import numpy as np
from KMeans import *

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

def plotSilhouettePlusPlus(data,maxIter):
    kValues = range(2, 10)
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


def main():
    data = loadDataset()
    plotSilhouette(data,maxIter=50)


main()