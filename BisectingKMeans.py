import numpy as np
import matplotlib.pyplot as plt
from KMeans import *


#Input:
#clusterData: A numpy array that contains data points belonging to a single cluster

#Output:
#sumSquareDistances: A float which is the sum of squared distances within the cluster

#Description: Computes the sum of squared distances within a cluster.

def computeSumfSquare(clusterData):
    centroid = np.mean(clusterData, axis=0)
    sumSquareDistances = np.sum(np.linalg.norm(clusterData - centroid, axis=1)**2)
    return sumSquareDistances



#Input:
#clusters: numpy array, cluster assignments for each data point
#data: numpy array, dataset

#Output:
#score: float, Silhouette coefficient

#Description:
#Computes the Silhouette coefficient for a given clustering.

def computeSilhouette(clusters, data):
    numOfObjects = len(data)
    silhouetteValues = []
    for i in range(numOfObjects):
        clusterIndex = clusters[i]
        clusterData = data[clusters == clusterIndex]
        intraClusterDistance = 0
        if len(clusterData) > 1:
            intraClusterDistance = computeSumfSquare(clusterData) / len(clusterData)
        interClusterDistances = [computeSumfSquare(data[clusters == otherCluster]) for otherCluster in np.unique(clusters) if otherCluster != clusterIndex]
        minInterClusterDistance = min(interClusterDistances) if interClusterDistances else 0
        silhouetteValue = (minInterClusterDistance - intraClusterDistance) / max(minInterClusterDistance, intraClusterDistance)
        silhouetteValues.append(silhouetteValue)
    score = np.mean(silhouetteValues)
    return score




#Input
#data: a numpy array containing the dataset
#k: and integer containing the number of clusters

#Output
#clusters: A numpy array containing cluster assignments for each data point.

#Description: Performs Bisecting k-Means algorithm to partition the dataset into k clusters.

def bisectingKMeans(data, k):
    clusters = np.zeros(len(data), dtype=int)
    while len(np.unique(clusters)) < k:
        # Select the cluster to split
        clusterToSplit = np.argmax(np.bincount(clusters))
        # Split that selected cluster
        clusterData = data[clusters == clusterToSplit]
        subClusters = kMeans(clusterData, 2, 50)
        subClusters[subClusters == 1] += max(clusters) + 1
        clusters[clusters == clusterToSplit] = subClusters
    return clusters



# Main function to compute Silhouette coefficient for different numbers of clusters and plot the results.
def main():
    data = loadDataset()
    kValues = range(1, 10)
    silhouetteCoefficients = []

    for k in kValues:
        clusters = bisectingKMeans(data, k)
        silhouetteCoefficients.append(computeSilhouette(clusters, data))

    plt.plot(kValues, silhouetteCoefficients, marker='o')
    plt.title('Silhouette Coefficients for Varied Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(kValues)
    plt.grid(True)
    plt.show()

main()