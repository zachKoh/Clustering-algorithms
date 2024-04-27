import numpy as np
import matplotlib.pyplot as plt
from KMeans import loadDataset, initialization, computeClusterRepresentatives, assignClusterIDs


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
#clusters: A numpy array containing cluster assignments for each data point
#data: A numpy array of the dataset

#Output:
#score: A float which is the Silhouette coefficient

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
#data: A numpy array containing the dataset
#k: An integer containing the number of clusters

#Output
#clusters: A numpy array containing cluster assignments for each data point.

#Description: Performs Bisecting k-Means algorithm to partition the dataset into k clusters.
'''
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
'''


def bisectingKMeans(data, k):
    # Initialize a tree with a single root node containing the entire dataset
    tree = [data]
    
    # Repeat until the number of leaf clusters in the tree is equal to k
    while len(tree) < k:
        # Select a leaf node with the largest sum of squared distances
        node_idx = selectNode(tree)
        node_data = tree[node_idx]
        
        # Split the selected node into two clusters using K-Means algorithm
        cluster1, cluster2 = kMeans(node_data, 2, 50)
        
        # Add both resulting clusters as new leaf nodes to the tree
        tree.pop(node_idx)  # Remove the original node
        tree.append(cluster1)
        tree.append(cluster2)
    
    # Assign cluster labels to each data point based on the tree structure
    clusters = assignClusterLabels(data, tree)
    
    return clusters




def selectNode(tree):
    # Find the leaf node with the largest sum of squared distances
    max_wcss = float('-inf')
    max_idx = -1
    for i, node in enumerate(tree):
        wcss = computeSumfSquare(node)
        if wcss > max_wcss:
            max_wcss = wcss
            max_idx = i
    return max_idx


def assignClusterLabels(data, tree):
    clusters = np.zeros(len(data), dtype=int)
    for i, point in enumerate(data):
        for j, cluster_data in enumerate(tree):
            if np.array_equal(point, cluster_data.mean(axis=0)):
                clusters[i] = j
                break
    return clusters


def kMeans(dataset, k, maxIter):
    centroids = initialization(dataset, k, 21)
    prevCentroids = None
    iterCount = 0
    while not np.array_equal(centroids, prevCentroids) and iterCount < maxIter:
        clusters = assignClusterIDs(dataset, k, centroids)
        # Update centroids
        prevCentroids = centroids
        centroids = computeClusterRepresentatives(dataset, clusters, k)
        iterCount += 1
    
    # After convergence, compute the sum of squared distances for each cluster
    sum_of_square_distances = [computeSumfSquare(dataset[clusters == i]) for i in range(k)]

    # Sort the clusters based on their sum of squared distances
    sorted_clusters = np.argsort(sum_of_square_distances)

    # Return the two clusters with the highest and second highest sum of squared distances
    cluster1 = dataset[clusters == sorted_clusters[-1]]
    cluster2 = dataset[clusters == sorted_clusters[-2]]

    return cluster1, cluster2








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