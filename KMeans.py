import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def loadDataset():
    # Load the dataset
    with open("dataset") as file:
        lines = file.readlines()


    # Extract data and labels
    data = []
    labels = []
    for line in lines:
        parts = line.split()
        labels.append(parts[0])
        data.append([float(x) for x in parts[1:]])

    data = np.array(data)
    return data



# Computes euclidean distance between two points
def computeDistance(x,y):
    #Return the Euclidean distance between x and y
    return np.linalg.norm(x-y)


# Generates the initial centroids
def initialization(data,k,seed):
    # Get the minimum and maximum values for each feature using seed value
    np.random.seed(seed)
    minValues = np.min(data, axis=0)
    maxValues = np.max(data, axis=0)
    # Initialize centroids as random points within the range of each feature
    centroids = np.random.uniform(low=minValues, high=maxValues, size=(k, data.shape[1]))
    return centroids


# Assign each data point to the nearest cluster centroid using euclidean distance
# x = dataset
# k = no of centroids
# centroids = Cluster centroids
def assignClusterIDs(dataset,k,centroids):
    numOfObjects = len(dataset)
    clusterIds = np.zeros(numOfObjects, dtype=int)  # Array to store cluster assignments for each data point

    # Compute distances between each data point and centroids
    # For every object in the dataset
    for i in range(numOfObjects):
        X = dataset[i]
        #find the closest centroid
        centroidIndOfX = 0
        distanceToClosestCentroid = np.Inf
        for j in range (k):
            currentCentroid = centroids[j]
            dist = computeDistance(X, currentCentroid)
            if dist < distanceToClosestCentroid:
                # Found closer centroid
                distanceToClosestCentroid = dist
                centroidIndOfX = j
        #assign to X its closest medoid
        clusterIds[i] = int(centroidIndOfX)

    return clusterIds


# Re-initialize the centroids by calculating the average of all data points in that cluster 
def computeClusterRepresentatives(data,clusters, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        # Calculate mean of data points in the cluster
        clusterPoints = data[clusters == i]
        centroids[i] = np.mean(clusterPoints, axis=0)
    return centroids


# Main thread that implements the k-means algorithm
def kMeans(dataset,k,maxIter):
    centroids = initialization(dataset,k,21)
    prevCentroids = None
    iterCount = 0
    while not np.array_equal(centroids, prevCentroids) and iterCount < maxIter:
        clusters = assignClusterIDs(dataset,k,centroids)
        # Update centroids
        prevCentroids = centroids
        centroids = computeClusterRepresentatives(dataset,clusters,k)
        iterCount += 1
    
    return clusters


def computeSilhouette(clusters, data):
    numOfObjects = len(data)
    silhouetteValues = []
    for i in range(numOfObjects):
        clusterIndex = clusters[i]
        clusterData = data[clusters == clusterIndex]
        intraClusterDistance = 0
        if len(clusterData) > 1:  # Check if the cluster has more than one data point
            intraClusterDistance = np.mean([computeDistance(data[i], point) for point in clusterData if not np.array_equal(data[i], point)])
        interClusterDistances = [np.mean([computeDistance(data[i], point) for point in data[clusters == otherCluster]]) for otherCluster in np.unique(clusters) if otherCluster != clusterIndex]
        minInterClusterDistance = min(interClusterDistances) if interClusterDistances else 0
        silhouetteValue = (minInterClusterDistance - intraClusterDistance) / max(minInterClusterDistance, intraClusterDistance)
        silhouetteValues.append(silhouetteValue)
    score = np.mean(silhouetteValues)
    return score


def plotSilhouette(data,maxIter):
    kValues = range(2, 10)
    silhouetteCoefficients = []

    for k in kValues:
        clusters = kMeans(data,k,maxIter)
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


# Test my functions using sklearn
def test(data):
    kValues = range(2, 10)
    silhouetteCoefficients = []
    seed = 21

    for k in kValues:
        centroids = initialization(data, k, seed)
        kmeans = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=seed)
        kmeans.fit(data)
        #silhouetteCoefficients.append(computeSilhouette(kmeans.labels_,data))
        silhouetteCoefficients.append(silhouette_score(data, kmeans.labels_, metric='euclidean'))
        
        
    print(silhouetteCoefficients)
    plt.plot(kValues, silhouetteCoefficients, marker='o')
    plt.title('Silhouette Coefficients for Different Values of k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.xticks(kValues)
    plt.grid(True)
    plt.show()



def main():
    data = loadDataset()
    plotSilhouette(data,maxIter=50)
    #test(data)


main()