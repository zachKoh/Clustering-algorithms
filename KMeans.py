import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#Output
#data: A numpy array of the dataset containing data points

#Description: Loads a dataset from a file and extracts data points and labels.
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




#Description: Computes the Euclidean distance between two points.

#Input
#x: Coordinates of the first point
#y: Coordinates of the second point

#Output
#distance: A float which is the Euclidean distance between the two points
    
def computeDistance(x,y):
    #Return the Euclidean distance between x and y
    return np.linalg.norm(x-y)



#Input
#data: A numpy array of the dataset containing the data points
#k: An int which is the number of centroids (clusters)
#seed: An integer which is a seed value for random number generation

#Output
#centroids: A numpy array of the initial centroids for the clusters

#Description: Generates initial centroids for the k-means algorithm.

def initialization(data,k,seed):
    # Get the minimum and maximum values for each feature using seed value
    np.random.seed(seed)
    minValues = np.min(data, axis=0)
    maxValues = np.max(data, axis=0)
    # Initialize centroids as random points within the range of each feature
    centroids = np.random.uniform(low=minValues, high=maxValues, size=(k, data.shape[1]))
    return centroids

#Input 
#dataset: A numpy array which is a dataset containing data points
#k: An integer which is the number of centroids 
#centroids: A numpy array containing the coordinates of all the centroids

#Output
#clusterIds: A numpy array with cluster assignments for each data point

#Description: Assigns cluster IDs to each data point based on the closest centroid.

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



#Input 
#data: A numpy array that contains the dataset containing data points
#clusters: A numpy array containing cluster assignments for each data point
#k: Integer which is the number of centroids

#Output 
#centroids: A numpy array of the coordinates of the centroids

#Description: Computes the centroids of the clusters based on the data points assigned to each cluster.

def computeClusterRepresentatives(data,clusters, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        # Calculate mean of data points in the cluster
        clusterPoints = data[clusters == i]
        centroids[i] = np.mean(clusterPoints, axis=0)
    return centroids


#Input
#dataset: numpy array, dataset containing data points
#k: int, number of centroids
#maxIter: int, maximum number of iterations

#Output:
#clusters: numpy array, cluster assignments for each data point

#Description: Performs the k-means algorithm to cluster the dataset into k clusters.

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


#Input
#clusters: numpy array, cluster assignments for each data point
#data: numpy array, dataset

#Output
#score: float, Silhouette coefficient

#Description: Computes the Silhouette coefficient for a given clustering.

def computeSilhouette(clusters, data):
    numOfObjects = len(data)
    silhouetteValues = []
    for i in range(numOfObjects):
        clusterIndex = clusters[i]
        clusterData = data[clusters == clusterIndex]
        intraClusterDistance = 0
        if len(clusterData) > 1:  #Check if the cluster has more than one data point
            intraClusterDistance = np.mean([computeDistance(data[i], point) for point in clusterData if not np.array_equal(data[i], point)])
        interClusterDistances = [np.mean([computeDistance(data[i], point) for point in data[clusters == otherCluster]]) for otherCluster in np.unique(clusters) if otherCluster != clusterIndex]
        if len(interClusterDistances) == 0:  #No inter-cluster distances for k = 1
            silhouetteValue = 0
        else:
            minInterClusterDistance = min(interClusterDistances)
            silhouetteValue = (minInterClusterDistance - intraClusterDistance) / max(minInterClusterDistance, intraClusterDistance)
        silhouetteValues.append(silhouetteValue)
    score = np.mean(silhouetteValues)
    return score



#Input
#data: numpy array, dataset
#maxIter: int, maximum number of iterations for kMeans algorithm

#Output 
#None

#Description: Plots the Silhouette coefficients for varied values of k.

def plotSilhouette(data,maxIter):
    kValues = range(1, 10)
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


#Input
#data: numpy array, dataset

#Description: Testing my functions using sklearn
def test(data):
    kValues = range(1, 10)
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


#Description: Main function to run K-means algorithm and plot the silhouette coefficients
def main():
    data = loadDataset()
    plotSilhouette(data,maxIter=50)
    #test(data)


#If statement so that Kmeans isn't plotted every time it is imported from another file
if __name__ == "__main__":
    main()