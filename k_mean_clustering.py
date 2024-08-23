import numpy as np
class kMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
    
    def fit(self, X):
        # assign centroids
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]
        for i in range(self.max_iterations):
            cluster_assignments = []
            for j in range(len(X)):
                # distances contains k value, 
                # euclidean distance datapoint X[j] to each centroid
                distances = np.linalg.norm(X[j]-self.centroids, axis=1)
                # append the idx of assigned cluster[0,k-1]
                cluster_assignments.append(np.argmin(distances))
            
            # update the centroids
            for k in range(self.k):
                cluster_data_points = X[np.where(np.array(cluster_assignments==k))]
                if len(cluster_data_points) > 0:
                    self.centroids[k] = np.mean(cluster_data_points, axis=0)
            
            if i > 0 and np.array_equal(self.centroids, previous_centriods):
                break
            previous_centriods = np.copy(self.centroids)

        # store the final assignment
        self.cluster_assignment = cluster_assignments
    
    def predict(self, X):
        # Assign each data point to the nearest centroid
        cluster_assignments = []
        for j in range(len(X)):
            distances = np.linalg.norm(X[j] - self.centroids, axis=1)
            cluster_assignments.append(np.argmin(distances))
        return cluster_assignments
    
def test():
    x1 = np.random.randn(5,2) + 5
    x2 = np.random.randn(5,2) - 5
    X = np.concatenate([x1,x2], axis=0)
    # Initialize the KMeans object with k=3
    kmeans = kMeans(k=2)
    # Fit the k-means model to the dataset
    kmeans.fit(X)
    # Get the cluster assignments for the input dataset
    cluster_assignments = kmeans.predict(X)
    # Print the cluster assignments
    print(cluster_assignments)
    # Print the learned centroids
    print(kmeans.centroids)
    visualize(X, kmeans, cluster_assignments)

from matplotlib import pyplot as plt
def visualize(X, kMeans, cluster_assignments):
    color = ['r', 'b']
    for i in range(kMeans.k):
        plt.scatter(X[np.where(np.array(cluster_assignments)==i)][:, 0], 
                    X[np.where(np.array(cluster_assignments)==i)][:, 1], 
                    color=color[i])

    plt.scatter(kMeans.centroids[:,0],kMeans.centroids[:,1], color='black', marker='o')
    plt.show()

if __name__ == '__main__':
    test()

