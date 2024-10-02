import numpy as np
from sklearn.cluster import KMeans

class KVClusterV2:
    def __init__(self, n):
        """
        Initialize the KVCluster with the number of clusters `n`.
        """
        self.n = n  # number of clusters
        self.kv_pairs = []  # list of (key, value) pairs
        self.clusters = []  # list of (c, cv) pairs
        self.labels = []  # store the cluster assignment for each key
    
    def add(self, key, value):
        """
        Add a new (key, value) pair. Both key and value should be 1D vectors of floats.
        """
        self.kv_pairs.append((np.array(key), np.array(value)))
    
    def cluster_lower_bound(self):
        """
        Perform K-means clustering on the keys and compute the cluster centers (c) 
        and corresponding lower bounds of value vectors (cv).
        """
        # Extract all keys and values
        keys = np.array([kv[0] for kv in self.kv_pairs])
        values = np.array([kv[1] for kv in self.kv_pairs])
        
        # Run K-means clustering on the keys
        kmeans = KMeans(n_clusters=self.n, random_state=0)
        kmeans.fit(keys)
        
        # Get the cluster centers (c) and the cluster assignments for each key
        cluster_centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_  # Store the labels (cluster assignment for each key)
        
        # For each cluster, find the lower bound of the value vectors in that cluster
        cluster_values = []
        for i in range(self.n):
            # Get the indices of the points assigned to the ith cluster
            cluster_indices = np.where(self.labels == i)[0]
            
            # Get the corresponding value vectors
            cluster_value_vectors = values[cluster_indices]
            
            # Compute the lower bound of the value vectors (element-wise minimum)
            if len(cluster_value_vectors) > 0:
                lower_bound = np.min(cluster_value_vectors, axis=0)
            else:
                lower_bound = None  # No values in this cluster
            
            cluster_values.append(lower_bound)
        
        # Update the clusters with (c, cv) pairs
        self.clusters = [(c, cv) for c, cv in zip(cluster_centers, cluster_values)]
    
    def cluster_average(self):
        """
        Perform K-means clustering on the keys and compute the cluster centers (c) 
        and corresponding lower bounds of value vectors (cv).
        """
        # Extract all keys and values
        keys = np.array([kv[0] for kv in self.kv_pairs])
        values = np.array([kv[1] for kv in self.kv_pairs])
        
        # Run K-means clustering on the keys
        kmeans = KMeans(n_clusters=self.n, random_state=0)
        kmeans.fit(keys)
        
        # Get the cluster centers (c) and the cluster assignments for each key
        cluster_centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_  # Store the labels (cluster assignment for each key)
        
        # For each cluster, find the lower bound of the value vectors in that cluster
        cluster_values = []
        for i in range(self.n):
            # Get the indices of the points assigned to the ith cluster
            cluster_indices = np.where(self.labels == i)[0]
            
            # Get the corresponding value vectors
            cluster_value_vectors = values[cluster_indices]
            
            # Compute the lower bound of the value vectors (element-wise minimum)
            if len(cluster_value_vectors) > 0:
                lower_bound = np.mean(cluster_value_vectors, axis=0)
            else:
                lower_bound = None  # No values in this cluster
            
            cluster_values.append(lower_bound)
        
        # Update the clusters with (c, cv) pairs
        self.clusters = [(c, cv) for c, cv in zip(cluster_centers, cluster_values)]
    
    def tell(self, key):
        """
        Return the cv whose corresponding cluster center c is closest to the given key.
        """
        key = np.array(key)
        closest_cluster = None
        min_distance = float('inf')
        
        for c, cv in self.clusters:
            distance = np.linalg.norm(key - c)  # Euclidean distance between key and cluster center
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cv
        
        return closest_cluster
    
    def self_tell(self):
        """
        Return a list of indices representing the assigned cluster center for each member in kv_pairs.
        """
        return list(self.labels)