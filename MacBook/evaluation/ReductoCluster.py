import numpy as np

class ReductoCluster:
    def __init__(self, n):
        """
        Initialize the ReductoCluster class.

        Parameters:
            n (int): Number of clusters.
        """
        self.n = n
        self.values = []  # List of vectors to cluster
        self.centers = []  # List of cluster centers

    def add(self, value):
        """
        Add a new value to the clustering.

        Parameters:
            value (list of floats): The vector to be added.
        """
        if not isinstance(value, list) or not all(isinstance(x, (int, float)) for x in value):
            raise ValueError("Value must be a list of floats or integers.")
        self.values.append(value)

    def kmean(self):
        """
        Perform K-Mean clustering on the values.
        """
        if len(self.values) < self.n:
            raise ValueError("Number of values must be at least equal to the number of clusters (n).")

        # Convert values to a NumPy array for easier calculations
        data = np.array(self.values)
        
        # Randomly initialize cluster centers
        centers = data[np.random.choice(data.shape[0], self.n, replace=False)]

        while True:
            # Assign each point to the closest center
            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2) ** 2
            labels = np.argmin(distances, axis=1)

            # Recalculate centers as the mean of points in each cluster
            new_centers = np.array([data[labels == i].mean(axis=0) for i in range(self.n)])

            # Stop if centers do not change
            if np.allclose(centers, new_centers, atol=1e-6):
                break

            centers = new_centers

        self.centers = centers.tolist()

    def tell(self, given_value):
        """
        Find the closest cluster center to the given value and the distance.

        Parameters:
            given_value (list of floats): The vector to find the closest cluster for.

        Returns:
            tuple: Closest cluster center and the square distance to it.
        """
        if not isinstance(given_value, list) or not all(isinstance(x, (int, float)) for x in given_value):
            raise ValueError("Given value must be a list of floats or integers.")
        if not self.centers:
            raise ValueError("Clusters have not been initialized. Perform kmean() first.")

        # Calculate the square distances to all centers
        given_value = np.array(given_value)
        distances = [np.sum((given_value - np.array(center)) ** 2) for center in self.centers]
        
        # Find the closest center
        min_distance = min(distances)
        closest_center = self.centers[np.argmin(distances)]

        return closest_center, min_distance