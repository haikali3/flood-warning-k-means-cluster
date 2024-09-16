import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 300
n_features = 2
n_clusters = 10
random_state = 42

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plot the results
plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Print cluster centers
# print("Cluster Centers:")
# for i, center in enumerate(centers):
#     print(f"Cluster {i+1}: {center}")

# Predict cluster for a new data point
new_data_point = np.array([[0, 0]])
predicted_cluster = kmeans.predict(new_data_point)
print(f"\nPredicted cluster for [0, 0]: {predicted_cluster[0]+1}")
