import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

def dendrogram_purity(clusters, labels):
    total_elements = sum([len(cluster) for cluster in clusters])
    purity_sum = 0
    
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        label_counts = Counter([labels[i] for i in cluster])
        max_count = max(label_counts.values())
        purity = max_count / len(cluster)
        purity_sum += purity * len(cluster)
    
    return purity_sum / total_elements

# Sample data (2D points)
data = np.array([[1, 2], [2, 3], [3, 4], [8, 7], [7, 8], [9, 6]])
# True labels of the data points
true_labels = np.array([0, 0, 0, 1, 1, 1])

# Perform hierarchical clustering
distance_matrix = pdist(data, 'euclidean')
Z = linkage(distance_matrix, method='ward')

# Plot the dendrogram (optional)
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(Z)
plt.show()

# Create clusters using a threshold to cut the dendrogram
num_clusters = 2
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# Group data points into clusters
clustered_data = [[] for _ in range(num_clusters)]
for idx, cluster_id in enumerate(clusters):
    clustered_data[cluster_id - 1].append(idx)

# Calculate dendrogram purity
purity = dendrogram_purity(clustered_data, true_labels)
print(f'Dendrogram Purity: {purity}')
