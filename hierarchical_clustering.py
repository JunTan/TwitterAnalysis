from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist

data = scipy.io.loadmat('2bags.mat')
indiv_data = data['individual_data']
indiv_data_normalized = np.dot(np.diag(1.0 / np.linalg.norm(indiv_data, axis=1)), indiv_data)
condensed_dist_vector = pdist(indiv_data_normalized, 'euclidean')

# generate the linkage matrix
Z = linkage(indiv_data_normalized, 'ward')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

plt.show()