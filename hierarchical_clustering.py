from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
import scipy.io
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from numpy import genfromtxt

data = scipy.io.loadmat('2bags.mat')
indiv_data = data['individual_data']
indiv_data_normalized = np.dot(np.diag(1.0 / np.linalg.norm(indiv_data, axis=1)), indiv_data)
condensed_dist_vector = pdist(indiv_data_normalized, 'euclidean')

# generate the linkage matrix
Z = linkage(indiv_data_normalized, 'ward')
""" Basic dendrogram
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8  # font size for the x axis labels
)
#plt.show()
"""

# How well does the clustering preserve the original distance
c, coph_dists = cophenet(Z, condensed_dist_vector)
c # Closer to 1 means better preservation

# Examine the clusters
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=8,  # show only the last p merged clusters
    leaf_rotation=90,
    leaf_font_size=12  # to get a distribution impression in truncated branches
)

data = scipy.io.loadmat('2bags.mat')
indiv_data = data['individual_data']
k=8
cluster = fcluster(Z, k, criterion='maxclust')

#Combine Stance
bow_data = scipy.io.loadmat('stance.mat')
bow_stance = bow_data['indiv_stance']
bow_stance = bow_stance.flatten()


classifier_data = scipy.io.loadmat('individual_strength_class.mat')
classifier_stance = classifier_data['prochoice_strength']
classifier_stance = classifier_stance.flatten()
classifier_stance = 2*(classifier_stance-0.5)

final_stance = 0.5*(bow_stance + classifier_stance)

#diversity
diversity = genfromtxt('proportion.csv', delimiter=',',usecols=2, skip_header=1)

plt.figure(1)
for i in range(len(np.unique(cluster))):
    index = np.where(cluster==(i+1))[0]
    x = final_stance[index]
    y = diversity[index]
    plot_num = 331+i
    plt.subplot(plot_num)
    plt.plot(x,y,'ro')
    plt.axis([-1,1,0,1])

plt.show()

