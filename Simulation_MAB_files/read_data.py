import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA


data=[]
for line in open('D:/bandit/CoLinUCB_Revised/ucl/Simulation_MAB_files/users_20+dim-5Ugroups5.json', 'r'):
	data.append(json.loads(line))

print (data)

data_array=[]
for i in range(len(data)):
	data_array.append(data[i][1])


data_array=np.reshape(data_array, (20,5))
print data_array

###########################################################
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plt.figure(figsize=(5,5))
X = data_array
spectral = cluster.SpectralClustering(n_clusters=5,
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
spectral.fit(X)
y_pred = spectral.labels_.astype(np.int)
plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=100)
plt.show()




similarities = euclidean_distances(data_array)


mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=2017,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=2017, n_jobs=1,
                    n_init=1)
npos = nmds.fit_transform(similarities, init=pos)

# Rescale the data_array
pos *= np.sqrt((data_array ** 2).sum()) / np.sqrt((pos ** 2).sum())
npos *= np.sqrt((data_array ** 2).sum()) / np.sqrt((npos ** 2).sum())

# Rotate the data_array
clf = PCA(n_components=2)

pca_data = clf.fit_transform(data_array)

pos = clf.fit_transform(pos)

npos = clf.fit_transform(npos)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])



similarities[np.isinf(similarities)] = 0
# similarities[similarities<1]=0
print np.max(similarities)
# Plot the edges
start_idx, end_idx = np.where(pos)
# a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[pca_data[i, :], pca_data[j, :]]
            for i in range(len(pos)) for j in range(len(pos))]
values = np.abs(similarities)
lc = LineCollection(segments,
                    zorder=2, cmap=plt.cm.Greys,
                    norm=plt.Normalize(0, values.max()))
lc.set_array(similarities.flatten())
lc.set_linewidths(0.5 * np.ones(len(segments)))
ax.add_collection(lc)

s = 100
plt.scatter(pca_data[:, 0], pca_data[:, 1], color='red', s=s, lw=0,
            label='True Position')
# plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
# plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)
plt.show()


