import pandas

import matplotlib.pyplot as pyplot

from sklearn.cluster import DBSCAN

from sklearn import metrics


# data = pandas.read_csv("dataset_moon.csv")
data = pandas.read_csv("dataset_kmean.csv")

print(data)

pyplot.scatter(data['x1'], data['x2'])
pyplot.savefig("scatter.png")
pyplot.close()

machine = DBSCAN(eps=0.49, min_samples=10)
machine.fit(data)
results = machine.labels_

print(results)
pyplot.scatter(data['x1'], data['x2'], c=results)
pyplot.savefig("scatter_with_color.png")
pyplot.close()

print(metrics.silhouette_score(data, results))

