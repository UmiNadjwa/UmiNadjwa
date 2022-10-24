import streamlit as st
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
iris = sns.load_dataset('iris')

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       iris=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);

from sklearn.cluster import iris
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = iris.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_iris, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
