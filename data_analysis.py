#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju'
# ---------------------------------------------------------------------------
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pickle
import sklearn.metrics as metrics
# Load and scale Iris dataset
iris = datasets.load_iris()
print(iris['DESCR'])
X = iris.data
y = iris.target
# Load saved variables
with open('train_single_node.pickle','rb') as f:
    [pca_transformed, encoded_data, encoded_data2, encoded_data3] = pickle.load(f)
"""
fpr, tpr, threshold = metrics.roc_curve(X, X)
target_names = iris.target_names
# Modify Data
# mod_list = [pca_transformed[:,:1], encoded_data[:,:1], encoded_data2[:,:1], encoded_data3[:,:1]]
# mod_list = [pca_transformed[:49,:1], pca_transformed[50:99,:1], pca_transformed[100:,:1]]
# mod_list = [encoded_data[:49,:1], encoded_data[50:99,:1], encoded_data[100:,:1]]
# mod_list = [encoded_data2[:49,:1], encoded_data2[50:99,:1], encoded_data2[100:,:1]]
mod_list = [encoded_data3[:49,:1], encoded_data3[50:99,:1], encoded_data3[100:,:1]]
colors = ['r', 'g', 'b']
labels = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
count = 0
for data in mod_list:
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color=colors[count])
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p,colors[count], linewidth=2, label=labels[count])
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.legend(loc='best')
    plt.title("Non-Linear relu-based AE")
    plt.xlabel("1st Encoded Node")
    plt.ylabel("Density")
    count += 1
plt.show()
"""