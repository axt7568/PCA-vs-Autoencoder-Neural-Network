#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju'
# ---------------------------------------------------------------------------
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import  norm
import pickle

# Load and scale Iris dataset
iris = datasets.load_iris()
print(iris['DESCR'])
X = iris.data
y = iris.target
target_names = iris.target_names

# Load saved variables
with open('train.pickle','rb') as f:
    [pca_transformed, encoded_data, encoded_data2, encoded_data3] = pickle.load(f)

# Function to plot data according to original labels
def plot3clusters(X, title, vtitle):
  plt.figure()
  # plt.subplots(2,2)
  colors = ['navy', 'turquoise', 'darkorange']
  lw = 2
  for color, i, target_name in zip(colors, [0, 1, 2], target_names):
      plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=1., lw=lw,
                  label=target_name)
  plt.legend(loc='best', shadow=False, scatterpoints=1)
  plt.title(title)
  plt.xlabel(vtitle + "1")
  plt.ylabel(vtitle + "2")
  plt.show()

# Set thw other two classes to zero
pca_transformed[48:,:2] = 0
encoded_data[48:,:2] = 0
encoded_data2[48:,:2] = 0
encoded_data3[48:,:2] = 0
# Plot PCA and 3 AE for comparison
plot3clusters(pca_transformed[:,:2], 'PCA', 'PC')
plot3clusters(encoded_data[:,:2], 'Linear AE', 'AE')
plot3clusters(encoded_data2[:,:2], 'Non-Linear sigmoid-based AE', 'AE')
plot3clusters(encoded_data3[:,:2], 'Non-Linear relu-based AE', 'AE')


# mod_list = [pca_transformed[:,:1], encoded_data[:,:1], encoded_data2[:,:1], encoded_data3[:,:1]]
# mod_list = [pca_transformed[:49,:1], pca_transformed[50:99,:1], pca_transformed[100:,:1]]
# mod_list = [encoded_data[:49,:1], encoded_data[50:99,:1], encoded_data[100:,:1]]
# mod_list = [encoded_data2[:49,:1], encoded_data2[50:99,:1], encoded_data2[100:,:1]]
mod_list = [encoded_data3[:49,:1], encoded_data3[50:99,:1], encoded_data3[100:,:1]]

# New mod list
# mod_list = [encoded_data3[:49,:1]]
# mod_list = [encoded_data[:49,:1]]
# mod_list = [encoded_data2[:49,:1]]
# mod_list = [encoded_data3[:49,:1]]


colors = ['r', 'g', 'b']
labels = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
count = 0
for data in mod_list:
    """
    # Fit to original distribution
    sns.distplot(data)
    sns.distplot(data, fit=norm, kde=False)
    plt.show()
    """
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    # Choose the class
    i = 2
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color=colors[i])
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p,colors[i], linewidth=2, label=labels[i])
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.legend(loc='best')
    plt.title("Non-Linear relu-based AE; " + title)
    plt.xlabel("1st Encoded Dimension")
    plt.ylabel("Density")
    count += 1
plt.show()
