import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the iris dataset as an example
iris = load_iris()
X = iris.data
y = iris.target

# Apply PCA to reduce the data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the transformed data points
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
