import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)
gen_model_class0 = GaussianMixture(n_components=1).fit(X[y == 0])
gen_model_class1 = GaussianMixture(n_components=1).fit(X[y == 1])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = np.c_[xx.ravel(), yy.ravel()]
Z_gen0 = np.exp(gen_model_class0.score_samples(grid)).reshape(xx.shape)
Z_gen1 = np.exp(gen_model_class1.score_samples(grid)).reshape(xx.shape)

plt.contour(xx, yy, Z_gen0, levels=5, colors='blue', alpha=0.5)
plt.contour(xx, yy, Z_gen1, levels=5, colors='red', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title("Generative Model: Gaussian Mixture")
plt.show()
