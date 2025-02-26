# scikit-learn
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
X, _ = load_digits(return_X_y=True)
print(X.shape)
embedding = Isomap(n_components=2)
X_transformed = embedding.fit_transform(X[:100])
print(X_transformed.shape)
