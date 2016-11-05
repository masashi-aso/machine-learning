import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class Perceptron(object):
	"""perceptron
	parameter
	----------------
	eta : float
		rate of learn
	n_iter : int
		how many times data is trained
	attribute
	---------------
	w_ : 1d array
		weight after adaption
	errors_ : list
		how many mistake epoch takes
	"""
	def __init__ (self,eta =0.01, n_iter=10):
		self.eta = eta
		self.n_iter = n_iter
	
	def fit(self, X, y):
		"""fit to training data
		
		parameter
		---------------
		X : {array}, shape = [n_samples, n_features]
			trainingdata
			n_samples is number of samples, n_features is number of features
		y : array, shape = [n_samples]
			object variant
		
		return
		----------------
		self : object
		
		"""
		self.w_ = np.zeros(1 + X.shape[1])
		self.errors_ = []
		
		for _ in range (self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int (update != 0.0)
			self.errors_.append(errors)
		return self
	
	def net_input(self, xi):
		return np.dot(xi, self.w_[1:]) + self.w_[0]
	
	def predict(self, xi):
		return np.where(self.net_input(xi) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, resolution=0.02):
	markers = ('s','x','^','V')
	colors = ('red', 'blue', 'lightgreen','gray','cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	
	x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max()+1
	x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max()+1
	xg1 = np.arange(x1_min, x1_max, resolution)
	xg2 = np.arange(x2_min, x2_max, resolution)
	xx1, xx2 = np.meshgrid(xg1,xg2)
	
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha = 0.8, c=cmap(idx),marker=markers[idx], label=cl)


