import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from adaline import AdalineGD
from perceptron import Perceptron
from perceptron import plot_decision_regions
from adalinesgd import AdalineSGD

if __name__ == '__main__':
	df = pd.read_csv('http://archive.ics.uci.edu/ml/'
		'machine-learning-databases/iris/iris.data', header=None)
	#print df.tail()

	y = df.iloc[0:100, 4].values
	y = np.where(y== 'Iris-setosa', -1, 1)
	X = df.iloc[0:100, [0,2]].values
#	"""
#	plt.scatter(x[:50,0], x[:50,1], color='red',marker='o', label='setosa')
#	plt.scatter(x[50:100,0], x[50:100,1], color='blue',marker='x', label='versicolor')
#	plt.xlabel('sepal length[cm]')
#	plt.ylabel('petal length[cm]')
#	plt.legend(loc='upper left')
#	plt.show()
#	"""
#	start = time.clock()
#	ppn = Perceptron(eta=0.1, n_iter=10)
#	end = time.clock()
#	print start-end
#	ppn.fit(X, y)
#	"""
#	plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#	plt.xlabel('Epochs')
#	plt.ylabel('Number of misclassificasions')
#	plt.show()
#	"""
#	plot_decision_regions(X, y, classifier=ppn)
#	plt.xlabel('sepal length[cm]')
#	plt.ylabel('petal length[cm]')
#	plt.legend(loc='upper left')
#	plt.show()
#	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#	ada1 = AdalineGD(n_iter=50, eta=0.00045).fit(X, y)
#	ax[0].plot(range(1,len(ada1.cost_)+1), ada1.cost_, marker='o')
#	ax[0].set_xlabel('Epochs')
#	ax[0].set_ylabel('log(Sum-squared-error)')
#	ax[0].set_title('Adaline - Learning rate 0.01')
#	ada2 = AdalineGD(n_iter=50, eta=0.00047).fit(X, y)
#	ax[1].plot(range(1,len(ada2.cost_)+1), ada2.cost_, marker='o')
#	ax[1].set_xlabel('Epochs')
#	ax[1].set_ylabel('Sum-squared-error')
#	ax[1].set_title('Adaline - Learning rate 0.0001')
#	plt.show()
	
	ada = AdalineSGD(n_iter = 15, eta = 0.01)
	X_std = np.copy(X)
	X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
	ada.fit(X_std, y)
	ada.partial_fit(np.array([10.0, 10.0]), np.array(-1))
	plot_decision_regions(X_std, y, classifier=ada)
	plt.title('Adaline - Gradient Descent')
	plt.xlabel('Sepal length[cm]')
	plt.ylabel('Petal length[cm]')
	plt.legend(loc='upper left')
	plt.show()
	plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Sum-squared-error')
	plt.show()