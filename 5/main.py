import  pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


import numpy as np
import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions

if __name__ == '__main__':
	df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
	X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
	X_train, X_test, y_train, y_test = train_test_split(
										X, y, test_size=0.3, random_state=0)
	sc = StandardScaler()
	X_train_std = sc.fit_transform(X_train)
	X_test_std = sc.transform(X_test)
	cov_mat = np.cov(X_train_std.T)
	eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
	print '\nEigenvalues \n{0}'.format(eigen_vals)
	tot = sum(eigen_vals)
	var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	
	plt.figure(1)
	
	plt.bar(range(1,14), var_exp, alpha=0.5, align='center',
			label='individualexplained variance')
	plt.step(range(1,14), cum_var_exp, where='mid',
			label='cumulative explained variance')
	plt.ylabel('Explained variance ratio')
	plt.xlabel('Principal components')
	plt.legend(loc='best')
	
	
	
	eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i])
	for i in range(len(eigen_vals))]
	eigen_pairs.sort(reverse=True)
	w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
	print 'Matrix w:{0}'.format(w)
	print
	print X_train_std[0].dot(w)
	
	
	X_train_pca = X_train_std.dot(w)
	
	plt.figure(2)
	
	colors = ['r', 'b', 'g']
	markers = ['s', 'x', 'o']
	for l, c, m in zip(np.unique(y_train), colors, markers):
		plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)
	
	
	
	plt.xlabel('PC 1')
	plt.ylabel('PC 2')
	plt.legend(loc='lower left')
	
	plt.show()
