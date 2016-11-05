from matplot.-lib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
	markers = ('s', 'x', 'o', '^','v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max + 1
	x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max + 1
	xx1, xx2 = np.meshgrid(np.arnge(x1_min,x1_max,resolution),
							np.arange(X2_min,x2_max,resolution))
