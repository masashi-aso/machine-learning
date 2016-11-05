import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from SBS import SBS


csv_data = '''A,B,C,D
	1.0,2.0,3.0,4.0
	5.0,6.0,,8.0
	10.0,11.0,12.0,'''
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))


#print df
#print df.isnull().sum()
#print df.dropna()
#print df.dropna(axis=1)
#print df.dropna(how='all')
#print df.dropna(thresh=4)
#print df.dropna(subset=['C'])
#print df.values


#imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imr = imr.fit(df)

#imputed_data = imr.transform(df.values)
#print imputed_data

df1 = pd.DataFrame([
	['green', 'M', 10.1, 'class1'],
	['red', 'L', 13.5, 'class2'],
	['blue', 'XL', 15.3,'class1']
])
df1.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
#inv_size_mapping = {v: k for k, v in size_mapping.items()}
df1['size'] = df1['size'].map(size_mapping)
#df1['size'] = df1['size'].map(inv_size_mapping)
#print df1

class_mapping = {label:idx for idx, label in enumerate(np.unique(df1['classlabel']))}
df1['classlabel'] = df1['classlabel'].map(class_mapping)
#print df1
#inv_class_mapping = {v: k for k, v in class_mapping.items()}
#df1['classlabel'] = df1['classlabel'].map(inv_class_mapping)
#print df1

#class_le = LabelEncoder()
#y = class_le.fit_transform(df1['classlabel'].values)
#print y
#print class_le.inverse_transform(y)

#X = df1[['color', 'size', 'price']].values
#color_le = LabelEncoder()
#X[:, 0] = color_le.fit_transform(X[:, 0])
##print X

#ohe = OneHotEncoder(categorical_features=[0], sparse=False)
#print ohe.fit_transform(X)
#print pd.get_dummies(df1[['price', 'color', 'size']])

df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


#print('Class labels', np.unique(df_wine['Class label']))
#print df_wine.head()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
	train_test_split(X, y, test_size=0.3, random_state=0)

#mms = MinMaxScaler()

#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.fit_transform(X_test)

#LogisticRegression(penalty='l1')
#lr = LogisticRegression(penalty='l1', C=0.1)
#lr.fit(X_train_std, y_train)
#print 'Training accuracy : {0}'.format(lr.score(X_train_std, y_train))
#print 'Test accuracy : {0}'.format(lr.score(X_test_std, y_test))
#print lr.intercept_
#print lr.coef_

#fig = plt.figure()
#ax = plt.subplot(111)
#colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
#weights, params = [], []
#for c in np.arange(-4,6):
#	lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
#	lr.fit(X_train_std, y_train)
#	weights.append(lr.coef_[1])
#	params.append(10**c)
#weights = np.array(weights)
#for column, color in zip(range(weights.shape[1]), colors):
#		plt.plot(params, weights[:, column], label=df_wine.columns[column+1],color=color)
#plt.axhline(0, color='black', linestyle='--', linewidth=3)
#plt.xlim([10**(-5), 10**5])
#plt.ylabel('weight coefficient')
#plt.xlabel('C')
#plt.xscale('log')
#plt.legend(loc='upper left')
#ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
#plt.show()

#knn = KNeighborsClassifier(n_neighbors=2)
#sbs = SBS(knn, k_features=1)
#sbs.fit(X_train_std, y_train)
#k_feat = [len(k) for k in sbs.subsets_]
#plt.plot(k_feat, sbs.scores_, marker='o')
#plt.ylim([0.7, 1.1])
#plt.ylabel('Accuracy')
#plt.xlabel('Number of features')
#plt.grid()
#plt.show()
#k5 = list(sbs.subsets_[8])
#print(df_wine.columns[1:][k5])
#knn.fit(X_train_std, y_train)
#print('Training accuracy:', knn.score(X_train_std, y_train))
#print('Test accuracy:', knn.score(X_test_std, y_test))
#knn.fit(X_train_std[:, k5], y_train)
#print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
#print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" %
		(f+1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

sfm = SelectFromModel(forest, prefit=True, threshold=0.15)
X_selected = sfm.transform(X_train)
print X_selected.shape


