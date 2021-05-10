import numpy as np # numpy is used to calculate the mean and standard deviation
import matplotlib.pyplot as plt # matplotlib is for drawing graphs
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.preprocessing import scale # scale and center data
from sklearn.svm import SVC # this will make a support vector machine for classificaiton
from sklearn.model_selection import GridSearchCV # this will do cross validation
from sklearn.metrics import confusion_matrix # this creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix
from sklearn.decomposition import PCA # to perform PCA to plot the data
import pandas as pd
import os 
print(os.getcwd())
df = pd.read_csv('processed.cleveland.data', header=None)

df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalasch', 'extang', 'oldpeak', 'slope', 'ca', 'thal', 'hd']
df['ca'].unique()
len(df.loc[(df['ca'] != '?' ) & (df['thal'] != '?' )])
df_no_missing = df.loc[(df['ca'] != '?' ) & (df['thal'] != '?' )]
df_no_missing['ca'].unique()
x = df_no_missing.drop('hd', axis=1).copy()
x.head()
y = df_no_missing['hd'].copy()
x['cp'].unique()
pd.get_dummies(x, columns=['cp']).head()
x_encoded = pd.get_dummies(x, columns=['cp', 'restecg', 'slope', 'thal'])
x_encoded.head()
y.unique()
y[y>0] = 1
y.unique()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
x_test_scaled = scale(x_test)
x_train_scaled = scale(x_train)
clf_svm = SVC(random_state= 42)
clf_svm.fit(x_train_scaled, y_train )
plot_confusion_matrix(clf_svm, x_test_scaled, y_test, display_labels=["No", "Has HD"])
#plt.show()
param_grid = [{'C' : [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],'kernel': ['rbf']},]
optimal_params = GridSearchCV(SVC(), param_grid, cv=5, verbose=0)
optimal_params.fit(x_train_scaled, y_train)
#plt.show()
optimal_params.best_params_
clf_svm = SVC(random_state=42, C= 10, gamma=0.001)
clf_svm.fit(x_train_scaled, y_train )
plt.show()
len(df.columns)
pca = PCA()
x_train_pca = pca.fit_transform(x_train_scaled)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
labels = ['PC'+ str(x) for x in range (1, len(per_var)+1) ]
plt.bar(x=range (1, len(per_var)+1),height = per_var, tick_label= labels)
plt.ylabel('percentage of explained variance')
plt.xlabel('principal component')
plt.title('screen plot')
#plt.show()


pc1 = x_train_pca[:, 0]
pc2 = x_train_pca[:, 1]
clf_svm.fit(np.column_stack((pc1, pc2)), y_train)
x_min = pc1.min() - 1
x_max = pc1.max() + 1
y_min = pc2.min() - 1
y_max = pc2.max() + 1
xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),np.arange(start=y_min, stop=y_max, step=0.1))
Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(10,10))
ax.contourf(xx, yy, Z, alpha=0.1)
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])
scatter = ax.scatter(pc1, pc2, c=y_train,cmap=cmap,  s=100,edgecolors='k', alpha=0.7)
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decison surface using the PCA transformed/projected features')
plt.show()

