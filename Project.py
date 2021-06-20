# Classifier List
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Processing Scripts
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Read CSV
df = pd.read_csv('emails.csv')
#print(df)

# IO / Train-Test
X=df.iloc[:,1:-1]
X = normalize(X)
y=np.array(df[["Prediction"]]).ravel()
#print(X)
#print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Classifier Objects
tree1=DecisionTreeClassifier()
rf=RandomForestClassifier()
knn=KNeighborsClassifier(2)
gnb=GaussianNB()
lr=LogisticRegression()

# Decision Tree
print("\nDecision Tree Classifier :\n")
tree1.fit(X_train,y_train)
p=tree1.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,p))
print("Confusion Matrix: \n",confusion_matrix(y_test,p))
print("Classification Report: \n",classification_report(y_test,p))
print("Precision Score: ",precision_score(y_test,p))
print("Recall Score: ",recall_score(y_test,p))

# Random Forest
print("\nRandom Forest Classifier :\n")
rf.fit(X_train,y_train)
p=rf.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,p))
print("Confusion Matrix: \n",confusion_matrix(y_test,p))
print("Classification Report: \n",classification_report(y_test,p))
print("Precision Score: ",precision_score(y_test,p))
print("Recall Score: ",recall_score(y_test,p))

# KNN
print("\nKNN Classifier :\n")
knn.fit(X_train,y_train)
p=knn.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,p))
print("Confusion Matrix: \n",confusion_matrix(y_test,p))
print("Classification Report: \n",classification_report(y_test,p))
print("Precision Score: ",precision_score(y_test,p))
print("Recall Score: ",recall_score(y_test,p))

# Gaussian Naive Bayes
print("\nGaussianNB Classifier :\n")
gnb.fit(X_train,y_train)
p=gnb.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,p))
print("Confusion Matrix: \n",confusion_matrix(y_test,p))
print("Classification Report: \n",classification_report(y_test,p))
print("Precision Score: ",precision_score(y_test,p))
print("Recall Score: ",recall_score(y_test,p))

# KNN
print("\nLogistic Regression :\n")
lr.fit(X_train,y_train)
p=lr.predict(X_test)
print("Accuracy Score: ",accuracy_score(y_test,p))
print("Confusion Matrix: \n",confusion_matrix(y_test,p))
print("Classification Report: \n",classification_report(y_test,p))
print("Precision Score: ",precision_score(y_test,p))
print("Recall Score: ",recall_score(y_test,p))

c1=cross_val_score(knn, X, y, scoring ='accuracy')

c2=cross_val_score(lr, X, y, scoring ='accuracy')

c3=cross_val_score(gnb, X, y, scoring ='accuracy')

c4=cross_val_score(tree1, X, y, scoring ='accuracy')

c5=cross_val_score(rf, X, y, scoring ='accuracy')

print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
data=np.array([c1,c2,c3,c4,c5])
plt.boxplot(data)
print(data)
plt.xticks([1, 2, 3, 4, 5], ['KNN', 'LR', 'NB', 'DTREE', 'RFOREST'])
plt.show()

# PCA and Clustering

pca=PCA(2)
X_trans=pca.fit_transform(X)
print(X_trans)

plt.scatter(X_trans[:,0],X_trans[:,1],s=50)
plt.show()

kmeans=KMeans(n_clusters=2)
kmeans.fit(X_trans)
Y_kmeans=kmeans.predict(X_trans)

plt.scatter(X_trans[:,0],X_trans[:,1],s=50,cmap="viridis",c=Y_kmeans)
plt.show()
