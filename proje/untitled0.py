# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:53:52 2024

@author: msı
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#veri onişleme
veriler =pd.read_csv('Iris.csv')

print(veriler)

x=veriler.iloc[:,1:4].values #bağımlı değişken
y=veriler.iloc[:,4:].values #bağımsız değişken

#veri kumesinin test ve veri olarak bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)
#verilerin olceklendirme
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#logistic regression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)#egitim

y_pred = logr.predict(X_test)#tahmin

#karmaşıklık matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)
print('LR')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)



fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

#Knn
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_pred,y_test) 
print('KNN')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#Svs
from sklearn.svm import  SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

cm=confusion_matrix(y_pred,y_test)
print('SVC')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#naive bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm=confusion_matrix(y_pred,y_test)
print('GNB')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#Decision tree
from sklearn.tree import  DecisionTreeClassifier    
dtc = DecisionTreeClassifier(splitter='random')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm=confusion_matrix(y_pred,y_test)
print('DTC')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
#random forest

from sklearn.ensemble import  RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion= 'entropy')

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
y_proba = rfc.predict_proba(X_test)

cm=confusion_matrix(y_pred,y_test)
print('RFC')
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(cm, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


