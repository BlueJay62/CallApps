#!/usr/bin/env python
# coding: utf-8

# In[331]:


import pandas as pd


# In[332]:


import numpy as np


# In[333]:


data = pd.read_excel('Desktop//Data SpaceApps.xlsx')


# In[334]:


data.head()


# In[335]:


data.isnull().sum()


# In[336]:


susceptibility_mapping = {
'Very Low' : -1,
'Low': 0,
'Moderate': 1,
'High': 2,
'Very High' : 3}


# In[337]:


data['Susceptibility'] = data['Susceptibility'].map(susceptibility_mapping)


# In[338]:


data.head()


# In[339]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import mglearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[340]:


a = data["Latitude"]
b = data["Longitude"]
X = np.column_stack((a, b))

y = data['Susceptibility']


# In[341]:


zxc = pd.DataFrame({"Latitude":list(X[:,0]),"Longitude":list(X[:,1]),"Susceptibility":y})
zxc


# In[342]:


mms = MinMaxScaler()
X_new = mms.fit_transform(X)
X_new


# In[343]:


mglearn.discrete_scatter(X_new[:, 0], X_new[:, 1], y)
plt.legend(["Very Low", "Low", "Moderate", "High", "Very High"], loc=4)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
print("X.shape: {}".format(X.shape))


# In[344]:


from sklearn.model_selection import train_test_split

# #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25,random_state=0)


# In[345]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7)


# In[346]:


clf.fit(X_train, y_train)


# In[347]:


print("Test set predictions: {}".format(clf.predict(X_test)))


# In[348]:


print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))


# In[349]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25,random_state=1)


training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 20)

for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
plt.figure(figsize=(15, 11))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.minorticks_on()
plt.legend()

