#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[67]:


df = pd.read_excel('luxarycars_from_pandas.xlsx', usecols=[1,2,3,4,5])
df


# In[68]:


dfX = df.drop(columns=['model'])
dfX
sy= df.model
sy


# In[69]:


from sklearn.model_selection import train_test_split
dfX_train, dfX_test, sy_train, sy_test = train_test_split(dfX,sy)


# In[70]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = le.fit(sy_train)
y_train = le.transform(sy_train)
#sy_train, y_train


# In[71]:


dfX_train.to_numpy()
nl = preprocessing.MinMaxScaler()
nl = nl.fit(dfX_train.to_numpy())
X_train = nl.transform(dfX_train.to_numpy())
#X_train


# In[72]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train, y_train)
knn


# In[73]:


X_test = nl.transform(dfX_test.to_numpy())
#dfX_test, X_test


# In[74]:


y_test = le.transform(sy_test.to_numpy())
#sy_test, y_test


# In[75]:


knn.predict(X_test)


# In[76]:


y_test


# In[77]:


knn.score(X_test,y_test)


# In[78]:


knn.score(X_train,y_train)


# In[79]:


#for i in range(6):
  #  knn = KNeighborsClassifier(n_neighbors = i)
   # knn = knn.fit(X_train, y_train)
   # score_train = knn.score(X_train,y_train)
  #  score_test = knn.score(X_test,y_train)
   # i += 1
#return score_train, score_test
    


# In[80]:


knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1 = knn_1.fit(X_train,y_train)

#print(knn_1.predict(X_test))
print(knn_1.score(X_test, y_test))
print(knn_1.score(X_train, y_train))


# In[81]:


knn_2 = KNeighborsClassifier(n_neighbors=2)
knn_2 = knn_2.fit(X_train,y_train)

#print(knn_2.predict(X_test))
print(knn_2.score(X_test, y_test))
print(knn_2.score(X_train, y_train))


# In[82]:


knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3 = knn_3.fit(X_train,y_train)

#print(knn_3.predict(X_test))
print(knn_3.score(X_test, y_test))
print(knn_3.score(X_train, y_train))


# In[83]:


knn_4 = KNeighborsClassifier(n_neighbors=4)
knn_4 = knn_4.fit(X_train,y_train)

#print(knn_4.predict(X_test))
print(knn_4.score(X_test, y_test))
print(knn_4.score(X_train, y_train))


# In[84]:


knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5 = knn_5.fit(X_train,y_train)

#print(knn_1.predict(X_test))
print(knn_5.score(X_test, y_test))
print(knn_5.score(X_train, y_train))


# In[85]:


knn_6 = KNeighborsClassifier(n_neighbors=6)
knn_6 = knn_6.fit(X_train,y_train)

#print(knn_6predict(X_test))
print(knn_6.score(X_test,y_test))
print(knn_6.score(X_train, y_train))


# In[95]:


knn_7 = KNeighborsClassifier(n_neighbors=7)
knn_7 = knn_7.fit(X_train,y_train)

#print(knn_7.predict(X_test))
print(knn_7.score(X_test,y_test))
print(knn_7.score(X_train, y_train))


# In[97]:


knn_8 = KNeighborsClassifier(n_neighbors=8)
knn_8 = knn_8.fit(X_train,y_train)

#print(knn_8.predict(X_test))
print(knn_8.score(X_test,y_test))
print(knn_8.score(X_train, y_train))


# In[98]:


knn_9 = KNeighborsClassifier(n_neighbors=9)
knn_9 = knn_9.fit(X_train,y_train)

#print(knn_9.predict(X_test))
print(knn_9.score(X_test,y_test))
print(knn_9.score(X_train, y_train))


# In[99]:


knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_10 = knn_10.fit(X_train,y_train)

#print(knn_10.predict(X_test))
print(knn_10.score(X_test,y_test))
print(knn_10.score(X_train, y_train))


# In[105]:


import matplotlib.pyplot as plt
X = [1,2,3,4,5,6,7,8,9,10]
Y_test = [knn_1.score(X_test,y_test), knn_2.score(X_test,y_test), knn_3.score(X_test,y_test), knn_4.score(X_test,y_test), knn_5.score(X_test,y_test), knn_6.score(X_test,y_test), knn_7.score(X_test, y_test), knn_8.score(X_test,y_test), knn_9.score(X_test,y_test), knn_10.score(X_test,y_test) ]
Y_train = [knn_1.score(X_train, y_train), knn_2.score(X_train, y_train), knn_3.score(X_train, y_train), knn_4.score(X_train, y_train), knn_5.score(X_train, y_train), knn_6.score(X_train, y_train), knn_7.score(X_train, y_train), knn_8.score(X_train, y_train), knn_9.score(X_train, y_train), knn_10.score(X_train, y_train)]

plt.plot(X, Y_train, color = 'blue', label = 'Train Score')
plt.plot(X, Y_test, color = 'red', label = 'Test Score')
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.ylabel('Model Accuracy')
plt.xlabel('K-Values')
plt.legend()
plt.show()


# In[ ]:




