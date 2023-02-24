#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[4]:


df=pd.read_csv('creditcard.csv')


# In[5]:


df.head()


# In[6]:


class_names={0:'Not Fraud',1:'Fraud'}
print(df.Class.value_counts().rename(index = class_names))


# In[7]:


print((df.groupby('Class')['Class'].count()/df['Class'].count()) *100) 
((df.groupby('Class')['Class'].count()/df['Class'].count())*100).plot.pie()


# In[8]:


corr=df.corr()
corr


# In[9]:


plt.figure(figsize=(24,18))
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()


# In[10]:


plt.figure(figsize=(7,5))
sns.countplot(df['Class'])
plt.title("Class Count", fontsize=18)
plt.xlabel("Record counts by class", fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.show()


# In[11]:


y=df['Class']
X=df.drop(['Class'],axis=1)


# In[12]:


X.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.20)


# In[15]:


print(np.sum(y)) 
print(np.sum(y_train)) 
print(np.sum(y_test))


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


model=LogisticRegression()
model.fit(X_train, y_train.values.ravel())


# In[18]:


pred=model.predict(X_test)


# In[19]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors= (pred != y_test).sum()
print("The model used is Logistic Regression")
acc=accuracy_score(y_test,pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test,pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test,pred)
dataframe=pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[20]:


from sklearn.neighbors import KNeighborsClassifier


# In[21]:


model1=KNeighborsClassifier() 
model1.fit(X_train,y_train)


# In[22]:


pred=model1.predict(X_test)


# In[23]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors=(pred != y_test).sum()
print("The model used is KNeighborsClassifier")
acc=accuracy_score(y_test, pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test, pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test, pred)
dataframe=pd.DataFrame (matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


model2=RandomForestClassifier()
model2.fit(X_train,y_train)


# In[26]:


pred=model2.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors=(pred != y_test).sum()
print("The model used is Random Forest classifier")
acc=accuracy_score(y_test, pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test, pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True,cbar=None,cmap="Blues",fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[28]:


from sklearn.tree import DecisionTreeClassifier


# In[29]:


model3=DecisionTreeClassifier()
model3.fit(X_train,y_train)


# In[30]:


pred=model3.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors=(pred != y_test).sum()
print("The model used is Decision Tree classifier")
acc=accuracy_score(y_test, pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test, pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True,cbar=None,cmap="Blues",fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[32]:


from sklearn.svm import SVC


# In[33]:


model4=SVC()
model4.fit(X_train,y_train)


# In[34]:


pred=model4.predict(X_test)


# In[35]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors=(pred != y_test).sum()
print("The model used is Support Vector classifier")
acc=accuracy_score(y_test, pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test, pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True,cbar=None,cmap="Blues",fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[37]:


import xgboost as xgb


# In[38]:


model5=xgb.XGBClassifier()
model5.fit(X_train,y_train)


# In[39]:


pred=model5.predict(X_test)


# In[40]:


from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix

n_errors=(pred != y_test).sum()
print("The model used is XG Boost classifier")
acc=accuracy_score(y_test, pred)
print("The accuracy is {}".format(acc))
prec=precision_score(y_test, pred)
print("The precision is {}".format(prec))
class_names=['not_fraud', 'fraud']
matrix=confusion_matrix(y_test, pred)
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
sns.heatmap(dataframe, annot=True,cbar=None,cmap="Blues",fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()


# In[41]:


scorelr=model.score(X_test,y_test)
scoreknn=model1.score(X_test,y_test)
scorerf=model2.score(X_test,y_test)
scoredt=model3.score(X_test,y_test)
scoresv=model4.score(X_test,y_test)
scorexgb=model5.score(X_test,y_test)


# In[45]:


x=['LR','KNN','RF','DT','SV','XGB']
y =[scorelr,scoreknn,scorerf,scoredt,scoresv,scorexgb]
ax=sns.stripplot(x, y);
ax.set(xlabel='Models', ylabel='Accuracy')
plt.title('Accuaracy Comparison Plot');
plt.show()


# In[ ]:




