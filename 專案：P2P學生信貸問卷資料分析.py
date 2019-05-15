#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_excel(r'C:\Users\Danny Wong\Desktop\Tammy.xlsx')


# In[3]:


df


# In[4]:


df1=df.drop(['datetime','session_id','id'], axis=1)
df1=df1.dropna()
df1=df1.drop([0])
df1


# In[5]:


def tran_cat_to_num(df1):
    if df1['parent_education'] == '國中':
        return 0
    if df1['parent_education'] == '高中/職':
        return 1
    if df1['parent_education'] == '專科':
        return 2
    if df1['parent_education'] == '大學':
        return 3
    if df1['parent_education'] == '研究所':
        return 4
    
df1['parent_education']=df1.apply(tran_cat_to_num,axis=1)
df1


# In[6]:


df1=df1.dropna()
df1


# In[7]:


def tran_cat_to_num2(df1):
    if df1['property'] == '動產':
        return 0
    if df1['property'] == '不動產':
        return 1
    if df1['property'] == '均有':
        return 2
    if df1['property'] == '名下無財產':
        return 3
    
df1['property']=df1.apply(tran_cat_to_num2,axis=1)
df1


# In[8]:


def tran_cat_to_num3(df1):
    if df1['usage'] == '購買汽/機車':
        return 0
    if df1['usage'] == '購買3C':
        return 1
    if df1['usage'] == '生活費':
        return 2
    if df1['usage'] == '娛樂費':
        return 3
    if df1['usage'] == '教育費':
        return 4
    if df1['usage'] == '投資理財':
        return 5
    
    
df1['usage']=df1.apply(tran_cat_to_num3,axis=1)
df1


# In[9]:


def tran_cat_to_num4(df1):
    if df1['hometown'] == 'keelung':
        return 0
    if df1['hometown'] == 'taipei':
        return 1
    if df1['hometown'] == 'newtaipei':
        return 2
    if df1['hometown'] == 'taoyuan':
        return 3
    if df1['hometown'] == 'hsinchucounty':
        return 4
    if df1['hometown'] == 'taichung':
        return 5
    if df1['hometown'] == 'nantou':
        return 6
    if df1['hometown'] == 'changhua':
        return 7
    if df1['hometown'] == 'tainan':
        return 8
    if df1['hometown'] == 'kaohsiung':
        return 9
    
    
df1['hometown']=df1.apply(tran_cat_to_num4,axis=1)
df1


# In[10]:


def tran_cat_to_num5(df1):
    if df1['living_place'] == 'keelung':
        return 0
    if df1['living_place'] == 'taipei':
        return 1
    if df1['living_place'] == 'newtaipei':
        return 2
    if df1['living_place'] == 'taoyuan':
        return 3
    if df1['living_place'] == 'hsinchucounty':
        return 4
    if df1['living_place'] == 'taichung':
        return 5
    if df1['living_place'] == 'nantou':
        return 6
    if df1['living_place'] == 'changhua':
        return 7
    if df1['living_place'] == 'tainan':
        return 8
    if df1['living_place'] == 'kaohsiung':
        return 9
    
    
df1['living_place']=df1.apply(tran_cat_to_num5,axis=1)
df1


# In[11]:


df1=df1.dropna()
df1


# In[12]:


df1=df1.drop("college",axis=1)


# 

# In[13]:


df1=df1.drop('current_grade',axis=1)
df1=df1.drop('highschool',axis=1)
df1=df1.drop('department',axis=1)
df1.columns


# In[14]:


Col=df1.to_dict()
colname={}
for x in Col:
    colname[x]='int'
    
df1=df1.astype(colname)
df1=df1.dropna()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split

X = df1.drop('default risk',axis=1)
y = df1['default risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=101)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

#n_estimators代表要使用多少CART樹（CART樹為使用GINI算法的決策樹）
rfc = RandomForestClassifier(n_estimators=200)

#從訓練組資料中建立隨機森林模型
rfc.fit(X_train, y_train)

#預測測試組的駝背是否發生
rfc_pred = rfc.predict(X_test)

#利用confusion_matrix來看實際及預測的差異
print(confusion_matrix(y_test,rfc_pred))

#利用classification_report來看precision、recall、f1-score、support
#print(classification_report(y_test,rfc_pred))


# In[16]:


y_test


# In[17]:


rfc_pred


# In[18]:


import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(df1)
PCA(copy=True, iterated_power='auto', n_components=5, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)

#print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
pca.n_components_


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import torch
a = torch.ones(2,2)
a[0][0]=30
a[0][1]=33
a[1][0]=12
a[1][1]=2

b=torch.ones(2,2)


c=a.mm(b)
c



d=torch.ones(2,2,2,2)
d[0]

