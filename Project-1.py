#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
data=pd.read_csv('C:/Users/vknsr/Downloads/train.csv')
data.head()


# In[2]:


get_ipython().run_line_magic('pip', 'install -Uq upgini catboost')


# In[3]:


data=data.sample(19_000,random_state=0)


# In[4]:


data['store']=data['store'].astype(str)
data['item']=data['item'].astype(str)
data['date']=pd.to_datetime(data['date'])
data.sort_values("date",inplace=True)


# In[5]:


data.reset_index(inplace=True,drop=True)
data.head()


# In[7]:


train =data[data['date']<"2017-01-01"]


# In[8]:


test=data[data['date']>="2017-01-01"]


# In[9]:


train_features= train.drop(columns=["sales"])
train_target=train["sales"]
test_feature=test.drop(columns=['sales'])
test_target=test['sales']


# In[15]:


from upgini import FeaturesEnricher , SearchKey
from upgini.metadata import CVType


# In[16]:


enricher=FeaturesEnricher(search_keys= {
    'date': SearchKey.DATE
},
cv=CVType.time_series)


# In[18]:


enricher.fit(train_features,train_target,eval_set=[(test_feature,test_target)])


# In[19]:


from catboost import CatBoostRegressor


# In[21]:


from catboost.utils import eval_metric


# In[22]:


model=CatBoostRegressor(verbose=False,allow_writing_files=False,random_state=0)


# In[24]:


enricher.calculate_metrics(train_features,train_target,
                           eval_set=[(test_feature,test_target)],estimator=model,scoring="mean_absolute_percentage_error")


# In[30]:


enriched_train_features = enricher.transform(train_features,keep_input=True)
enriched_test_features= enricher.transform(test_feature,keep_input=True)
enriched_train_features.head()


# In[32]:


model.fit(train_features,train_target)
preds=model.predict(test_feature)
eval_metric(test_target.values,preds,"SMAPE")


# In[35]:


model.fit(enriched_train_features,train_target)
enriched_preds=model.predict(enriched_test_features)
eval_metric(test_target.values,enriched_preds,"SMAPE")


# In[ ]:




