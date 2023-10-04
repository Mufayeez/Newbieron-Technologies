#!/usr/bin/env python
# coding: utf-8

# # Newbieron Technology Internship 

# # Name : Syed Mufayeez Raza

# # Project : Store Sales Prediction 

# In[2]:


import pandas as pd 


# In[3]:


import numpy as np


# In[4]:


import seaborn as sns


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[7]:


data = pd.read_csv("F://Profession//Interships//Newbieron technologies//Final Project//Store Sales data.csv")


# In[8]:


data.head()


# In[9]:


data.Item_Identifier.nunique()


# In[10]:


unique_item_identifier = data.Item_Identifier.str[:3].unique()


# In[11]:


unique_item_identifier


# In[12]:


data.shape


# In[13]:


data.info()


# In[14]:


categorical_features = data.select_dtypes(include=['object']).columns.to_list()
numerical_features = data.select_dtypes(exclude=['object']).columns.to_list()

print("Cateforical Features: ",categorical_features)
print("Numerical Features: ", numerical_features)


# In[15]:


data.isnull().sum()


# In[16]:


data.Item_Weight.mean()


# In[17]:


data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)


# In[18]:


data.isnull().sum()


# In[19]:


mode_of_outlet_size = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x:x.mode()[0]))


# In[20]:


mode_of_outlet_size


# In[21]:


missing_values = data.Outlet_Size.isnull()
missing_values


# In[22]:


data.loc[missing_values, 'Outlet_Size'] = data.loc[missing_values, 'Outlet_Type'].apply(lambda x: mode_of_outlet_size[x])


# In[23]:


data.isnull().sum()


# # Analysis

# In[25]:


data.describe()


# In[26]:


sns.set()


# # Visualization

# In[27]:


numerical_features


# In[32]:


plt.figure(figsize=(5,5))
sns.distplot(data.Item_Weight)
plt.show()


# In[33]:


plt.figure(figsize=(5,5))
sns.distplot(data.Item_Visibility)
plt.show()


# In[34]:


plt.figure(figsize=(5,6))
sns.distplot(data.Item_MRP)
plt.show()


# In[36]:


plt.figure(figsize=(6,5))
sns.countplot(x=data.Outlet_Establishment_Year)
plt.show()


# In[37]:


plt.figure(figsize=(4,4))
sns.distplot(data.Item_Outlet_Sales)
plt.show()


# In[38]:


categorical_features


# In[39]:


plt.figure(figsize=(6,6))
sns.countplot(x=data.Item_Fat_Content)
plt.show()


# In[40]:


data.Item_Fat_Content.value_counts()


# In[41]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(
    {
        'low fat' : 'Low Fat',
        'LF' : 'Low Fat',
        'reg' : 'Regular'
    })


# In[42]:


plt.figure(figsize=(6,6))
sns.countplot(x=data.Item_Fat_Content)
plt.show()


# In[43]:


item_type_count = data['Item_Type'].value_counts().sort_values(ascending=False)


# In[44]:


plt.figure(figsize=(12,6))
ax = sns.countplot(data=data, x='Item_Type', order=item_type_count.index)
plt.xticks(rotation=45)
plt.xlabel('Item Type')
plt.ylabel('Count')
plt.title('Count of Items by type')
plt.show()


# In[45]:


outlet_size_count = data['Outlet_Size'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
ax = sns.countplot(data=data, x='Outlet_Size', order=outlet_size_count.index)
plt.xticks(rotation=45)
plt.xlabel('Outlet Size')
plt.ylabel('Count')
plt.title('Count of Outlet by size')
plt.show()


# In[46]:


outlet_location_type_count = data['Outlet_Location_Type'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
ax = sns.countplot(data=data, x='Outlet_Location_Type', order=outlet_location_type_count.index)
plt.xticks(rotation=45)
plt.xlabel('Outlet Location Type')
plt.ylabel('Count')
plt.title('Count of Outlet by Location Type')
plt.show()


# In[61]:


outlet_type_count = data['Outlet_Type'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
ax = sns.countplot(data=data, x='Outlet_Type', order=outlet_type_count.index)
plt.xticks(rotation=45)
plt.xlabel('Outlet Type')
plt.ylabel('Count')
plt.title('Count of Outlet by Type')
plt.show()


# In[48]:


from sklearn.preprocessing import LabelEncoder


# In[49]:


encoder = LabelEncoder()


# In[50]:


data['Item_Identifier'] = encoder.fit_transform(data['Item_Identifier'])

data['Item_Fat_Content'] = encoder.fit_transform(data['Item_Fat_Content'])

data['Item_Type'] = encoder.fit_transform(data['Item_Type'])

data['Outlet_Identifier'] = encoder.fit_transform(data['Outlet_Identifier'])

data['Outlet_Size'] = encoder.fit_transform(data['Outlet_Size'])

data['Outlet_Location_Type'] = encoder.fit_transform(data['Outlet_Location_Type'])

data['Outlet_Type'] = encoder.fit_transform(data['Outlet_Type'])


# In[51]:


data.head(15)


# In[52]:


X = data.drop(columns='Item_Outlet_Sales', axis=1)
y = data['Item_Outlet_Sales']
print('X:',X)
print('Y:',y)


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2,test_size=0.2)


# In[54]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[55]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[56]:


LR = LinearRegression()
LR.fit(X_train, y_train)


# In[57]:


y_pred = LR.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression:")
print("Mean Squared Error: ", mse)
print("R-squared: ",r2)


# In[58]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Random Forest Regression:")
print("Mean Squared Error: ", mse_rf)
print("R-squared: ",r2_rf)


# In[ ]:




