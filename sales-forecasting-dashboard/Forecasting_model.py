#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install ucimlrepo


# In[2]:


from ucimlrepo import fetch_ucirepo

# fetch dataset
online_retail = fetch_ucirepo(id=352)

# data (as pandas dataframes)
X = online_retail.data.features
y = online_retail.data.targets

# metadata
print(online_retail.metadata)

# variable information
print(online_retail.variables)


# In[3]:


import pandas as pd

df = X.copy()
df.head()


# In[4]:


df = df[['InvoiceDate', 'Quantity', 'UnitPrice']]


# In[5]:


df['Sales'] = df['Quantity'] * df['UnitPrice']


# In[6]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[7]:


daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()

daily_sales.columns = ['Date', 'Sales']

daily_sales.head()


# In[8]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.plot(daily_sales['Date'], daily_sales['Sales'])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.show()


# In[9]:


daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
daily_sales['Day_Index'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days


# In[10]:


from sklearn.linear_model import LinearRegression

X = daily_sales[['Day_Index']]
y = daily_sales['Sales']

model = LinearRegression()
model.fit(X, y)


# In[11]:


future_days = 30

last_day = daily_sales['Day_Index'].max()

future_X = pd.DataFrame({
    'Day_Index': range(last_day, last_day + future_days)
})

future_predictions = model.predict(future_X)


# In[12]:


plt.figure(figsize=(12,5))

plt.plot(daily_sales['Day_Index'], daily_sales['Sales'], label="Actual Sales")
plt.plot(future_X['Day_Index'], future_predictions, label="Forecast", linestyle='dashed')

plt.title("Sales Forecast")
plt.xlabel("Day Index")
plt.ylabel("Sales")
plt.legend()
plt.show()


# In[13]:


daily_sales.to_csv("sales_forecast_data.csv", index=False)


# In[14]:


forecast_output = future_predictions


# In[15]:


forecast_df = future_X.copy()
forecast_df["Forecast"] = future_predictions

forecast_df.to_csv("sales_forecast_data.csv", index=False)


# In[ ]:




