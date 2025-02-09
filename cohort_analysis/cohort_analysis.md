# Portofolio: Cohort Analysis - Optimize Customer Retention with Acquisition Time-Based Segmentation in Python

## Business Understanding

### Introduction

Cohort analysis is one of the low complexity but high impact to the business decision. Like RFM analysis, cohort analysis also tries to segment customers based on their characteristics such as channel acquisition or their first time purchase. The group is called cohort. One of the most used variable is transaction date.

### Objectives

This analysis aims to find trend and pattern of customer behaviour based on the month they first time purchased. These objectives include:
- which cohort has the most new customers.
- how much percentage of retention and churn rate month to month.

### Dataset and Tools Needed

We will use seblak prasmanan database. Seblak prasmanan is one of the new traditional Indonesian food. It's a boiled krupuk with various topping and what make seblak prasmanan special is you can take only topping that you like as many as you want.

The tools we need for this analysis is only Python programming language and the library such as numpy, pandas, datetime, matplotlib, and seaborn. 

### Methodology

For this analysis, we only need customer_id and transaction date where later we will do some data preprocessing to find their first time purchase and other metrics. Here are the steps to do cohort analysis:
- find the first time purchase for each customer. name it as cohort month.
- use like distinct function in order to make one customer if purchases many times at the same month, count as one time for transaction month.
- substract the transaction month with cohort month to get cohort index.
- make pivot table so we can see which cohort month has how many customers for each cohort index.

## Data Understanding

### Data Acquisition


```python
import numpy as  np
import pandas as pd
import datetime as dt
```


```python
df = pd.read_csv('data/seblak_dataset.csv')
print(df.head())
```

       customer_id transaction_date       city customers  purchase
    0            7       2023-09-23     Bekasi    member       136
    1           22       2023-05-05      Depok    member       142
    2           36       2023-12-16  Tangerang    member       156
    3           28       2023-03-19     Bekasi    member       179
    4           24       2023-01-19     Bekasi    member       164
    

### Data Profiling


```python
# column type
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3936 entries, 0 to 3935
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   customer_id       3936 non-null   int64 
     1   transaction_date  3936 non-null   object
     2   city              3936 non-null   object
     3   customers         3936 non-null   object
     4   purchase          3936 non-null   int64 
    dtypes: int64(2), object(3)
    memory usage: 153.9+ KB
    


```python
# missing value
df.isnull().sum()
```




    customer_id         0
    transaction_date    0
    city                0
    customers           0
    purchase            0
    dtype: int64




```python
# duplicated value
df.duplicated().sum()
```




    np.int64(0)



NOTE:
- transaction_date which means datetime is still an object. we can change data format.
- the data has no missing value
- the data has no duplicated value


```python
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3936 entries, 0 to 3935
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   customer_id       3936 non-null   int64         
     1   transaction_date  3936 non-null   datetime64[ns]
     2   city              3936 non-null   object        
     3   customers         3936 non-null   object        
     4   purchase          3936 non-null   int64         
    dtypes: datetime64[ns](1), int64(2), object(2)
    memory usage: 153.9+ KB
    

### Descriptive Statistics

Because our main goal is to understand customer retention using time-based acquisition, we just need customer_id and transaction_date.


```python
df['customer_id'].value_counts()
```




    customer_id
    68     28
    21     24
    136    24
    38     23
    71     23
           ..
    412     1
    416     1
    408     1
    421     1
    571     1
    Name: count, Length: 552, dtype: int64




```python
df['transaction_date'].value_counts()
```




    transaction_date
    2023-11-15    34
    2023-12-08    28
    2023-12-31    28
    2023-10-06    27
    2023-12-24    26
                  ..
    2023-01-27     1
    2023-01-06     1
    2023-01-02     1
    2023-03-29     1
    2023-02-13     1
    Name: count, Length: 362, dtype: int64



## Data Preprocessing


```python
def get_month(x):
    return dt.datetime(x.year, x.month, 1)
```


```python
df['transaction_month'] = df['transaction_date'].apply(get_month)
```


```python
group = df.groupby('customer_id')['transaction_month']
```


```python
df['cohort_month'] = group.transform('min')
```


```python
print(df.head())
```

       customer_id transaction_date       city customers  purchase  \
    0            7       2023-09-23     Bekasi    member       136   
    1           22       2023-05-05      Depok    member       142   
    2           36       2023-12-16  Tangerang    member       156   
    3           28       2023-03-19     Bekasi    member       179   
    4           24       2023-01-19     Bekasi    member       164   
    
      transaction_month cohort_month  
    0        2023-09-01   2023-01-01  
    1        2023-05-01   2023-02-01  
    2        2023-12-01   2023-01-01  
    3        2023-03-01   2023-01-01  
    4        2023-01-01   2023-01-01  
    


```python
def get_month_int(dframe, column):
    year = dframe[column].dt.year
    month = dframe[column].dt.month
    day = dframe[column].dt.day
    return year, month, day
```


```python
invoice_year, invoice_month, _ = get_month_int(df, 'transaction_month')
```


```python
cohort_year, cohort_month, _ = get_month_int(df, 'cohort_month')
```


```python
year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohort_month
```


```python
df['index_cohort'] = year_diff * 12 + month_diff
```


```python
print(df.head())
```

       customer_id transaction_date       city customers  purchase  \
    0            7       2023-09-23     Bekasi    member       136   
    1           22       2023-05-05      Depok    member       142   
    2           36       2023-12-16  Tangerang    member       156   
    3           28       2023-03-19     Bekasi    member       179   
    4           24       2023-01-19     Bekasi    member       164   
    
      transaction_month cohort_month  index_cohort  
    0        2023-09-01   2023-01-01             8  
    1        2023-05-01   2023-02-01             3  
    2        2023-12-01   2023-01-01            11  
    3        2023-03-01   2023-01-01             2  
    4        2023-01-01   2023-01-01             0  
    


```python
grouping = df.groupby(['cohort_month', 'index_cohort'])
cohort_data = grouping['customer_id'].apply(pd.Series.nunique)
```


```python
cohort_data = cohort_data.reset_index()
```


```python
cohort_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 78 entries, 0 to 77
    Data columns (total 3 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   cohort_month  78 non-null     datetime64[ns]
     1   index_cohort  78 non-null     int32         
     2   customer_id   78 non-null     int64         
    dtypes: datetime64[ns](1), int32(1), int64(1)
    memory usage: 1.7 KB
    


```python
cohort_data['cohort_month'] = pd.DatetimeIndex(cohort_data['cohort_month']).to_period('M')
```

## Modeling


```python
cohort_count = cohort_data.pivot(columns='index_cohort', index='cohort_month', values='customer_id')
```


```python
retention = cohort_count.divide(cohort_count[0], axis=0)
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(16,9), dpi=300)
plt.title('Customer Retention Rate by First Time Purchase', fontsize=24, pad=18)
sns.heatmap(retention.round(3), annot=True, fmt='.0%', cmap='Greens_r')
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_45_0.png)
    



```python
plt.figure(figsize=(16,9), dpi=300)
plt.title('Customer Retention by First Time Purchase', fontsize=24, pad=18)
sns.heatmap(cohort_count, annot=True, cmap='PuBu')
plt.yticks(rotation=0)
plt.show()
```


    
![png](output_46_0.png)
    


## Insights

### Interpretation and Reporting

This cohort table contains twelve month observation where first month is January and the last month is December. The highest amount of acquisition is in October, 59 new customers and the lowest amount of acquisition is in January and February with 37 new customers. In December, almost all cohort month has more than 50% retention which is still acceptable. Only cohort July has 48% retention rate in December. There are 80% of new customers from cohort May comes back to purchase in November which is great.

### Actions

We know that it's a good job to reach 50% retention rate. But, there are still some space we can improve to increase customer retention rate. Retention is where the customers purchase more than one time. Higher retention rate means high frequency of customers purchasing our products. There are some actions we can apply for our marketing strategy, here they are:
- increase customer **satisfaction** by prioritizing great customer service and support.
- improve product from package or feature, build quality or experiences, and ease of use or price to meet the needs of customers.
- apply loyalty programs like membership for personalized experience or special price.

### Further Analysis

- do churn analysis to understand churn rate, identifying what factors make customers stop purchasing or using our products.
- do survival analysis to understand customer lifetime value, predicting how often customers will purchase.
- do comparative analysis like t-test to understand whether the customers with membership do more purchase rather than casual customers.