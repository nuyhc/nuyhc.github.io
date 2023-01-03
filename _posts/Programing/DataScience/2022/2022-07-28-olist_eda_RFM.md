---
title: Olist EDA 및 RFM 분석
date: 2022-07-28T02:26:21.280Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Numpy
  - matplot
  - Seaborn
---

# Kaggle - Brazilian E-Commerce Public Dataset by Olist
[참고 노트북 | Customer Segmentation & LTV](https://www.kaggle.com/code/richardnnamdi/customer-segmentation-ltv)

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import warnings
warnings.filterwarnings("ignore")
```

### 시각화 함수 정의


```python
def format_spines(ax, right_border=True):
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    ax.spines['top'].set_visible(False)
    if right_border:
        ax.spines['right'].set_color('#FFFFFF')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')
    

def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):
    ncount = len(df)
    if hue != False:
        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax)
    else:
        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax)
    format_spines(ax)

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')
    if not hue:
        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)
    else:
        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)  
    if title != '':
        ax.set_title(title)       
    plt.tight_layout()
    
    
def bar_plot(x, y, df, colors='Blues_d', hue=False, ax=None, value=False, title=''):
    try:
        ncount = sum(df[y])
    except:
        ncount = sum(df[x])
    if hue != False:
        ax = sns.barplot(x=x, y=y, data=df, palette=colors, hue=hue, ax=ax, ci=None)
    else:
        ax = sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax, ci=None)
    format_spines(ax)
    for p in ax.patches:
        xp=p.get_bbox().get_points()[:,0]
        yp=p.get_bbox().get_points()[1,1]
        if value:
            ax.annotate('{:.2f}k'.format(yp/1000), (xp.mean(), yp), 
                    ha='center', va='bottom') 
        else:
            ax.annotate('{:.1f}%'.format(100.*yp/ncount), (xp.mean(), yp), 
                    ha='center', va='bottom') 
    if not hue:
        ax.set_title(df[x].describe().name + ' Analysis', size=12, pad=15)
    else:
        ax.set_title(df[x].describe().name + ' Analysis by ' + hue, size=12, pad=15)
    if title != '':
        ax.set_title(title)  
    plt.tight_layout()
```

### Data Load


```python
customers_ = pd.read_csv("./data/olist_customers_dataset.csv")
order_items_ = pd.read_csv("./data/olist_order_items_dataset.csv")
order_payments_ = pd.read_csv("./data/olist_order_payments_dataset.csv")
orders_ = pd.read_csv("./data/olist_orders_dataset.csv")

dataset = {
    'Customers': customers_,
    'Order Items': order_items_,
    'Payments': order_payments_,
    'Orders': orders_
}

for x, y in dataset.items():
    print(f'{x}', (list(y.shape)))
```

    Customers [99441, 5]
    Order Items [112650, 7]
    Payments [103886, 5]
    Orders [99441, 8]
    

### EDA
#### Columns Names


```python
for x, y in dataset.items():
    print(f'{x}', f'{list(y.columns)}\n')
```

    Customers ['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']
    
    Order Items ['order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date', 'price', 'freight_value']
    
    Payments ['order_id', 'payment_sequential', 'payment_type', 'payment_installments', 'payment_value']
    
    Orders ['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    
    

#### 결측치 확인


```python
# null values
for x, y in dataset.items():
    print(f"{x}: {y.isnull().any().any()}")
```

    Customers: False
    Order Items: False
    Payments: False
    Orders: True
    


```python
# missing values
for x, y in dataset.items():
    if y.isnull().any().any():
        print(f'{x}', (list(y.shape)),'\n')
        print(f'{y.isnull().sum()}\n')
```

    Orders [99441, 8] 
    
    order_id                            0
    customer_id                         0
    order_status                        0
    order_purchase_timestamp            0
    order_approved_at                 160
    order_delivered_carrier_date     1783
    order_delivered_customer_date    2965
    order_estimated_delivery_date       0
    dtype: int64
    
    

#### 통합 데이터 프레임 생성


```python
df1 = pd.merge(left=order_payments_, right=order_items_, on="order_id")
df2 = pd.merge(left=df1, right=orders_, on="order_id")
df = pd.merge(left=df2, right=customers_, on="customer_id")
print(df.shape)
```

    (117601, 22)
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>payment_sequential</th>
      <th>payment_type</th>
      <th>payment_installments</th>
      <th>payment_value</th>
      <th>order_item_id</th>
      <th>product_id</th>
      <th>seller_id</th>
      <th>shipping_limit_date</th>
      <th>price</th>
      <th>...</th>
      <th>order_status</th>
      <th>order_purchase_timestamp</th>
      <th>order_approved_at</th>
      <th>order_delivered_carrier_date</th>
      <th>order_delivered_customer_date</th>
      <th>order_estimated_delivery_date</th>
      <th>customer_unique_id</th>
      <th>customer_zip_code_prefix</th>
      <th>customer_city</th>
      <th>customer_state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b81ef226f3fe1789b1e8b2acac839d17</td>
      <td>1</td>
      <td>credit_card</td>
      <td>8</td>
      <td>99.33</td>
      <td>1</td>
      <td>af74cc53dcffc8384b29e7abfa41902b</td>
      <td>213b25e6f54661939f11710a6fddb871</td>
      <td>2018-05-02 22:15:09</td>
      <td>79.80</td>
      <td>...</td>
      <td>delivered</td>
      <td>2018-04-25 22:01:49</td>
      <td>2018-04-25 22:15:09</td>
      <td>2018-05-02 15:20:00</td>
      <td>2018-05-09 17:36:51</td>
      <td>2018-05-22 00:00:00</td>
      <td>708ab75d2a007f0564aedd11139c7708</td>
      <td>39801</td>
      <td>teofilo otoni</td>
      <td>MG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a9810da82917af2d9aefd1278f1dcfa0</td>
      <td>1</td>
      <td>credit_card</td>
      <td>1</td>
      <td>24.39</td>
      <td>1</td>
      <td>a630cc320a8c872f9de830cf121661a3</td>
      <td>eaf6d55068dea77334e8477d3878d89e</td>
      <td>2018-07-02 11:18:58</td>
      <td>17.00</td>
      <td>...</td>
      <td>delivered</td>
      <td>2018-06-26 11:01:38</td>
      <td>2018-06-26 11:18:58</td>
      <td>2018-06-28 14:18:00</td>
      <td>2018-06-29 20:32:09</td>
      <td>2018-07-16 00:00:00</td>
      <td>a8b9d3a27068454b1c98cc67d4e31e6f</td>
      <td>2422</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25e8ea4e93396b6fa0d3dd708e76c1bd</td>
      <td>1</td>
      <td>credit_card</td>
      <td>1</td>
      <td>65.71</td>
      <td>1</td>
      <td>2028bf1b01cafb2d2b1901fca4083222</td>
      <td>cc419e0650a3c5ba77189a1882b7556a</td>
      <td>2017-12-26 09:52:34</td>
      <td>56.99</td>
      <td>...</td>
      <td>delivered</td>
      <td>2017-12-12 11:19:55</td>
      <td>2017-12-14 09:52:34</td>
      <td>2017-12-15 20:13:22</td>
      <td>2017-12-18 17:24:41</td>
      <td>2018-01-04 00:00:00</td>
      <td>6f70c0b2f7552832ba46eb57b1c5651e</td>
      <td>2652</td>
      <td>sao paulo</td>
      <td>SP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ba78997921bbcdc1373bb41e913ab953</td>
      <td>1</td>
      <td>credit_card</td>
      <td>8</td>
      <td>107.78</td>
      <td>1</td>
      <td>548e5bfe28edceab6b51fa707cc9556f</td>
      <td>da8622b14eb17ae2831f4ac5b9dab84a</td>
      <td>2017-12-12 12:13:20</td>
      <td>89.90</td>
      <td>...</td>
      <td>delivered</td>
      <td>2017-12-06 12:04:06</td>
      <td>2017-12-06 12:13:20</td>
      <td>2017-12-07 20:28:28</td>
      <td>2017-12-21 01:35:51</td>
      <td>2018-01-04 00:00:00</td>
      <td>87695ed086ebd36f20404c82d20fca87</td>
      <td>36060</td>
      <td>juiz de fora</td>
      <td>MG</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42fdf880ba16b47b59251dd489d4441a</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2</td>
      <td>128.45</td>
      <td>1</td>
      <td>386486367c1f9d4f587a8864ccb6902b</td>
      <td>cca3071e3e9bb7d12640c9fbe2301306</td>
      <td>2018-05-31 16:14:41</td>
      <td>113.57</td>
      <td>...</td>
      <td>delivered</td>
      <td>2018-05-21 13:59:17</td>
      <td>2018-05-21 16:14:41</td>
      <td>2018-05-22 11:46:00</td>
      <td>2018-06-01 21:44:53</td>
      <td>2018-06-13 00:00:00</td>
      <td>4291db0da71914754618cd789aebcd56</td>
      <td>18570</td>
      <td>conchas</td>
      <td>SP</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



#### Converting DateTime


```python
date_columns = ["shipping_limit_date", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
for col in date_columns:
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
```

#### Cleaning up name columns


```python
df["customer_city"] = df["customer_city"].str.title()
df["payment_type"] = df["payment_type"].str.replace("_", " ").str.title()
```

#### 파생 변수 생성


```python
df['delivery_against_estimated'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
df['order_purchase_year'] = df["order_purchase_timestamp"].apply(lambda x: x.year)
df['order_purchase_month'] = df["order_purchase_timestamp"].apply(lambda x: x.month)
df['order_purchase_dayofweek'] = df["order_purchase_timestamp"].apply(lambda x: x.dayofweek)
df['order_purchase_hour'] = df["order_purchase_timestamp"].apply(lambda x: x.hour)
df['order_purchase_day'] = df['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
df['order_purchase_mon'] = df["order_purchase_timestamp"].apply(lambda x: x.month).map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})

df['month_year'] = df['order_purchase_month'].astype(str).apply(lambda x: '0' + x if len(x) == 1 else x)
df['month_year'] = df['order_purchase_year'].astype(str) + '-' + df['month_year'].astype(str)

df['month_y'] = df['order_purchase_timestamp'].map(lambda date: 100*date.year + date.month)
```

#### Summary


```python
df.describe(include="all")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>payment_sequential</th>
      <th>payment_type</th>
      <th>payment_installments</th>
      <th>payment_value</th>
      <th>order_item_id</th>
      <th>product_id</th>
      <th>seller_id</th>
      <th>shipping_limit_date</th>
      <th>price</th>
      <th>...</th>
      <th>customer_state</th>
      <th>delivery_against_estimated</th>
      <th>order_purchase_year</th>
      <th>order_purchase_month</th>
      <th>order_purchase_dayofweek</th>
      <th>order_purchase_hour</th>
      <th>order_purchase_day</th>
      <th>order_purchase_mon</th>
      <th>month_year</th>
      <th>month_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>117601</td>
      <td>117601.000000</td>
      <td>117601</td>
      <td>117601.000000</td>
      <td>117601.000000</td>
      <td>117601.000000</td>
      <td>117601</td>
      <td>117601</td>
      <td>117601</td>
      <td>117601.000000</td>
      <td>...</td>
      <td>117601</td>
      <td>115034.000000</td>
      <td>117601.000000</td>
      <td>117601.000000</td>
      <td>117601.000000</td>
      <td>117601.000000</td>
      <td>117601</td>
      <td>117601</td>
      <td>117601</td>
      <td>117601.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>98665</td>
      <td>NaN</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32951</td>
      <td>3095</td>
      <td>93317</td>
      <td>NaN</td>
      <td>...</td>
      <td>27</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>12</td>
      <td>24</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>895ab968e7bb0d5659d16cd74cd1650c</td>
      <td>NaN</td>
      <td>Credit Card</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>aca2eb7d00ea1a7b8ebd4e68314663af</td>
      <td>4a3ca9315b744ce9f8e9374361493884</td>
      <td>2017-08-14 20:43:31</td>
      <td>NaN</td>
      <td>...</td>
      <td>SP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mon</td>
      <td>Aug</td>
      <td>2017-11</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>63</td>
      <td>NaN</td>
      <td>86769</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>536</td>
      <td>2133</td>
      <td>63</td>
      <td>NaN</td>
      <td>...</td>
      <td>49566</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19130</td>
      <td>12632</td>
      <td>9016</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2016-09-19 00:15:34</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>last</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-04-09 22:35:08</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>1.093528</td>
      <td>NaN</td>
      <td>2.939482</td>
      <td>172.686752</td>
      <td>1.195900</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>120.824783</td>
      <td>...</td>
      <td>NaN</td>
      <td>11.043326</td>
      <td>2017.538193</td>
      <td>6.028129</td>
      <td>2.745750</td>
      <td>14.760002</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201759.847399</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>0.726692</td>
      <td>NaN</td>
      <td>2.774223</td>
      <td>267.592290</td>
      <td>0.697706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>184.479323</td>
      <td>...</td>
      <td>NaN</td>
      <td>10.162307</td>
      <td>0.505065</td>
      <td>3.229579</td>
      <td>1.961257</td>
      <td>5.325670</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>48.798820</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.850000</td>
      <td>...</td>
      <td>NaN</td>
      <td>-189.000000</td>
      <td>2016.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201609.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>60.870000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.900000</td>
      <td>...</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>2017.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>11.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201709.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>108.210000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.900000</td>
      <td>...</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>2018.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>15.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201801.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>189.260000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>134.900000</td>
      <td>...</td>
      <td>NaN</td>
      <td>16.000000</td>
      <td>2018.000000</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>19.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201805.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>29.000000</td>
      <td>NaN</td>
      <td>24.000000</td>
      <td>13664.080000</td>
      <td>21.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6735.000000</td>
      <td>...</td>
      <td>NaN</td>
      <td>146.000000</td>
      <td>2018.000000</td>
      <td>12.000000</td>
      <td>6.000000</td>
      <td>23.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>201809.000000</td>
    </tr>
  </tbody>
</table>
<p>13 rows × 31 columns</p>
</div>



#### 결측치 개수와 비율 확인


```python
missing_values = df.isnull().sum().sort_values(ascending = False)
percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_delivered_customer_date</th>
      <th>delivery_against_estimated</th>
      <th>order_delivered_carrier_date</th>
      <th>order_approved_at</th>
      <th>month_year</th>
      <th>order_purchase_mon</th>
      <th>order_purchase_day</th>
      <th>order_purchase_hour</th>
      <th>order_purchase_dayofweek</th>
      <th>order_purchase_month</th>
      <th>...</th>
      <th>freight_value</th>
      <th>price</th>
      <th>shipping_limit_date</th>
      <th>seller_id</th>
      <th>product_id</th>
      <th>order_item_id</th>
      <th>payment_value</th>
      <th>payment_installments</th>
      <th>payment_type</th>
      <th>month_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Values</th>
      <td>2567.000000</td>
      <td>2567.000000</td>
      <td>1245.000000</td>
      <td>15.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Percentage</th>
      <td>2.182805</td>
      <td>2.182805</td>
      <td>1.058664</td>
      <td>0.012755</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 31 columns</p>
</div>




```python
# 결측치 버리기
df.dropna(inplace=True)
df.isnull().values.any()
```




    False



#### Monthly Revenue


```python
df_revenue = df.groupby("month_year")["payment_value"].sum().reset_index()
df_revenue
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_year</th>
      <th>payment_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-10</td>
      <td>62591.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-12</td>
      <td>19.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01</td>
      <td>176376.56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-02</td>
      <td>323815.95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03</td>
      <td>505735.83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-04</td>
      <td>456108.32</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-05</td>
      <td>701119.60</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-06</td>
      <td>585400.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-07</td>
      <td>716069.98</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-08</td>
      <td>842689.94</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-09</td>
      <td>996085.61</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-10</td>
      <td>998609.62</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-11</td>
      <td>1548547.86</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-12</td>
      <td>1020067.26</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-01</td>
      <td>1374064.02</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018-02</td>
      <td>1280014.54</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2018-03</td>
      <td>1435458.33</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-04</td>
      <td>1466607.15</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018-05</td>
      <td>1480667.59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2018-06</td>
      <td>1285396.78</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018-07</td>
      <td>1306707.42</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018-08</td>
      <td>1211240.09</td>
    </tr>
  </tbody>
</table>
</div>



#### Monthly Revenue Growth Rate


```python
df_revenue["MonthlyGrowth"] = df_revenue["payment_value"].pct_change()
plt.figure(figsize=(20, 4))
_ = sns.pointplot(data=df_revenue, x="month_year", y="MonthlyGrowth", ci=None).set_title("월 매출 증감률")
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_28_0.png)
    


#### Monthly Active Customers


```python
df_monthly_active = df.groupby("month_year")["customer_unique_id"].nunique().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette="muted", color_codes=True, style="whitegrid")
bar_plot(x="month_year", y="customer_unique_id", df=df_monthly_active, value=True)
ax.tick_params(axis="x", labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_30_0.png)
    


#### Monthly Order Count


```python
df_monthly_sales = df.groupby("month_year")["order_status"].count().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette="muted", color_codes=True, style="whitegrid")
bar_plot(x="month_year", y="order_status", df=df_monthly_sales, value=True)
ax.tick_params(axis="x", labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_32_0.png)
    


#### ARPU (Average Revenue per Customer Purchase)


```python
df_monthly_order_avg = df.groupby('month_year')['payment_value'].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
bar_plot(x='month_year', y='payment_value', df=df_monthly_order_avg, value=True)
ax.tick_params(axis='x', labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_34_0.png)
    


#### 신규 가입자 비율


```python
df_min_purchase = df.groupby('customer_unique_id')["order_purchase_timestamp"].min().reset_index()
df_min_purchase.columns = ['customer_unique_id','minpurchasedate']
df_min_purchase['minpurchasedate'] = df_min_purchase['minpurchasedate'].map(lambda date: 100*date.year + date.month)

df = pd.merge(df, df_min_purchase, on='customer_unique_id')

df['usertype'] = 'New'
df.loc[df['month_y']>df['minpurchasedate'],'usertype'] = 'Existing'

df_user_type_revenue = df.groupby(['month_y','usertype', 'month_year'])['payment_value'].sum().reset_index()

df_user_type_revenue
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_y</th>
      <th>usertype</th>
      <th>month_year</th>
      <th>payment_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201610</td>
      <td>New</td>
      <td>2016-10</td>
      <td>62591.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201612</td>
      <td>New</td>
      <td>2016-12</td>
      <td>19.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201701</td>
      <td>Existing</td>
      <td>2017-01</td>
      <td>19.62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>201701</td>
      <td>New</td>
      <td>2017-01</td>
      <td>176356.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>201702</td>
      <td>Existing</td>
      <td>2017-02</td>
      <td>111.07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>201702</td>
      <td>New</td>
      <td>2017-02</td>
      <td>323704.88</td>
    </tr>
    <tr>
      <th>6</th>
      <td>201703</td>
      <td>Existing</td>
      <td>2017-03</td>
      <td>596.38</td>
    </tr>
    <tr>
      <th>7</th>
      <td>201703</td>
      <td>New</td>
      <td>2017-03</td>
      <td>505139.45</td>
    </tr>
    <tr>
      <th>8</th>
      <td>201704</td>
      <td>Existing</td>
      <td>2017-04</td>
      <td>2789.06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>201704</td>
      <td>New</td>
      <td>2017-04</td>
      <td>453319.26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>201705</td>
      <td>Existing</td>
      <td>2017-05</td>
      <td>6733.95</td>
    </tr>
    <tr>
      <th>11</th>
      <td>201705</td>
      <td>New</td>
      <td>2017-05</td>
      <td>694385.65</td>
    </tr>
    <tr>
      <th>12</th>
      <td>201706</td>
      <td>Existing</td>
      <td>2017-06</td>
      <td>6956.06</td>
    </tr>
    <tr>
      <th>13</th>
      <td>201706</td>
      <td>New</td>
      <td>2017-06</td>
      <td>578444.92</td>
    </tr>
    <tr>
      <th>14</th>
      <td>201707</td>
      <td>Existing</td>
      <td>2017-07</td>
      <td>13632.49</td>
    </tr>
    <tr>
      <th>15</th>
      <td>201707</td>
      <td>New</td>
      <td>2017-07</td>
      <td>702437.49</td>
    </tr>
    <tr>
      <th>16</th>
      <td>201708</td>
      <td>Existing</td>
      <td>2017-08</td>
      <td>15000.05</td>
    </tr>
    <tr>
      <th>17</th>
      <td>201708</td>
      <td>New</td>
      <td>2017-08</td>
      <td>827689.89</td>
    </tr>
    <tr>
      <th>18</th>
      <td>201709</td>
      <td>Existing</td>
      <td>2017-09</td>
      <td>14067.94</td>
    </tr>
    <tr>
      <th>19</th>
      <td>201709</td>
      <td>New</td>
      <td>2017-09</td>
      <td>982017.67</td>
    </tr>
    <tr>
      <th>20</th>
      <td>201710</td>
      <td>Existing</td>
      <td>2017-10</td>
      <td>20695.65</td>
    </tr>
    <tr>
      <th>21</th>
      <td>201710</td>
      <td>New</td>
      <td>2017-10</td>
      <td>977913.97</td>
    </tr>
    <tr>
      <th>22</th>
      <td>201711</td>
      <td>Existing</td>
      <td>2017-11</td>
      <td>25261.14</td>
    </tr>
    <tr>
      <th>23</th>
      <td>201711</td>
      <td>New</td>
      <td>2017-11</td>
      <td>1523286.72</td>
    </tr>
    <tr>
      <th>24</th>
      <td>201712</td>
      <td>Existing</td>
      <td>2017-12</td>
      <td>24133.48</td>
    </tr>
    <tr>
      <th>25</th>
      <td>201712</td>
      <td>New</td>
      <td>2017-12</td>
      <td>995933.78</td>
    </tr>
    <tr>
      <th>26</th>
      <td>201801</td>
      <td>Existing</td>
      <td>2018-01</td>
      <td>25079.90</td>
    </tr>
    <tr>
      <th>27</th>
      <td>201801</td>
      <td>New</td>
      <td>2018-01</td>
      <td>1348984.12</td>
    </tr>
    <tr>
      <th>28</th>
      <td>201802</td>
      <td>Existing</td>
      <td>2018-02</td>
      <td>26661.62</td>
    </tr>
    <tr>
      <th>29</th>
      <td>201802</td>
      <td>New</td>
      <td>2018-02</td>
      <td>1253352.92</td>
    </tr>
    <tr>
      <th>30</th>
      <td>201803</td>
      <td>Existing</td>
      <td>2018-03</td>
      <td>34684.83</td>
    </tr>
    <tr>
      <th>31</th>
      <td>201803</td>
      <td>New</td>
      <td>2018-03</td>
      <td>1400773.50</td>
    </tr>
    <tr>
      <th>32</th>
      <td>201804</td>
      <td>Existing</td>
      <td>2018-04</td>
      <td>41982.07</td>
    </tr>
    <tr>
      <th>33</th>
      <td>201804</td>
      <td>New</td>
      <td>2018-04</td>
      <td>1424625.08</td>
    </tr>
    <tr>
      <th>34</th>
      <td>201805</td>
      <td>Existing</td>
      <td>2018-05</td>
      <td>36272.26</td>
    </tr>
    <tr>
      <th>35</th>
      <td>201805</td>
      <td>New</td>
      <td>2018-05</td>
      <td>1444395.33</td>
    </tr>
    <tr>
      <th>36</th>
      <td>201806</td>
      <td>Existing</td>
      <td>2018-06</td>
      <td>40940.89</td>
    </tr>
    <tr>
      <th>37</th>
      <td>201806</td>
      <td>New</td>
      <td>2018-06</td>
      <td>1244455.89</td>
    </tr>
    <tr>
      <th>38</th>
      <td>201807</td>
      <td>Existing</td>
      <td>2018-07</td>
      <td>33086.53</td>
    </tr>
    <tr>
      <th>39</th>
      <td>201807</td>
      <td>New</td>
      <td>2018-07</td>
      <td>1273620.89</td>
    </tr>
    <tr>
      <th>40</th>
      <td>201808</td>
      <td>Existing</td>
      <td>2018-08</td>
      <td>27898.55</td>
    </tr>
    <tr>
      <th>41</th>
      <td>201808</td>
      <td>New</td>
      <td>2018-08</td>
      <td>1183341.54</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(15, 6))
sns.set(palette='muted', color_codes=True)
ax = sns.lineplot(x='month_year', y='payment_value', data=df_user_type_revenue.query("usertype == 'New'"), label='New')
ax = sns.lineplot(x='month_year', y='payment_value', data=df_user_type_revenue.query("usertype == 'Existing'"), label='Existing')
format_spines(ax, right_border=False)
ax.set_title('Existing vs New Customer Comparison')
ax.tick_params(axis='x', labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_37_0.png)
    



```python
df_user_ratio = df.query("usertype == 'New'").groupby(['month_year'])['customer_unique_id'].nunique()/df.query("usertype == 'Existing'").groupby(['month_year'])['customer_unique_id'].nunique() 
df_user_ratio = df_user_ratio.reset_index()

df_user_ratio = df_user_ratio.dropna()
df_user_ratio.columns = ['month_year','NewCusRatio']

df_user_ratio
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_year</th>
      <th>NewCusRatio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2017-01</td>
      <td>715.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-02</td>
      <td>808.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-03</td>
      <td>500.600000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2017-04</td>
      <td>125.333333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2017-05</td>
      <td>123.214286</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017-06</td>
      <td>77.871795</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017-07</td>
      <td>75.040000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017-08</td>
      <td>71.175439</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2017-09</td>
      <td>50.670886</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2017-10</td>
      <td>49.193182</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2017-11</td>
      <td>57.390244</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2017-12</td>
      <td>47.660714</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2018-01</td>
      <td>51.833333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2018-02</td>
      <td>56.151786</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2018-03</td>
      <td>48.385714</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-04</td>
      <td>40.629630</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2018-05</td>
      <td>34.791444</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2018-06</td>
      <td>32.103825</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2018-07</td>
      <td>39.377483</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2018-08</td>
      <td>37.012048</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
bar_plot(x='month_year', y='NewCusRatio', df=df_user_ratio, value=True)
ax.tick_params(axis='x', labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_39_0.png)
    


#### Monthly Retention Rate


```python
df_user_purchase = df.groupby(['customer_unique_id','month_y'])['payment_value'].sum().reset_index()
df_user_purchase.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_unique_id</th>
      <th>month_y</th>
      <th>payment_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>
      <td>201805</td>
      <td>141.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>
      <td>201805</td>
      <td>27.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000f46a3911fa3c0805444483337064</td>
      <td>201703</td>
      <td>86.22</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000f6ccb0745a6a4b88665a16c9f078</td>
      <td>201710</td>
      <td>43.62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004aac84e0df4da2b147fca70cf8255</td>
      <td>201711</td>
      <td>196.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_user_purchase = df.groupby(['customer_unique_id','month_y'])['payment_value'].count().reset_index()
df_user_purchase.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_unique_id</th>
      <th>month_y</th>
      <th>payment_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>
      <td>201805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>
      <td>201805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000f46a3911fa3c0805444483337064</td>
      <td>201703</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000f6ccb0745a6a4b88665a16c9f078</td>
      <td>201710</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004aac84e0df4da2b147fca70cf8255</td>
      <td>201711</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_retention = pd.crosstab(df_user_purchase['customer_unique_id'], df_user_purchase['month_y']).reset_index()
df_retention.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>month_y</th>
      <th>customer_unique_id</th>
      <th>201610</th>
      <th>201612</th>
      <th>201701</th>
      <th>201702</th>
      <th>201703</th>
      <th>201704</th>
      <th>201705</th>
      <th>201706</th>
      <th>201707</th>
      <th>...</th>
      <th>201711</th>
      <th>201712</th>
      <th>201801</th>
      <th>201802</th>
      <th>201803</th>
      <th>201804</th>
      <th>201805</th>
      <th>201806</th>
      <th>201807</th>
      <th>201808</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000366f3b9a7992bf8c76cfdf3221e2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000b849f77a49e4a4ce2b2a4ca5be3f</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000f46a3911fa3c0805444483337064</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000f6ccb0745a6a4b88665a16c9f078</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004aac84e0df4da2b147fca70cf8255</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
months = df_retention.columns[2:]
retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['month_y'] = int(selected_month)
    retention_data['TotalUserCount'] = df_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = df_retention[(df_retention[selected_month]>0) & (df_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    
df_retention = pd.DataFrame(retention_array)
df_retention['RetentionRate'] = df_retention['RetainedUserCount']/df_retention['TotalUserCount']

df_retention
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month_y</th>
      <th>TotalUserCount</th>
      <th>RetainedUserCount</th>
      <th>RetentionRate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>201701</td>
      <td>716</td>
      <td>1</td>
      <td>0.001397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>201702</td>
      <td>1618</td>
      <td>2</td>
      <td>0.001236</td>
    </tr>
    <tr>
      <th>2</th>
      <td>201703</td>
      <td>2508</td>
      <td>3</td>
      <td>0.001196</td>
    </tr>
    <tr>
      <th>3</th>
      <td>201704</td>
      <td>2274</td>
      <td>11</td>
      <td>0.004837</td>
    </tr>
    <tr>
      <th>4</th>
      <td>201705</td>
      <td>3478</td>
      <td>14</td>
      <td>0.004025</td>
    </tr>
    <tr>
      <th>5</th>
      <td>201706</td>
      <td>3076</td>
      <td>16</td>
      <td>0.005202</td>
    </tr>
    <tr>
      <th>6</th>
      <td>201707</td>
      <td>3802</td>
      <td>16</td>
      <td>0.004208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>201708</td>
      <td>4114</td>
      <td>23</td>
      <td>0.005591</td>
    </tr>
    <tr>
      <th>8</th>
      <td>201709</td>
      <td>4082</td>
      <td>32</td>
      <td>0.007839</td>
    </tr>
    <tr>
      <th>9</th>
      <td>201710</td>
      <td>4417</td>
      <td>32</td>
      <td>0.007245</td>
    </tr>
    <tr>
      <th>10</th>
      <td>201711</td>
      <td>7182</td>
      <td>37</td>
      <td>0.005152</td>
    </tr>
    <tr>
      <th>11</th>
      <td>201712</td>
      <td>5450</td>
      <td>41</td>
      <td>0.007523</td>
    </tr>
    <tr>
      <th>12</th>
      <td>201801</td>
      <td>6974</td>
      <td>16</td>
      <td>0.002294</td>
    </tr>
    <tr>
      <th>13</th>
      <td>201802</td>
      <td>6401</td>
      <td>27</td>
      <td>0.004218</td>
    </tr>
    <tr>
      <th>14</th>
      <td>201803</td>
      <td>6914</td>
      <td>23</td>
      <td>0.003327</td>
    </tr>
    <tr>
      <th>15</th>
      <td>201804</td>
      <td>6744</td>
      <td>31</td>
      <td>0.004597</td>
    </tr>
    <tr>
      <th>16</th>
      <td>201805</td>
      <td>6693</td>
      <td>45</td>
      <td>0.006723</td>
    </tr>
    <tr>
      <th>17</th>
      <td>201806</td>
      <td>6058</td>
      <td>38</td>
      <td>0.006273</td>
    </tr>
    <tr>
      <th>18</th>
      <td>201807</td>
      <td>6097</td>
      <td>26</td>
      <td>0.004264</td>
    </tr>
    <tr>
      <th>19</th>
      <td>201808</td>
      <td>6310</td>
      <td>37</td>
      <td>0.005864</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12, 6))
sns.set(palette='muted', color_codes=True, style='whitegrid')
bar_plot(x='month_y', y='RetentionRate', df=df_retention, value=True)
ax.tick_params(axis='x', labelrotation=90)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_45_0.png)
    


#### Cohort Based Retention Rate 


```python
df_retention = pd.crosstab(df_user_purchase['customer_unique_id'], df_user_purchase['month_y']).reset_index()
new_column_names = [ 'm_' + str(column) for column in df_retention.columns]
df_retention.columns = new_column_names

retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count =  retention_data['TotalUserCount'] = df_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1 
    
    query = "{} > 0".format('m_' + str(selected_month))
    

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(df_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)
    
    retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count =  retention_data['TotalUserCount'] = df_retention['m_' + str(selected_month)].sum()
    retention_data[selected_month] = 1 
    
    query = "{} > 0".format('m_' + str(selected_month))
    

    for next_month in next_months:
        query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(df_retention.query(query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)
    
df_retention = pd.DataFrame(retention_array)
df_retention.index = months

df_retention
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TotalUserCount</th>
      <th>201612</th>
      <th>201701</th>
      <th>201702</th>
      <th>201703</th>
      <th>201704</th>
      <th>201705</th>
      <th>201706</th>
      <th>201707</th>
      <th>201708</th>
      <th>...</th>
      <th>201711</th>
      <th>201712</th>
      <th>201801</th>
      <th>201802</th>
      <th>201803</th>
      <th>201804</th>
      <th>201805</th>
      <th>201806</th>
      <th>201807</th>
      <th>201808</th>
    </tr>
    <tr>
      <th>month_y</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>201612</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201701</th>
      <td>716</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201702</th>
      <td>1618</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201703</th>
      <td>2508</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201704</th>
      <td>2274</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201705</th>
      <td>3478</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201706</th>
      <td>3076</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201707</th>
      <td>3802</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.01</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201708</th>
      <td>4114</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201709</th>
      <td>4082</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201710</th>
      <td>4417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201711</th>
      <td>7182</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201712</th>
      <td>5450</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201801</th>
      <td>6974</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201802</th>
      <td>6401</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201803</th>
      <td>6914</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201804</th>
      <td>6744</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201805</th>
      <td>6693</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201806</th>
      <td>6058</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>201807</th>
      <td>6097</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>201808</th>
      <td>6310</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>21 rows × 22 columns</p>
</div>



### RFM
#### Recency


```python
df_user = pd.DataFrame(df['customer_unique_id'])
df_user.columns = ['customer_unique_id']
df_max_purchase = df.groupby('customer_unique_id')["order_purchase_timestamp"].max().reset_index()
df_max_purchase.columns = ['customer_unique_id', 'MaxPurchaseDate']
df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days

df_user = pd.merge(df_user, df_max_purchase[['customer_unique_id','Recency']], on='customer_unique_id')
```


```python
sns.set(palette='muted', color_codes=True, style='white')
fig, ax = plt.subplots(figsize=(12, 6))
sns.despine(left=True)
sns.distplot(df_user['Recency'], bins=30)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_50_0.png)
    



```python
# 군집화
from sklearn.cluster import KMeans

sse={}
df_recency = df_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_recency)
    df_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_51_0.png)
    



```python
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_user[['Recency']])
df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)
```


```python
df_user.groupby('RecencyCluster')['Recency'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>RecencyCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13777.0</td>
      <td>512.656311</td>
      <td>48.794950</td>
      <td>448.0</td>
      <td>473.0</td>
      <td>506.0</td>
      <td>543.0</td>
      <td>694.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18773.0</td>
      <td>381.158259</td>
      <td>35.208183</td>
      <td>323.0</td>
      <td>350.0</td>
      <td>380.0</td>
      <td>410.0</td>
      <td>447.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27171.0</td>
      <td>263.266755</td>
      <td>31.174965</td>
      <td>210.0</td>
      <td>234.0</td>
      <td>267.0</td>
      <td>285.0</td>
      <td>322.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30594.0</td>
      <td>154.050042</td>
      <td>31.607872</td>
      <td>102.0</td>
      <td>126.0</td>
      <td>155.0</td>
      <td>182.0</td>
      <td>209.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24703.0</td>
      <td>48.616646</td>
      <td>28.030105</td>
      <td>0.0</td>
      <td>24.0</td>
      <td>44.0</td>
      <td>73.0</td>
      <td>101.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Frequency


```python
df_frequency = df.groupby('customer_unique_id').order_purchase_timestamp.count().reset_index()
df_frequency.columns = ['customer_unique_id','Frequency']

df_user = pd.merge(df_user, df_frequency, on='customer_unique_id')
```


```python
sns.set(palette='muted', color_codes=True, style='whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))
sns.despine(left=True)
sns.distplot(df_user['Frequency'], hist=False)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_56_0.png)
    



```python
# 군집화
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_user[['Frequency']])
df_user['FrequencyCluster'] = kmeans.predict(df_user[['Frequency']])

df_user = order_cluster('FrequencyCluster', 'Frequency',df_user,True)

df_user.groupby('FrequencyCluster')['Frequency'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>FrequencyCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100150.0</td>
      <td>1.207968</td>
      <td>0.405856</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12428.0</td>
      <td>3.921307</td>
      <td>1.048323</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1968.0</td>
      <td>9.850610</td>
      <td>2.563686</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>397.0</td>
      <td>24.536524</td>
      <td>6.099081</td>
      <td>18.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.0</td>
      <td>75.000000</td>
      <td>0.000000</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>75.0</td>
      <td>75.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Revenue


```python
df_revenue = df.groupby('customer_unique_id').payment_value.sum().reset_index()

df_user = pd.merge(df_user, df_revenue, on='customer_unique_id')
```


```python
sns.set(palette='muted', color_codes=True, style='white')
fig, ax = plt.subplots(figsize=(12, 6))
sns.despine(left=True)
sns.distplot(df_user['payment_value'], hist=False)
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_60_0.png)
    



```python
sse={}
df_revenue = df_user[['payment_value']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_revenue)
    df_revenue["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
    
plt.figure(figsize=(10, 5))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_61_0.png)
    



```python
kmeans = KMeans(n_clusters=6)
kmeans.fit(df_user[['payment_value']])
df_user['RevenueCluster'] = kmeans.predict(df_user[['payment_value']])

df_user = order_cluster('RevenueCluster', 'payment_value',df_user,True)

df_user.groupby('RevenueCluster')['payment_value'].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>RevenueCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108472.0</td>
      <td>192.576355</td>
      <td>191.057389</td>
      <td>9.59</td>
      <td>67.53</td>
      <td>124.24</td>
      <td>233.535</td>
      <td>1033.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5565.0</td>
      <td>1881.126228</td>
      <td>817.833740</td>
      <td>1034.06</td>
      <td>1252.81</td>
      <td>1608.36</td>
      <td>2218.000</td>
      <td>4415.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>743.0</td>
      <td>7000.283419</td>
      <td>2137.860492</td>
      <td>4447.80</td>
      <td>5289.12</td>
      <td>6317.22</td>
      <td>7971.880</td>
      <td>12490.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>184.0</td>
      <td>20278.110435</td>
      <td>5207.417793</td>
      <td>14196.28</td>
      <td>16313.60</td>
      <td>19174.38</td>
      <td>25051.890</td>
      <td>30186.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46.0</td>
      <td>43587.292174</td>
      <td>2837.177072</td>
      <td>36489.24</td>
      <td>44048.00</td>
      <td>44048.00</td>
      <td>45256.000</td>
      <td>45256.00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.0</td>
      <td>109312.640000</td>
      <td>0.000000</td>
      <td>109312.64</td>
      <td>109312.64</td>
      <td>109312.64</td>
      <td>109312.640</td>
      <td>109312.64</td>
    </tr>
  </tbody>
</table>
</div>



#### Overall Score


```python
df_user.columns = ['customer_unique_id', 'Recency', 'RecencyCluster', 'Frequency', 'FrequencyCluster', 'Monetary', 'RevenueCluster']

df_user['OverallScore'] = df_user['RecencyCluster'] + df_user['FrequencyCluster'] + df_user['RevenueCluster']
df_user.groupby('OverallScore')['Recency','Frequency','Monetary'].mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Recency</th>
      <th>Frequency</th>
      <th>Monetary</th>
    </tr>
    <tr>
      <th>OverallScore</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>511.759001</td>
      <td>1.207100</td>
      <td>167.861348</td>
    </tr>
    <tr>
      <th>1</th>
      <td>390.772210</td>
      <td>1.352797</td>
      <td>195.827734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>276.351342</td>
      <td>1.468897</td>
      <td>222.252763</td>
    </tr>
    <tr>
      <th>3</th>
      <td>170.635585</td>
      <td>1.577970</td>
      <td>251.513166</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70.135504</td>
      <td>1.674742</td>
      <td>306.303808</td>
    </tr>
    <tr>
      <th>5</th>
      <td>105.696962</td>
      <td>5.948294</td>
      <td>1120.916258</td>
    </tr>
    <tr>
      <th>6</th>
      <td>99.065306</td>
      <td>6.785034</td>
      <td>3074.301905</td>
    </tr>
    <tr>
      <th>7</th>
      <td>110.812245</td>
      <td>11.448980</td>
      <td>5401.498551</td>
    </tr>
    <tr>
      <th>8</th>
      <td>143.403226</td>
      <td>11.462366</td>
      <td>22318.521129</td>
    </tr>
    <tr>
      <th>10</th>
      <td>96.454545</td>
      <td>22.181818</td>
      <td>35259.341818</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_user['Segment'] = 'Low-Value'
df_user.loc[df_user['OverallScore']>3,'Segment'] = 'Mid-Value' 
df_user.loc[df_user['OverallScore']>6,'Segment'] = 'High-Value' 
```


```python
sns.set(palette='muted', color_codes=True, style='whitegrid')
fig, axs = plt.subplots(1, 3, figsize=(22, 5))
sns.despine(left=True)
sns.scatterplot(x='Recency', y='Frequency', ax=axs[0], hue='Segment', data=df_user, size='Segment', sizes=(50,150), size_order=['High-Value','Mid-Value','Low-Value'])
sns.scatterplot(x='Frequency', y='Monetary', ax=axs[1], hue='Segment', data=df_user, size='Segment' , sizes=(50,150), size_order=['High-Value','Mid-Value','Low-Value'])
sns.scatterplot(x='Recency', y='Monetary', ax=axs[2], hue='Segment', data=df_user, size='Segment' , sizes=(50,150), size_order=['High-Value','Mid-Value','Low-Value'])
axs[0].set_title('Customer Segments by Recency & Frequency')
axs[1].set_title('Customer Segments by Frequency & Monetary')
axs[2].set_title('Customer Segments by Recency & Monetary')
plt.show()
```


    
![png](olist_eda_RFM_files/olist_eda_RFM_66_0.png)
    

### 마무리
그래프 그리는 함수를 미리 정의해 놓고 사용했다는 점과, 등급을 나누기 위해 군집화를 사용했다는 점이 신선했다.