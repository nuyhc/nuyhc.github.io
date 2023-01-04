---
title: Kaggle - New York City Taxi Trip Duration
date: 2022-07-07T10:13:44.849Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - XGBoost
  - lightGBM
  - Pandas
  - Numpy
  - Seaborn
  - matplot
  - sklearn
---

# Kaggle - New York City Taxi Trip Duration
## EDA and Preprocessing
### Data
[Kaggle - New York City Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data)
### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgbm
from lightgbm import LGBMRegressor
```

### Data Load


```python
path = glob.glob("data/*")
path
```




    ['data\\pre_test.csv',
     'data\\pre_train.csv',
     'data\\sample_submission.zip',
     'data\\test.zip',
     'data\\train.zip']




```python
train, test = pd.read_csv(path[-1]), pd.read_csv(path[-2])

train.shape, test.shape
```




    ((1458644, 11), (625134, 9))




```python
display(train.head())
display(test.head())
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
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>2124</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id3004672</td>
      <td>1</td>
      <td>2016-06-30 23:59:58</td>
      <td>1</td>
      <td>-73.988129</td>
      <td>40.732029</td>
      <td>-73.990173</td>
      <td>40.756680</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id3505355</td>
      <td>1</td>
      <td>2016-06-30 23:59:53</td>
      <td>1</td>
      <td>-73.964203</td>
      <td>40.679993</td>
      <td>-73.959808</td>
      <td>40.655403</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id1217141</td>
      <td>1</td>
      <td>2016-06-30 23:59:47</td>
      <td>1</td>
      <td>-73.997437</td>
      <td>40.737583</td>
      <td>-73.986160</td>
      <td>40.729523</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2150126</td>
      <td>2</td>
      <td>2016-06-30 23:59:41</td>
      <td>1</td>
      <td>-73.956070</td>
      <td>40.771900</td>
      <td>-73.986427</td>
      <td>40.730469</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id1598245</td>
      <td>1</td>
      <td>2016-06-30 23:59:33</td>
      <td>1</td>
      <td>-73.970215</td>
      <td>40.761475</td>
      <td>-73.961510</td>
      <td>40.755890</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>


- `vendor_id`: 제공회사
- `store_and_fwd_flag`: 회사로 데이터를 전송하는지

### EDA and Preprocessing
#### 기본정보


```python
display(train.dtypes)
display(test.dtypes)
```


    id                     object
    vendor_id               int64
    pickup_datetime        object
    dropoff_datetime       object
    passenger_count         int64
    pickup_longitude      float64
    pickup_latitude       float64
    dropoff_longitude     float64
    dropoff_latitude      float64
    store_and_fwd_flag     object
    trip_duration           int64
    dtype: object



    id                     object
    vendor_id               int64
    pickup_datetime        object
    passenger_count         int64
    pickup_longitude      float64
    pickup_latitude       float64
    dropoff_longitude     float64
    dropoff_latitude      float64
    store_and_fwd_flag     object
    dtype: object


#### 결측치


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
sns.heatmap(train.isnull(), ax=ax[0]).set_title("Train - Missing")
sns.heatmap(test.isnull(), ax=ax[1]).set_title("Test - Missing")
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_10_0.png?raw=true)
    


결측치가 없는 데이터임

#### 중복값


```python
train[train.duplicated()]
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
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
test[test.duplicated()]
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
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



중복값도 없음

#### vendor_id


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=train["vendor_id"], ax=ax[0]).set_title("vendor_id (Train)")
sns.countplot(x=test["vendor_id"], ax=ax[1]).set_title("vendor_id (Test)")
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_16_0.png?raw=true)
    


#### pickup_datetime / dropoff_datetime


```python
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["dropoff_datetime"] = pd.to_datetime(train["dropoff_datetime"])

test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
```


```python
# train - pickup
train["p_year"] = train["pickup_datetime"].dt.year
train["p_month"] = train["pickup_datetime"].dt.month
train["p_day"] = train["pickup_datetime"].dt.day
train["p_dow"] = train["pickup_datetime"].dt.dayofweek
train["p_hour"] = train["pickup_datetime"].dt.hour
train["p_min"] = train["pickup_datetime"].dt.minute
```


```python
# # train - dropoff
# train["d_year"] = train["dropoff_datetime"].dt.year
# train["d_month"] = train["dropoff_datetime"].dt.month
# train["d_day"] = train["dropoff_datetime"].dt.day
# train["d_dow"] = train["dropoff_datetime"].dt.dayofweek
# train["d_hour"] = train["dropoff_datetime"].dt.hour
# train["d_min"] = train["dropoff_datetime"].dt.minute
```

`dropoff_datetime`은 `test`에 없으므로 사용 안함


```python
# test - pickup
test["p_year"] = test["pickup_datetime"].dt.year
test["p_month"] = test["pickup_datetime"].dt.month
test["p_day"] = test["pickup_datetime"].dt.day
test["p_dow"] = test["pickup_datetime"].dt.dayofweek
test["p_hour"] = test["pickup_datetime"].dt.hour
test["p_min"] = test["pickup_datetime"].dt.minute
```


```python
date = ["p_year", "p_month", "p_day", "p_dow", "p_hour", "p_min"]
```


```python
fig, ax = plt.subplots(1, len(date), figsize=(30, 5))
for col, ax in zip(date, ax):
    sns.countplot(x=train[col], ax=ax).set_title(f"{col} - Train")
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_24_0.png?raw=true)
    



```python
fig, ax = plt.subplots(1, len(date), figsize=(30, 5))
for col, ax in zip(date, ax):
    sns.countplot(x=test[col], ax=ax).set_title(f"{col} - Test")
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_25_0.png?raw=true)
    


2016년 상반기(1~6) 데이터임  
이동 거리를 예측해야하는 문제라 `test`에는 `dropoff_datetime`이 없는 듯

#### longitude / latitude


```python
cor = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
```

#### Haversine formula
$ haversine(θ) = sin²{θ \over 2} $  

a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)  
c = 2 * atan2( √a, √(1−a) )  
d = R ⋅ c  
d = Haversine distance


```python
def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for _ in data:
        R = 6371 # km, 지구의 반지름
        phi1 = np.radians(_[lat1])
        phi2 = np.radians(_[lat2])
        
        delta_phi = np.radians(_[lat2]-_[lat1])
        delta_lambda = np.radians(_[long2]-_[long2])
        
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = (R * c)
        _["H_Distance"] = d
    return d
```


```python
haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
```




    0          2.741019
    1          2.734232
    2          0.896282
    3          4.606964
    4          0.620992
                ...    
    625129     0.949304
    625130     4.301558
    625131     1.245378
    625132    17.593930
    625133     5.837496
    Length: 625134, dtype: float64



#### passenger_count


```python
_ = (train["passenger_count"].value_counts().sort_index()).plot.bar()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_32_0.png?raw=true)
    



```python
train["passenger_count"].value_counts().sort_index()
```




    0         60
    1    1033540
    2     210318
    3      59896
    4      28404
    5      78088
    6      48333
    7          3
    8          1
    9          1
    Name: passenger_count, dtype: int64



0, 7, 8, 9명이 탄것은 이상치  
5명 이상도 조금 이상한거 같지만, 대형 택시라고 생각함


```python
cond1 = train["passenger_count"]==0
cond2 = train["passenger_count"]==7
cond3 = train["passenger_count"]==8
cond4 = train["passenger_count"]==9

cond = cond1 | cond2 | cond3 | cond4
```


```python
train = train[~cond]
```

#### store_and_fwd_flag


```python
_ = sns.countplot(x=train["store_and_fwd_flag"])
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_38_0.png?raw=true)
    


## Model - Regressor
### Train Data Split


```python
label = "trip_duration"
features = train.columns.tolist()
features.remove(label)
features.remove("id")
features.remove("pickup_datetime")
features.remove("dropoff_datetime")
features
```




    ['vendor_id',
     'passenger_count',
     'pickup_longitude',
     'pickup_latitude',
     'dropoff_longitude',
     'dropoff_latitude',
     'store_and_fwd_flag',
     'p_year',
     'p_month',
     'p_day',
     'p_dow',
     'p_hour',
     'p_min',
     'H_Distance']




```python
train["store_and_fwd_flag"] = pd.get_dummies(train["store_and_fwd_flag"], drop_first=True)
test["store_and_fwd_flag"] = pd.get_dummies(test["store_and_fwd_flag"], drop_first=True)
```


```python
X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.3)

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")
```

    X_train: (1021005, 14)
    y_train: (1021005,)
    X_test: (437574, 14)
    y_test: (437574,)
    

#### Random Forest


```python
reg_rf = RandomForestRegressor()

pred_rf = reg_rf.fit(X_train, y_train).predict(X_test)

mean_squared_log_error(y_test, pred_rf)
```




    0.3483171563954924




```python
_ = sns.barplot(x=reg_rf.feature_importances_, y=reg_rf.feature_names_in_)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_45_0.png?raw=true)
    


#### XGBoost


```python
reg_xgb = XGBRegressor()

pred_xgb = reg_xgb.fit(X_train, y_train).predict(X_test)

mean_squared_log_error(y_test, abs(pred_xgb))
```




    0.3297185297357314




```python
_ = xgb.plot_importance(reg_xgb)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_48_0.png?raw=true)
    


#### LGBM


```python
reg_lgbm = LGBMRegressor()

pred_lgbm = reg_lgbm.fit(X_train, y_train).predict(X_test)

mean_squared_log_error(y_test, abs(pred_lgbm))
```




    0.3620231060763725




```python
_ = lgbm.plot_importance(reg_lgbm)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/NY_taxi_trip_duratioin_files/NY_taxi_trip_duratioin_51_0.png?raw=true)
    

