---
title: 영화 관객수 예측 경진대회
date: 2022-07-14T08:32:19.715Z

categories:
  - Programming
  - Machine Learning
tags:
  - Pandas
  - Seaborn
  - matplot
  - Numpy
---

# DACON - 영화 관객수 예측 경진대회
[DACON - 영화 관객수 예측 경진대회](https://dacon.io/competitions/open/235536/data)

너무 귀찮아서 대충했다보니 성능이 참.. EDA를 집중해서 진행하지 못했다.  

## EDA and Preprocessing

### 사용 라이브러리




```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import re
import glob
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import xgboost as xgb
from xgboost import XGBRegressor

import lightgbm as lgbm
from lightgbm import LGBMRegressor
```

### Data Load


```python
paths = glob.glob("./data/*")
paths
```




    ['./data\\movies_test.csv',
     './data\\movies_train.csv',
     './data\\submission.csv']




```python
train, test = pd.read_csv(paths[1]), pd.read_csv(paths[0])

train.shape, test.shape
```




    ((600, 12), (243, 11))




```python
display(train.head(3))
display(test.head(3))
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>개들의 전쟁</td>
      <td>롯데엔터테인먼트</td>
      <td>액션</td>
      <td>2012-11-22</td>
      <td>96</td>
      <td>청소년 관람불가</td>
      <td>조병옥</td>
      <td>NaN</td>
      <td>0</td>
      <td>91</td>
      <td>2</td>
      <td>23398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>내부자들</td>
      <td>(주)쇼박스</td>
      <td>느와르</td>
      <td>2015-11-19</td>
      <td>130</td>
      <td>청소년 관람불가</td>
      <td>우민호</td>
      <td>1161602.50</td>
      <td>2</td>
      <td>387</td>
      <td>3</td>
      <td>7072501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>은밀하게 위대하게</td>
      <td>(주)쇼박스</td>
      <td>액션</td>
      <td>2013-06-05</td>
      <td>123</td>
      <td>15세 관람가</td>
      <td>장철수</td>
      <td>220775.25</td>
      <td>4</td>
      <td>343</td>
      <td>4</td>
      <td>6959083</td>
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
      <th>title</th>
      <th>distributor</th>
      <th>genre</th>
      <th>release_time</th>
      <th>time</th>
      <th>screening_rat</th>
      <th>director</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>시네마서비스</td>
      <td>느와르</td>
      <td>2010-01-07</td>
      <td>125</td>
      <td>청소년 관람불가</td>
      <td>김형준</td>
      <td>3.005290e+05</td>
      <td>2</td>
      <td>304</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>(주)쇼박스</td>
      <td>멜로/로맨스</td>
      <td>2010-01-14</td>
      <td>113</td>
      <td>12세 관람가</td>
      <td>이광재</td>
      <td>3.427002e+05</td>
      <td>4</td>
      <td>275</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>CJ 엔터테인먼트</td>
      <td>드라마</td>
      <td>2010-01-28</td>
      <td>115</td>
      <td>12세 관람가</td>
      <td>강대규</td>
      <td>4.206611e+06</td>
      <td>3</td>
      <td>419</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


- title : 영화의 제목
- distributor : 배급사
- genre : 장르
- release_time : 개봉일
- time : 상영시간(분)
- screening_rat : 상영등급
- director : 감독이름
- dir_prev_bfnum : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수(단 관객수가 알려지지 않은 영화 제외)
- dir_prev_num : 해당 감독이 이 영화를 만들기 전 제작에 참여한 영화의 개수(단 관객수가 알려지지 않은 영화 제외)
- num_staff : 스텝수
- num_actor : 주연배우수
- box_off_num : 관객수

### EDA and Preprocessing
#### 기본 정보


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 600 entries, 0 to 599
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           600 non-null    object 
     1   distributor     600 non-null    object 
     2   genre           600 non-null    object 
     3   release_time    600 non-null    object 
     4   time            600 non-null    int64  
     5   screening_rat   600 non-null    object 
     6   director        600 non-null    object 
     7   dir_prev_bfnum  270 non-null    float64
     8   dir_prev_num    600 non-null    int64  
     9   num_staff       600 non-null    int64  
     10  num_actor       600 non-null    int64  
     11  box_off_num     600 non-null    int64  
    dtypes: float64(1), int64(5), object(6)
    memory usage: 56.4+ KB
    


```python
train.describe().round(2)
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
      <th>time</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>600.00</td>
      <td>270.00</td>
      <td>600.00</td>
      <td>600.00</td>
      <td>600.00</td>
      <td>600.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>100.86</td>
      <td>1050442.89</td>
      <td>0.88</td>
      <td>151.12</td>
      <td>3.71</td>
      <td>708181.75</td>
    </tr>
    <tr>
      <th>std</th>
      <td>18.10</td>
      <td>1791408.30</td>
      <td>1.18</td>
      <td>165.65</td>
      <td>2.45</td>
      <td>1828005.85</td>
    </tr>
    <tr>
      <th>min</th>
      <td>45.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>89.00</td>
      <td>20380.00</td>
      <td>0.00</td>
      <td>17.00</td>
      <td>2.00</td>
      <td>1297.25</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>100.00</td>
      <td>478423.62</td>
      <td>0.00</td>
      <td>82.50</td>
      <td>3.00</td>
      <td>12591.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>114.00</td>
      <td>1286568.62</td>
      <td>2.00</td>
      <td>264.00</td>
      <td>4.00</td>
      <td>479886.75</td>
    </tr>
    <tr>
      <th>max</th>
      <td>180.00</td>
      <td>17615314.00</td>
      <td>5.00</td>
      <td>869.00</td>
      <td>25.00</td>
      <td>14262766.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 243 entries, 0 to 242
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   title           243 non-null    object 
     1   distributor     243 non-null    object 
     2   genre           243 non-null    object 
     3   release_time    243 non-null    object 
     4   time            243 non-null    int64  
     5   screening_rat   243 non-null    object 
     6   director        243 non-null    object 
     7   dir_prev_bfnum  107 non-null    float64
     8   dir_prev_num    243 non-null    int64  
     9   num_staff       243 non-null    int64  
     10  num_actor       243 non-null    int64  
    dtypes: float64(1), int64(4), object(6)
    memory usage: 21.0+ KB
    


```python
test.describe().round(2)
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
      <th>time</th>
      <th>dir_prev_bfnum</th>
      <th>dir_prev_num</th>
      <th>num_staff</th>
      <th>num_actor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>243.00</td>
      <td>107.00</td>
      <td>243.00</td>
      <td>243.00</td>
      <td>243.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>109.80</td>
      <td>891669.52</td>
      <td>0.85</td>
      <td>159.32</td>
      <td>3.48</td>
    </tr>
    <tr>
      <th>std</th>
      <td>124.02</td>
      <td>1217341.45</td>
      <td>1.20</td>
      <td>162.98</td>
      <td>2.11</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40.00</td>
      <td>34.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>91.00</td>
      <td>62502.00</td>
      <td>0.00</td>
      <td>18.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>104.00</td>
      <td>493120.00</td>
      <td>0.00</td>
      <td>105.00</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>114.50</td>
      <td>1080849.58</td>
      <td>1.00</td>
      <td>282.00</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.00</td>
      <td>6173099.50</td>
      <td>6.00</td>
      <td>776.00</td>
      <td>16.00</td>
    </tr>
  </tbody>
</table>
</div>



#### 결측치 확인


```python
train.isnull().sum()
```




    title               0
    distributor         0
    genre               0
    release_time        0
    time                0
    screening_rat       0
    director            0
    dir_prev_bfnum    330
    dir_prev_num        0
    num_staff           0
    num_actor           0
    box_off_num         0
    dtype: int64




```python
test.isnull().sum()
```




    title               0
    distributor         0
    genre               0
    release_time        0
    time                0
    screening_rat       0
    director            0
    dir_prev_bfnum    136
    dir_prev_num        0
    num_staff           0
    num_actor           0
    dtype: int64




```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
_ = sns.heatmap(train.isnull(), ax=ax[0]).set_title("Train - Missing")
_ = sns.heatmap(test.isnull(), ax=ax[1]).set_title("Test - Missing")
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_15_0.png)
    


`dir_prev_bfnum`은 해당 감독이 영화를 만들기 전 제작에 참여한 영화에서의 평균 관객수부분에 결측치가 존재함  
관객수가 알려지지 않은 부분이 결측치로 존재하는거라, 정보가 없다라는 정보 그 자체로 사용해도 괜찮을꺼 같음

#### distributor: 배급사


```python
train["distributor"] = train["distributor"].str.replace("\(|주|\)", "", regex=True)
test["distributor"] = test["distributor"].str.replace("\(|주|\)", "", regex=True)
```


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(x=train["distributor"], ax=ax[0]).set_title("Train - distributor")
sns.countplot(x=test["distributor"], ax=ax[1]).set_title("Test - distributor")
plt.show()
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_19_0.png)
    



```python
# 정규 표현식으로 문자와 숫자만 
train["distributor"] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in train.distributor]
test['distributor'] = [re.sub(r'[^0-9a-zA-Z가-힣]', '', x) for x in test.distributor]
```


```python
_ = train["distributor"].value_counts().hist()
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_21_0.png)
    



```python
def distributor_band(col, d=dict(train["distributor"].value_counts())):
    try:
        if d[col]<=15: return "소형"
        else: return "중대형"
    except:
        return "소형"
```


```python
train["distributor"].apply(distributor_band).value_counts()
```




    소형     357
    중대형    243
    Name: distributor, dtype: int64




```python
train["distributor"] = train["distributor"].apply(distributor_band)
test["distributor"] = test["distributor"].apply(distributor_band)
```


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].pie(train["distributor"].value_counts().values, labels=train["distributor"].value_counts().index, autopct="%.2f%%")
ax[1].pie(test["distributor"].value_counts().values, labels=test["distributor"].value_counts().index, autopct="%.2f%%")
plt.show()
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_25_0.png)
    


#### genre: 장르



```python
train.groupby("genre")["box_off_num"].mean().sort_values()
```




    genre
    뮤지컬       6.627000e+03
    다큐멘터리     6.717226e+04
    서스펜스      8.261100e+04
    애니메이션     1.819267e+05
    멜로/로맨스    4.259680e+05
    미스터리      5.275482e+05
    공포        5.908325e+05
    드라마       6.256898e+05
    코미디       1.193914e+06
    SF        1.788346e+06
    액션        2.203974e+06
    느와르       2.263695e+06
    Name: box_off_num, dtype: float64




```python
rank = {'뮤지컬' : 1, '다큐멘터리' : 2, '서스펜스' : 3, '애니메이션' : 4, '멜로/로맨스' : 5, '미스터리' : 6, '공포' : 7, '드라마' : 8, '코미디' : 9, 'SF' : 10, '액션' : 11, '느와르' : 12}
```


```python
train["rank_genre"] = train["genre"].apply(lambda x: rank[x])
test["rank_genre"] = test["genre"].apply(lambda x: rank[x])
```


```python
train.drop(columns="genre", axis=1, inplace=True)
test.drop(columns="genre", axis=1, inplace=True)
```

### release_time: 개봉일


```python
train["release_time"] = pd.to_datetime(train["release_time"])
test["release_time"] = pd.to_datetime(test["release_time"])
```


```python
target = [train, test]

for t in target:
    t["year"] = t["release_time"].dt.year
    t["month"] = t["release_time"].dt.month
    t["day"] = t["release_time"].dt.day
    t["dayofweek"] = t["release_time"].dt.dayofweek
```


```python
train.drop(columns="release_time", axis=1, inplace=True)
test.drop(columns="release_time", axis=1, inplace=True)
```


```python
date = ["year", "month", "day", "dayofweek"]

fig, ax = plt.subplots(1, len(date), figsize=(24, 7))
for col_name, ax in zip(date, ax):
    sns.countplot(data=train, x=col_name, ax=ax).set_title(f"{col_name}")
plt.show()
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_35_0.png)
    


수요일, 목요일에 개봉한 영화들이 많고 목요일에 개봉한 영화가 유독 많음

## Train


```python
train.drop(columns=["title", "director"], axis=1, inplace=True)
test.drop(columns=["title", "director"], axis=1, inplace=True)
```


```python
train["dir_prev_bfnum"].fillna(0, inplace=True)
test["dir_prev_bfnum"].fillna(0, inplace=True)
```


```python
train = pd.get_dummies(train, columns=["distributor", "screening_rat"])
test = pd.get_dummies(test, columns=["distributor", "screening_rat"])
```


```python
label = "box_off_num"
features = train.columns.tolist()
features.remove(label)
features
```




    ['time',
     'dir_prev_bfnum',
     'dir_prev_num',
     'num_staff',
     'num_actor',
     'rank_genre',
     'year',
     'month',
     'day',
     'dayofweek',
     'distributor_소형',
     'distributor_중대형',
     'screening_rat_12세 관람가',
     'screening_rat_15세 관람가',
     'screening_rat_전체 관람가',
     'screening_rat_청소년 관람불가']



### Log Scale


```python
train["num_actor"] = np.log1p(train["num_actor"])
test["num_actor"] = np.log1p(test["num_actor"])
```


```python
X_train, X_val, y_train, y_val = train_test_split(train[features], train[label], test_size=0.15)

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_val: {X_val.shape}\ny_val: {y_val.shape}")
```

    X_train: (510, 16)
    y_train: (510,)
    X_val: (90, 16)
    y_val: (90,)
    


```python
test.shape
```




    (243, 16)



### Random Forest


```python
reg_rf = RandomForestRegressor()

pred_rf = reg_rf.fit(X_train, y_train).predict(X_val)

print(f"rmse: {np.sqrt(mean_squared_error(y_val, pred_rf))}")

```

    rmse: 1377636.6138653848
    


```python
_ = sns.barplot(x=reg_rf.feature_importances_, y=reg_rf.feature_names_in_)
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_47_0.png)
    


### XGBoost


```python
reg_xgb = XGBRegressor()

pred_xgb = reg_xgb.fit(X_train, y_train).predict(X_val)

print(f"rmse: {np.sqrt(mean_squared_error(y_val, pred_xgb))}")

```

    rmse: 1444161.2032999645
    


```python
_ = xgb.plot_importance(reg_xgb)
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_50_0.png)
    


### LightGBM


```python
reg_lgbm = LGBMRegressor()

pred_lgbm = reg_lgbm.fit(X_train, y_train).predict(X_val)

print(f"rmse: {np.sqrt(mean_squared_error(y_val, pred_lgbm))}")

```

    rmse: 1444161.2032999645
    


```python
_ = lgbm.plot_importance(reg_lgbm)
```


    
![png](/assets/images/sourceImg/movie_prediction_files/movie_prediction_53_0.png)
    


## Submit


```python
sub = pd.read_csv(paths[2])
sub.head()
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
      <th>title</th>
      <th>box_off_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>용서는 없다</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>아빠가 여자를 좋아해</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>하모니</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>의형제</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>평행 이론</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred = (0.6*reg_rf.predict(X_val)) + (0.2*reg_xgb.predict(X_val)) + (0.2*reg_lgbm.predict(X_val))
print(f"rmse: {np.sqrt(mean_squared_error(y_val, pred))}")
```

    rmse: 1368430.7286695272
    


```python
pred = (0.6*reg_rf.predict(test)) + (0.2*reg_xgb.predict(test)) + (0.2*reg_lgbm.predict(test))
```


```python
sub["box_off_num"] = pred
```


```python
sub.to_csv("sub_rmse_1368.csv", index=False)
```
