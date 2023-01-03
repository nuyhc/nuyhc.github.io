---
title: DACON 쇼핑몰 지점별 매출액 예측 경진대회 1 EDA/Preprocessing
date: 2022-07-19T14:32:56.883Z

categories:
  - Programming
  - Machine Learning
tags:
  - Pandas
  - Numpy
  - matplot
  - Seaborn
  - sklearn
---

# DACON - 쇼핑몰 지점별 매출액 예측 경진대회
## EDA and Preprocessing
### Data
[DACON - 쇼핑몰 지점별 매출액 예측 경진대회](https://dacon.io/competitions/official/235942/data)

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import glob
import warnings
warnings.filterwarnings("ignore")
```

### Data Load


```python
train, test = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")

train.shape, test.shape
```




    ((6255, 13), (180, 12))




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
      <th>id</th>
      <th>Store</th>
      <th>Date</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>Promotion1</th>
      <th>Promotion2</th>
      <th>Promotion3</th>
      <th>Promotion4</th>
      <th>Promotion5</th>
      <th>Unemployment</th>
      <th>IsHoliday</th>
      <th>Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>05/02/2010</td>
      <td>42.31</td>
      <td>2.572</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.106</td>
      <td>False</td>
      <td>1643690.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>12/02/2010</td>
      <td>38.51</td>
      <td>2.548</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.106</td>
      <td>True</td>
      <td>1641957.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>19/02/2010</td>
      <td>39.93</td>
      <td>2.514</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.106</td>
      <td>False</td>
      <td>1611968.17</td>
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
      <th>Store</th>
      <th>Date</th>
      <th>Temperature</th>
      <th>Fuel_Price</th>
      <th>Promotion1</th>
      <th>Promotion2</th>
      <th>Promotion3</th>
      <th>Promotion4</th>
      <th>Promotion5</th>
      <th>Unemployment</th>
      <th>IsHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>05/10/2012</td>
      <td>68.55</td>
      <td>3.617</td>
      <td>8077.89</td>
      <td>NaN</td>
      <td>18.22</td>
      <td>3617.43</td>
      <td>3626.14</td>
      <td>6.573</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>12/10/2012</td>
      <td>62.99</td>
      <td>3.601</td>
      <td>2086.18</td>
      <td>NaN</td>
      <td>8.11</td>
      <td>602.36</td>
      <td>5926.45</td>
      <td>6.573</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>19/10/2012</td>
      <td>67.97</td>
      <td>3.594</td>
      <td>950.33</td>
      <td>NaN</td>
      <td>4.93</td>
      <td>80.25</td>
      <td>2312.85</td>
      <td>6.573</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>


### EDA and Preprocessing
#### 기본 정보 확인


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6255 entries, 0 to 6254
    Data columns (total 13 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   id            6255 non-null   int64  
     1   Store         6255 non-null   int64  
     2   Date          6255 non-null   object 
     3   Temperature   6255 non-null   float64
     4   Fuel_Price    6255 non-null   float64
     5   Promotion1    2102 non-null   float64
     6   Promotion2    1592 non-null   float64
     7   Promotion3    1885 non-null   float64
     8   Promotion4    1819 non-null   float64
     9   Promotion5    2115 non-null   float64
     10  Unemployment  6255 non-null   float64
     11  IsHoliday     6255 non-null   bool   
     12  Weekly_Sales  6255 non-null   float64
    dtypes: bool(1), float64(9), int64(2), object(1)
    memory usage: 592.6+ KB
    


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180 entries, 0 to 179
    Data columns (total 12 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   id            180 non-null    int64  
     1   Store         180 non-null    int64  
     2   Date          180 non-null    object 
     3   Temperature   180 non-null    float64
     4   Fuel_Price    180 non-null    float64
     5   Promotion1    178 non-null    float64
     6   Promotion2    45 non-null     float64
     7   Promotion3    161 non-null    float64
     8   Promotion4    146 non-null    float64
     9   Promotion5    180 non-null    float64
     10  Unemployment  180 non-null    float64
     11  IsHoliday     180 non-null    bool   
    dtypes: bool(1), float64(8), int64(2), object(1)
    memory usage: 15.8+ KB
    

`Promotion`부분에 결측치가 존재하는걸 확인할 수 있음


```python
_ = train.hist(bins=50, figsize=(12, 10))
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_9_0.png)
    


- id : 샘플 아이디
- Store : 쇼핑몰 지점
- Date : 주 단위(Weekly) 날짜
- Temperature : 해당 쇼핑몰 주변 기온
- Fuel_Price : 해당 쇼핑몰 주변 연료 가격
- Promotion 1~5 : 해당 쇼핑몰의 비식별화된 프로모션 정보
- Unemployment : 해당 쇼핑몰 지역의 실업률
- IsHoliday : 해당 기간의 공휴일 포함 여부
- Weekly_Sales : 주간 매출액 (목표 예측값)

#### Store: 쇼핑몰 지점


```python
train["Store"].isnull().sum(), test["Store"].isnull().sum()
```




    (0, 0)




```python
set(train["Store"].unique()) == set(test["Store"].unique())
```




    True



쇼핑몰 지점은 `train`과 `test`가 같다라는 점을 확일할 수 있음


```python
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.countplot(x=train["Store"], ax=ax[0]).set_title("Store - Train")
sns.countplot(x=test["Store"], ax=ax[1]).set_title("Store - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_15_0.png)
    


#### Date: 주 단위 날짜
일/월/년 형식의 `object` 타입


```python
train["Date"] = pd.to_datetime(train["Date"], format="%d/%m/%Y")
test["Date"] = pd.to_datetime(test["Date"], format="%d/%m/%Y")
```


```python
plt.figure(figsize=(18, 7))
sns.lineplot(data=train, x="Date", y="Weekly_Sales", ci=None)
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_18_0.png)
    


특정 기간에 매출이 급격히 오르고/내려가는 경우가 있다는 것을 확인 가능함  
여러개의 매장이 합쳐져 있는 값이므로 극단값의 영향이 클꺼라는 생각이 듬


```python
plt.figure(figsize=(18, 7))
sns.lineplot(data=train, x="Date", y="Weekly_Sales", ci=None, hue="Store")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_20_0.png)
    


`Store`별로 나누어 찍었을때 보면, 특정 매장의 매출이 급격하게 변하는 시점이 있다는 것을 확인함


```python
train["Year"] = train["Date"].dt.year
test["Year"] = test["Date"].dt.year

train["Month"] = train["Date"].dt.month
test["Month"] = test["Date"].dt.month

train["Day"] = train["Date"].dt.day
test["Day"] = test["Date"].dt.day
```

#### Temperature: 해당 쇼핑몰 주변 기온


```python
train["Temperature"].isnull().sum(), test["Temperature"].isnull().sum()
```




    (0, 0)




```python
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.lineplot(data=train, x="Date", y="Temperature", ci=None, ax=ax[0]).set_title("Temperature - Train")
sns.lineplot(data=test, x="Date", y="Temperature", ci=None, ax=ax[1]).set_title("Temperature - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_25_0.png)
    


`train`은 12년 10월 이전의 데이터고, `test`는 12년 10월 이후의 데이터이므로 기온의 추이가 다른것을 볼 수 있음  
기온의 값이 생각보다 높아서, 화씨가 아닐까라는 생각을 함


```python
train["Cel"] = train["Temperature"].apply(lambda x: round((x-32)*5/9, 1))
test["Cel"] = test["Temperature"].apply(lambda x: round((x-32)*5/9, 1))
```


```python
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.lineplot(data=train, x="Date", y="Cel", ci=None, ax=ax[0]).set_title("Cel - Train")
sns.lineplot(data=test, x="Date", y="Cel", ci=None, ax=ax[1]).set_title("Cel - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_28_0.png)
    


기존 데이터로 주어진 온도는 화씨로 주어진 데이터임

#### Fuel_Price: 해당 쇼핑몰 주변 연료 가격


```python
train["Fuel_Price"].isnull().sum(), test["Fuel_Price"].isnull().sum()
```




    (0, 0)




```python
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
sns.lineplot(data=train, x="Date", y="Fuel_Price", ci=None, ax=ax[0]).set_title("Fuel_Price - Train")
sns.lineplot(data=test, x="Date", y="Fuel_Price", ci=None, ax=ax[1]).set_title("Fuel_Price - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_31_0.png)
    


#### Promotion


```python
promotion = ["Promotion1", "Promotion2", "Promotion3", "Promotion4", "Promotion5"]
data = [train, test]

for pro in promotion:
    print(f"{pro}: {train[pro].isnull().sum()}")
print("---------")
for pro in promotion:
    print(f"{pro}: {test[pro].isnull().sum()}")
```

    Promotion1: 4153
    Promotion2: 4663
    Promotion3: 4370
    Promotion4: 4436
    Promotion5: 4140
    ---------
    Promotion1: 2
    Promotion2: 135
    Promotion3: 19
    Promotion4: 34
    Promotion5: 0
    

`Promotion`에는 결측치들이 존재함


```python
fig, ax = plt.subplots(len(promotion), 1, figsize=(15, 22))
for col, ax in zip(promotion, ax):
    sns.lineplot(data=train, x="Date", y=col, ci=None, ax=ax).set_title(f"{col}")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_35_0.png)
    



```python
fig, ax = plt.subplots(len(promotion), 1, figsize=(15, 22))
for col, ax in zip(promotion, ax):
    sns.lineplot(data=test, x="Date", y=col, ci=None, ax=ax).set_title(f"{col}")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_36_0.png)
    


`Promotion`들은 2011년 11월 이후에 발생한 이벤트라고 생각할 수 있음 -> 2011년 11월 이전의 `Promotion`의 결측치는 0으로 처리하고, 해당하지 않는 데이터는 선형 보간법을 이용  
선형 보간을 위해 `train`과 `test`를 임시로 합치고 결측치 처리 후 다시 나눠줌


```python
temp = pd.concat([train, test])
temp.shape
```




    (6435, 17)




```python
def fill_promotion(cols):
    year, promotion = cols[0], cols[1]
    if pd.isnull(promotion):
        if year<2011: return 0
        else: return promotion
    else:
        return promotion
```


```python
for p in promotion:
    temp[p] = temp[["Year", p]].apply(fill_promotion, axis=1)
```


```python
# 선형 보간
temp[promotion] = temp[promotion].interpolate("values")
```


```python
train = temp.iloc[:train.shape[0]]
test = temp.iloc[train.shape[0]:]

train.shape, test.shape
```




    ((6255, 17), (180, 17))



#### Unemployment: 해당 쇼핑몰 지역의 실업률


```python
_ = sns.lineplot(data=train, x="Date", y="Unemployment", ci=None)
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_44_0.png)
    


실업률은 감소하는 추세임


```python
_ = train["Unemployment"].hist(bins=50)
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_46_0.png)
    



```python
train["Unemployment"].describe()
```




    count    6255.000000
    mean        8.029236
    std         1.874875
    min         4.077000
    25%         6.916500
    50%         7.906000
    75%         8.622000
    max        14.313000
    Name: Unemployment, dtype: float64




```python
def get_band_unemployment(col):
    if col<7: return "Low"
    elif col>=7 and col<=9: return "Middle"
    else: return "High"
```


```python
train["Unemployment"] = train["Unemployment"].apply(get_band_unemployment)
test["Unemployment"] = test["Unemployment"].apply(get_band_unemployment)
```

#### IsHoliday: 해당 기간의 공휴일 포함 여부


```python
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
sns.countplot(x=train["IsHoliday"], ax=ax[0]).set_title("IsHoliday - Train")
sns.countplot(x=test["IsHoliday"], ax=ax[1]).set_title("IsHoliday - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_51_0.png)
    


해당 데이터는 원-핫 인코딩을 해주면 될꺼 같음

#### Weekly_Sales: 주간 매출액 (목표 예측값)


```python
_ = sns.kdeplot(train["Weekly_Sales"], shade=True)
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_54_0.png)
    


로그 스케일링을 적용함


```python
train["Weekly_Sales_log1p"] = np.log1p(train["Weekly_Sales"])
_ = sns.kdeplot(train["Weekly_Sales_log1p"], shade=True)
```


    
![png](/assets/images/sourceImg/sale_forecast_shopping_mall_EDA_files/sale_forecast_shopping_mall_EDA_56_0.png)
    



```python
train.to_csv("data/pre_train.csv", index=False)
test.to_csv("data/pre_test.csv", index=False)
```
