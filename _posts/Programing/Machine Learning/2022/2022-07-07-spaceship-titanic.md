---
title: Spaceship Titanic - 머신러닝으로 분류하기
date: 2022-07-06T18:37:20.814Z

categories:
  - Programming
  - Machine Learning
tags:
  - Pandas
  - Numpy
  - Machine Learning
  - Seaborn
  - matplot
  - XGBoost
  - lightGBM
  - sklearn
---

# Kaggle Spaceship Titanic

## EDA & Preprocessing
### 데이터 셋
[Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/data)

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
import random
import glob

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")
```

### Data Load


```python
path = glob.glob("data/*")
path
```




    ['data\\sample_submission.csv', 'data\\test.csv', 'data\\train.csv']




```python
train = pd.read_csv(path[2])
test = pd.read_csv(path[1])
sub = pd.read_csv(path[0])

train.shape, test.shape
```




    ((8693, 14), (4277, 13))



### EDA
#### 기본 정보


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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0013_01</td>
      <td>Earth</td>
      <td>True</td>
      <td>G/3/S</td>
      <td>TRAPPIST-1e</td>
      <td>27.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Nelly Carsoning</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0018_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/4/S</td>
      <td>TRAPPIST-1e</td>
      <td>19.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2823.0</td>
      <td>0.0</td>
      <td>Lerome Peckers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0019_01</td>
      <td>Europa</td>
      <td>True</td>
      <td>C/0/S</td>
      <td>55 Cancri e</td>
      <td>31.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Sabih Unhearfus</td>
    </tr>
  </tbody>
</table>
</div>


- `PassengerID`: `ggg-pp` 포멧, ggg는 함께 여행하는 그룹, pp는 그룹내의 번호
- `HomePlanet`: 출발 행성
- `CryoSleep`: 극저온 수면 여부, 극저온 수면 중인 경우 객실에 있음
- `Cabin`: `deck/num/side` 포멧, 객실 번호
  - `side`: P=Port, S=Starboard
- `Destination`: 도착지
- `Age`
- `VIP`
- `RoomService, FoodCourt, ShoppingMall, Spa, VRDeck`: 편의시설 이용 여부 (요금 청구 내역)
- `Name`
- `Transported`: 다른 차원으로 이송되었는지 여부 (목표 값)


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8693 entries, 0 to 8692
    Data columns (total 14 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   PassengerId   8693 non-null   object 
     1   HomePlanet    8492 non-null   object 
     2   CryoSleep     8476 non-null   object 
     3   Cabin         8494 non-null   object 
     4   Destination   8511 non-null   object 
     5   Age           8514 non-null   float64
     6   VIP           8490 non-null   object 
     7   RoomService   8512 non-null   float64
     8   FoodCourt     8510 non-null   float64
     9   ShoppingMall  8485 non-null   float64
     10  Spa           8510 non-null   float64
     11  VRDeck        8505 non-null   float64
     12  Name          8493 non-null   object 
     13  Transported   8693 non-null   bool   
    dtypes: bool(1), float64(6), object(7)
    memory usage: 891.5+ KB
    


```python
train.describe()
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
      <th>Age</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8514.000000</td>
      <td>8512.000000</td>
      <td>8510.000000</td>
      <td>8485.000000</td>
      <td>8510.000000</td>
      <td>8505.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>28.827930</td>
      <td>224.687617</td>
      <td>458.077203</td>
      <td>173.729169</td>
      <td>311.138778</td>
      <td>304.854791</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.489021</td>
      <td>666.717663</td>
      <td>1611.489240</td>
      <td>604.696458</td>
      <td>1136.705535</td>
      <td>1145.717189</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.000000</td>
      <td>47.000000</td>
      <td>76.000000</td>
      <td>27.000000</td>
      <td>59.000000</td>
      <td>46.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.000000</td>
      <td>14327.000000</td>
      <td>29813.000000</td>
      <td>23492.000000</td>
      <td>22408.000000</td>
      <td>24133.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = train.hist(bins=50, figsize=(12, 8))
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_10_0.png)
    


#### 결측치 확인


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
sns.heatmap(train.isnull(), ax=ax[0]).set_title("Train-Missing")
sns.heatmap(test.isnull(), ax=ax[1]).set_title("Test-Missing")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_12_0.png)
    



```python
temp_train = train.isnull().mean()*100
temp_test = test.isnull().mean()*100

display(temp_train)
display(temp_test)
```


    PassengerId     0.000000
    HomePlanet      2.312205
    CryoSleep       2.496261
    Cabin           2.289198
    Destination     2.093639
    Age             2.059128
    VIP             2.335212
    RoomService     2.082135
    FoodCourt       2.105142
    ShoppingMall    2.392730
    Spa             2.105142
    VRDeck          2.162660
    Name            2.300702
    Transported     0.000000
    dtype: float64



    PassengerId     0.000000
    HomePlanet      2.034136
    CryoSleep       2.174421
    Cabin           2.338087
    Destination     2.151040
    Age             2.127660
    VIP             2.174421
    RoomService     1.917232
    FoodCourt       2.478373
    ShoppingMall    2.291326
    Spa             2.361468
    VRDeck          1.870470
    Name            2.197802
    dtype: float64


대략 2% 정도의 결측치가 존대함

#### 상관 관계
결측치 대체를 상관 관계를 이용해 처리할 예정


```python
plt.figure(figsize=(12, 8))
_ = sns.heatmap(train.corr(), cmap="coolwarm", annot=True, mask=np.triu(np.ones_like(train.corr()))).set_title("Corr")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_16_0.png)
    


#### PassengerID
`ggg`의 경우 함께 여행하는 그룹을 의미하므로, 동반자 여부를 확인할 수 있을거라 생각됨  
`pp`는 개인 번호이므로 크게 의미가 없을것이라 생각



```python
train["ggg"] = train["PassengerId"].str.split("_", expand=True)[0]
test["ggg"] = test["PassengerId"].str.split("_", expand=True)[0]
```


```python
(train["ggg"].value_counts()>1).sum()
```




    1412




```python
(train["ggg"].value_counts()>1).sum() / train["ggg"].count() * 100
```




    16.242954101000805




```python
(test["ggg"].value_counts()>1).sum() / test["ggg"].count() * 100
```




    16.90437222352116



1412명, 약 16% 정도는 일행이 있음  
이를 바탕으로, 일행 수를 나타내는 파생 변수를 생성


```python
ggg_num_train, ggg_num_test = train["ggg"].value_counts().sort_index(), test["ggg"].value_counts().sort_index()

train["ggg"] = train["ggg"].apply(lambda x: ggg_num_train[x])
test["ggg"] = train["ggg"].apply(lambda x: ggg_num_test[x])
```


```python
display(train["ggg"].sample(3))
display(test["ggg"].sample(3))
```


    7004    1
    207     7
    2897    3
    Name: ggg, dtype: int64



    2134    1
    678     1
    2535    1
    Name: ggg, dtype: int64


#### HomePlanet


```python
train["HomePlanet"].isnull().sum(), test["HomePlanet"].isnull().sum()
```




    (201, 87)




```python
fig, ax =plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=train["HomePlanet"].sort_values(), ax=ax[0]).set_title("Train")
sns.countplot(x=test["HomePlanet"].sort_values(), ax=ax[1]).set_title("Test")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_27_0.png)
    



```python
_ = sns.countplot(data=train, x="HomePlanet", hue="Transported")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_28_0.png)
    


결측치를 랜덤하게 설정


```python
train["HomePlanet"].fillna(random.choice(train["HomePlanet"].unique()), inplace=True)
test["HomePlanet"].fillna(random.choice(test["HomePlanet"].unique()), inplace=True)
```


```python
train["HomePlanet"].isnull().sum(), test["HomePlanet"].isnull().sum()
```




    (0, 0)



#### Destination


```python
train["Destination"].isnull().sum(), test["Destination"].isnull().sum()
```




    (182, 92)




```python
fig, ax =plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=train["Destination"].sort_values(), ax=ax[0]).set_title("Train")
sns.countplot(x=test["Destination"].sort_values(), ax=ax[1]).set_title("Test")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_34_0.png)
    



```python
_ = sns.countplot(data=train, x="Destination", hue="Transported")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_35_0.png)
    


결측치를 랜덤하게 설정


```python
train["Destination"].fillna(random.choice(train["Destination"].unique()), inplace=True)
test["Destination"].fillna(random.choice(test["Destination"].unique()), inplace=True)
```


```python
train["Destination"].isnull().sum(), test["Destination"].isnull().sum()
```




    (0, 0)



#### CryoSleep


```python
train["CryoSleep"].isnull().sum(), test["CryoSleep"].isnull().sum()
```




    (217, 93)




```python
fig, ax =plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=train["CryoSleep"].sort_values(), ax=ax[0]).set_title("Train")
sns.countplot(x=test["CryoSleep"].sort_values(), ax=ax[1]).set_title("Test")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_41_0.png)
    


#### 극저온 수면을 취하는 사람들은 전부 사망하지 않았을까?


```python
_ = sns.countplot(data=train, x="CryoSleep", hue="Transported")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_43_0.png)
    



```python
data = train.groupby("CryoSleep")["Transported"].count().values.tolist()
labels = train.groupby("CryoSleep")["Transported"].count().index.tolist()
colors = sns.color_palette('pastel')[0:2]

_ = plt.pie(data, labels=labels, colors=colors, autopct="%.2f%%")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_44_0.png)
    


8500여 명 중, 3000(36%)명 정도가 극저온 수면을 선택함


```python
# 극저온 수면을하지 않은 그룹
data = train.groupby("CryoSleep")["Transported"].value_counts()[0].values.tolist()
labels = train.groupby("CryoSleep")["Transported"].value_counts()[0].index.tolist()
colors = sns.color_palette('pastel')[2:4]

_ = plt.pie(data, labels=labels, colors=colors, autopct="%.2f%%")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_46_0.png)
    


극저온 수면을하지 않은 사람들의 약 67%는 생존했음


```python
# 극저운 수면을한 그룹
data = train.groupby("CryoSleep")["Transported"].value_counts()[1].values.tolist()
labels = train.groupby("CryoSleep")["Transported"].value_counts()[1].index.tolist()
colors = sns.color_palette('pastel')[4:6]

_ = plt.pie(data, labels=labels, colors=colors, autopct="%.2f%%")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_48_0.png)
    


극저온 수면을 한 사람들의 약 82%는 사망했음


```python
_ = train.groupby("CryoSleep")["Transported"].value_counts().plot.bar(rot=0)
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_50_0.png)
    


극저온 수면을 선택한 사람들의 사망 비율이 높다는 것을 확인 할 수 있음  
`train`의 `CryoSleep`의 결측치는 해당 정보를 이용해 보완해 줄 수 있을 것 같음

#### 극저온 수면을 선택한 사람들의 출발/도착 행성은?


```python
_ = sns.countplot(data=train, x="CryoSleep", hue="HomePlanet").set_title("CryoSleep - HomePlanet")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_53_0.png)
    



```python
_ = sns.countplot(data=train, x="CryoSleep", hue="Destination").set_title("CryoSleep - Destination")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_54_0.png)
    


출발지/도착지와 극저온 수면 선택 여부는 크게 상관이 없어보임  
단순히, 지구 출발자와 TRAPPIST-1e가 도착지인 사람들이 많이 선택했음



```python
train.groupby("Destination")["CryoSleep"].value_counts(normalize=True)*100
```




    Destination    CryoSleep
    55 Cancri e    False        57.915718
                   True         42.084282
    PSO J318.5-22  False        50.707851
                   True         49.292149
    TRAPPIST-1e    False        67.777217
                   True         32.222783
    Name: CryoSleep, dtype: float64



위 확률을 이용해 랜덤하게 부여할 예정

#### 거리와 상관있지 않을까?
출발지는 `Earth, Mars, Europa`,  
도착지는 `TRAPPIST-1e, PSO J318.5-22, 55 Cancri e`

- `Mars`는 지구로부터 약 0.000042 광년 떨어져 있음  
- `Europa`는 지구로부터 약 0.000066 광년 떨어져 있음  
- `TRAPPIST-1e`는 지구로부터 약 39.6 광년 떨어져 있음  
- `PSO J318.5-22`는 지구로부터 약 80광년 떨어져 있음  
- `55 Cancri e`는 지구로부터 약 40.9광년 떨어져 있음  

사실상 목저지만 이용해도 될 것 같음


위, `Destination` 파트에서 봤듯이 `Transported`와 크게 상관이 없는 것으로 보임  

`train`의 `CryoSleep`은 `Transported`를 바탕으로 결측치를 채우고,  
`test`의 `CryoSleep`은 `Destination`을 바탕으로 채워줌


```python
# train
def base_Transported(cols):
    Cryosleep, Transported = cols[0], cols[1]
    # Cryosleep이 결측치면,
    if pd.isnull(Cryosleep):
        if Transported: return True # 사망했다면 True
        else: return False # 사망하지 않았다면 False
    else: return Cryosleep # 결측치가 아닌 경우 그대로 반환
```


```python
train["CryoSleep"] = train[["CryoSleep", "Transported"]].apply(base_Transported, axis=1)
```


```python
# test
def base_Destination_proba(cols):
    Cryosleep, Destination = cols[0], cols[1]
    if pd.isnull(Cryosleep):
        if Destination=="TRAPPIST-1e":
            return random.choices([True, False], weights=[0.33, 0.67])[-1]
        elif Destination=="55 Cancri e":
            return random.choices([True, False], weights=[0.41, 0.59])[-1]
        else:
            return random.choices([True, False], weights=[0.49, 0.51])[-1]
    else: return Cryosleep
```


```python
test["CryoSleep"] = test[["CryoSleep", "Destination"]].apply(base_Destination_proba, axis=1)
```


```python
train["CryoSleep"].isnull().sum(), test["CryoSleep"].isnull().sum()
```




    (0, 0)



#### Cabin
`deck/num/side` 형식을 가지고 있음  


```python
train["Cabin"].isnull().sum(), test["Cabin"].isnull().sum()
```




    (199, 100)




```python
train[train["Cabin"].duplicated()].iloc[:5]
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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>ggg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0008_02</td>
      <td>Europa</td>
      <td>True</td>
      <td>B/1/P</td>
      <td>TRAPPIST-1e</td>
      <td>34.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Altardr Flatic</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0008_03</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/1/P</td>
      <td>55 Cancri e</td>
      <td>45.0</td>
      <td>False</td>
      <td>39.0</td>
      <td>7295.0</td>
      <td>589.0</td>
      <td>110.0</td>
      <td>124.0</td>
      <td>Wezena Flatic</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0020_02</td>
      <td>Earth</td>
      <td>True</td>
      <td>E/0/S</td>
      <td>55 Cancri e</td>
      <td>49.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Glendy Brantuarez</td>
      <td>False</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0020_03</td>
      <td>Earth</td>
      <td>True</td>
      <td>E/0/S</td>
      <td>55 Cancri e</td>
      <td>29.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Mollen Mcfaddennon</td>
      <td>False</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



같은 객실을 사용한 사람들은 같은 그룹에 속해있는 사람임  
`deck`과 `side` 정보만 파생변수로 만들어 사용함  
결측치 처리를 위해 결측치가 없는 데이터만 일부 먼저 사용



```python
temp = train[~train["Cabin"].isnull()]
temp.shape
```




    (8494, 15)




```python
temp["Deck"] = temp["Cabin"].apply(lambda x: x.split("/")[0])
temp["Side"] = temp["Cabin"].apply(lambda x: x.split("/")[-1])
```


```python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.countplot(x=temp["Deck"], ax=ax[0]).set_title("Deck")
sns.countplot(x=temp["Side"], ax=ax[1]).set_title("Side")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_71_0.png)
    


`Deck`과 달리, `Side`는 거의 균일하게 분포되어 있음  
규모가 다를까라는 의문이 생김


```python
_ = sns.countplot(data=temp, x="Deck", hue="Side")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_73_0.png)
    


`Side`에 따라 `Deck`도 비교적 균일하게 분포되어있음 -> 규모도 거의 동일


```python
temp.groupby("Transported")["Deck"].value_counts(normalize=True)*100
```




    Transported  Deck
    False        F       37.120493
                 G       29.364326
                 E       13.353890
                 D        6.427894
                 C        5.668880
                 B        4.909867
                 A        3.059772
                 T        0.094877
    True         G       30.878915
                 F       28.728378
                 B       13.370734
                 C       11.874708
                 E        7.316503
                 D        4.838710
                 A        2.968677
                 T        0.023375
    Name: Deck, dtype: float64




```python
temp.groupby("Transported")["Side"].value_counts(normalize=True)*100
```




    Transported  Side
    False        P       54.743833
                 S       45.256167
    True         S       55.633474
                 P       44.366526
    Name: Side, dtype: float64




```python
# train
def base_Transported_proba(cols):
    Cabin, Transported = cols[0], cols[1]
    if pd.isnull(Cabin):
        if Transported:
            Deck = random.choices(["G", "F", "B", "C", "E", "D", "A", "T"], weights=[0.31, 0.29, 0.13, 0.12, 0.07, 0.05, 0.03, 0.002])[0]
            Side = random.choices(["P", "S"], weights=[0.56, 0.44])[0]
            return f"{Deck}/{Side}"
        else:
            Deck = random.choices(["F", "G", "E", "D", "C", "B", "A", "T"], weights=[0.37, 0.29, 0.13, 0.06, 0.05, 0.04, 0.03, 0.001])[0]
            Side = random.choices(["P", "S"], weights=[0.55, 0.45])[0]
            return f"{Deck}/{Side}"
    else: return Cabin
```


```python
train["Cabin"] = train[["Cabin", "Transported"]].apply(base_Transported_proba, axis=1)
```

`train`의 `Cabin`은 위 2개의 표를 이용해 랜덤하게 채워줌  


```python
temp["Deck"].value_counts(normalize=True)*100
```




    F    32.893807
    G    30.127149
    E    10.313162
    B     9.171180
    C     8.794443
    D     5.627502
    A     3.013892
    T     0.058865
    Name: Deck, dtype: float64




```python
temp["Side"].value_counts(normalize=True)*100
```




    S    50.482694
    P    49.517306
    Name: Side, dtype: float64




```python
# test
test["Cabin"].fillna(str(random.choices(["F", "G", "E", "B", "C", "D", "A", "T"], weights=[0.33, 0.30, 0.1, 0.09, 0.09, 0.06, 0.03, 0.0006])[0]) \
    + "/" + str(random.choices(["S", "P"], weights=[0.5, 0.5])[0]), inplace=True)
```


```python
train["Cabin"].isnull().sum(), test["Cabin"].isnull().sum()
```




    (0, 0)



`test`의 `Cabin`은 위 2개의 표를 이용해 랜덤하게 채워줌


```python
train["Deck"] = train["Cabin"].apply(lambda x: x.split("/")[0])
train["Side"] = train["Cabin"].apply(lambda x: x.split("/")[-1])

test["Deck"] = test["Cabin"].apply(lambda x: x.split("/")[0])
test["Side"] = test["Cabin"].apply(lambda x: x.split("/")[-1])
```

#### Age


```python
train["Age"].isnull().sum(), test["Age"].isnull().sum()
```




    (179, 91)




```python
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.kdeplot(data=train, x="Age", hue="VIP", shade=True, ax=ax[0]).set_title("Age - Train")
sns.kdeplot(data=test, x="Age", hue="VIP", shade=True, ax=ax[1]).set_title("Age - Test")
plt.show()
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_88_0.png)
    



```python
train.groupby("VIP")["Age"].describe()
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
      <th>VIP</th>
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
      <th>False</th>
      <td>8119.0</td>
      <td>28.639611</td>
      <td>14.469895</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>True</th>
      <td>198.0</td>
      <td>37.449495</td>
      <td>11.611957</td>
      <td>18.0</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>73.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.groupby("VIP")["Age"].describe()
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
      <th>VIP</th>
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
      <th>False</th>
      <td>4023.0</td>
      <td>28.487447</td>
      <td>14.238827</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>26.0</td>
      <td>37.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>True</th>
      <td>73.0</td>
      <td>34.534247</td>
      <td>9.771106</td>
      <td>18.0</td>
      <td>27.0</td>
      <td>32.0</td>
      <td>39.0</td>
      <td>67.0</td>
    </tr>
  </tbody>
</table>
</div>



`VIP` 여부를 이용해 결측치를 채워줌


```python
def age_by_VIP(cols):
    age, vip = cols[0], cols[1]
    if pd.isna(age):
        if vip: return 35
        else: return 28
    else: return age
```


```python
train["Age"] = train[["Age", "VIP"]].apply(age_by_VIP, axis=1)
test["Age"] = test[["Age", "VIP"]].apply(age_by_VIP, axis=1)
```


```python
train["Age"].isnull().sum(), test["Age"].isnull().sum()
```




    (0, 0)



`band_age` 컬럼을 생성


```python
train["Band_Age"] = train["Age"].apply(lambda x: int(str(x)[0]+"0"))
test["Band_Age"] = test["Age"].apply(lambda x: int(str(x)[0]+"0"))
```


```python
_ = sns.countplot(data=train, x="Band_Age", hue="Transported").set_title("Transported - Age Band")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_97_0.png)
    


20대 미만 -> `child`  
20~40대 -> `adult`  
40~60대 -> `middle`  
60대~ -> `old`


```python
def age_str(col):
    if col in [0, 10]: return "child"
    elif col in [20, 40]: return "adult"
    elif col in [50, 60]: return "middle"
    else: return "old"
```


```python
train["Band_Age"] = train["Band_Age"].apply(age_str)
test["Band_Age"] = test["Band_Age"].apply(age_str)
```

#### VIP


```python
train["VIP"].isnull().sum(), test["VIP"].isnull().sum()
```




    (203, 93)



`adult, middle`이면 `True`, 아니면 `False`로 결측치를 대체함


```python
def base_band_VIP(cols):
    band, vip = cols[0], cols[1]
    
    if pd.isna(vip):
        if band in ["adult", "middle"]: return True
        else: return False
    else: return vip
```


```python
train["VIP"] = train[["Band_Age", "VIP"]].apply(base_band_VIP, axis=1)
test["VIP"] = test[["Band_Age", "VIP"]].apply(base_band_VIP, axis=1)
```


```python
train["VIP"].isnull().sum(), test["VIP"].isnull().sum()
```




    (0, 0)



#### 실수형 변수


```python
nums = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
```


```python
train[nums].isnull().sum()
```




    RoomService     181
    FoodCourt       183
    ShoppingMall    208
    Spa             183
    VRDeck          188
    dtype: int64




```python
test[nums].isnull().sum()
```




    RoomService      82
    FoodCourt       106
    ShoppingMall     98
    Spa             101
    VRDeck           80
    dtype: int64




```python
_ = train[nums].hist(bins=30, figsize=(14, 7))
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_EDA_files/spaceship_titanic_EDA_111_0.png)
    


이용 여부로 변경해도 좋을꺼 같다는 생각을 함


```python
train[nums] = train[nums].fillna(0)
test[nums] = test[nums].fillna(0)
```


```python
train["Service"] = train[nums].sum(axis=1)
test["Service"] = test[nums].sum(axis=1)
```


```python
train["Service"] = train["Service"].apply(lambda x: False if x==0 else True)
test["Service"] = test["Service"].apply(lambda x: False if x==0 else True)
```


```python
train.head()
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
      <th>PassengerId</th>
      <th>HomePlanet</th>
      <th>CryoSleep</th>
      <th>Cabin</th>
      <th>Destination</th>
      <th>Age</th>
      <th>VIP</th>
      <th>RoomService</th>
      <th>FoodCourt</th>
      <th>ShoppingMall</th>
      <th>Spa</th>
      <th>VRDeck</th>
      <th>Name</th>
      <th>Transported</th>
      <th>ggg</th>
      <th>Deck</th>
      <th>Side</th>
      <th>Band_Age</th>
      <th>Service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0001_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>B/0/P</td>
      <td>TRAPPIST-1e</td>
      <td>39.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Maham Ofracculy</td>
      <td>False</td>
      <td>1</td>
      <td>B</td>
      <td>P</td>
      <td>old</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>24.0</td>
      <td>False</td>
      <td>109.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>549.0</td>
      <td>44.0</td>
      <td>Juanna Vines</td>
      <td>True</td>
      <td>1</td>
      <td>F</td>
      <td>S</td>
      <td>adult</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0003_01</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>58.0</td>
      <td>True</td>
      <td>43.0</td>
      <td>3576.0</td>
      <td>0.0</td>
      <td>6715.0</td>
      <td>49.0</td>
      <td>Altark Susent</td>
      <td>False</td>
      <td>2</td>
      <td>A</td>
      <td>S</td>
      <td>middle</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0003_02</td>
      <td>Europa</td>
      <td>False</td>
      <td>A/0/S</td>
      <td>TRAPPIST-1e</td>
      <td>33.0</td>
      <td>False</td>
      <td>0.0</td>
      <td>1283.0</td>
      <td>371.0</td>
      <td>3329.0</td>
      <td>193.0</td>
      <td>Solam Susent</td>
      <td>False</td>
      <td>2</td>
      <td>A</td>
      <td>S</td>
      <td>old</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0004_01</td>
      <td>Earth</td>
      <td>False</td>
      <td>F/1/S</td>
      <td>TRAPPIST-1e</td>
      <td>16.0</td>
      <td>False</td>
      <td>303.0</td>
      <td>70.0</td>
      <td>151.0</td>
      <td>565.0</td>
      <td>2.0</td>
      <td>Willy Santantines</td>
      <td>True</td>
      <td>1</td>
      <td>F</td>
      <td>S</td>
      <td>child</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### 불필요한 열 삭제


```python
drop_list = ["PassengerId", "Cabin", "Age", "Name"]
drop_list.extend(nums)
drop_list
```




    ['PassengerId',
     'Cabin',
     'Age',
     'Name',
     'RoomService',
     'FoodCourt',
     'ShoppingMall',
     'Spa',
     'VRDeck']




```python
train.drop(columns=drop_list, axis=1, inplace=True)
test.drop(columns=drop_list, axis=1, inplace=True)
```


```python
train.to_csv("data/pre_train.csv", index=False)
test.to_csv("data/pre_test.csv", index=False)
```

## Model - Classifier

### Data Load


```python
path = glob.glob("data/*")
path
```




    ['data\\pre_test.csv',
     'data\\pre_train.csv',
     'data\\sample_submission.csv',
     'data\\test.csv',
     'data\\train.csv']




```python
train = pd.read_csv(path[1])
test = pd.read_csv(path[0])
sub = pd.read_csv(path[2])

train.shape, test.shape, sub.shape
```




    ((8693, 10), (4277, 9), (4277, 2))



### 인코딩


```python
display(train.dtypes)
display(test.dtypes)
```


    HomePlanet     object
    CryoSleep        bool
    Destination    object
    VIP              bool
    Transported      bool
    ggg             int64
    Deck           object
    Side           object
    Band_Age       object
    Service          bool
    dtype: object



    HomePlanet     object
    CryoSleep        bool
    Destination    object
    VIP              bool
    ggg             int64
    Deck           object
    Side           object
    Band_Age       object
    Service          bool
    dtype: object



```python
encode_list = train.columns.tolist()
encode_list.remove("Transported")
encode_list.remove("ggg")
encode_list.remove("CryoSleep")
encode_list.remove("VIP")
encode_list.remove("Service")
encode_list
```




    ['HomePlanet', 'Destination', 'Deck', 'Side', 'Band_Age']




```python
temp_train, temp_test = [], []

for col in encode_list:
    temp_train.append(pd.get_dummies(train[col], drop_first=True))
    temp_test.append(pd.get_dummies(test[col], drop_first=True))
    
temp_train = pd.concat(temp_train, axis=1)
temp_test = pd.concat(temp_test, axis=1)
```


```python
train = pd.concat([train.drop(columns=encode_list, axis=1), temp_train], axis=1)
test = pd.concat([test.drop(columns=encode_list, axis=1), temp_test], axis=1)
```


```python
train.shape, test.shape
```




    ((8693, 20), (4277, 19))



### Train


```python
label = "Transported"
features = test.columns.tolist()
```


```python
X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.2, stratify=train[label])

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")
```

    X_train: (6954, 19)
    y_train: (6954,)
    X_test: (1739, 19)
    y_test: (1739,)
    

#### Random Forest


```python
clf_rf = RandomForestClassifier()

pred_rf = clf_rf.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, pred_rf)
```




    0.7199539965497412




```python
_ = sns.barplot(x=clf_rf.feature_importances_, y=features).set_title("RF")
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_model_files/spaceship_titanic_model_16_0.png)
    


#### XGBoost


```python
clf_xgb = XGBClassifier()

pred_xgb = clf_xgb.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, pred_rf)
```

    0.7199539965497412




```python
_ = xgb.plot_importance(clf_xgb)
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_model_files/spaceship_titanic_model_19_0.png)
    


#### LGBM


```python
clf_lgbm = LGBMClassifier()

pred_lgbm = clf_lgbm.fit(X_train, y_train).predict(X_test)

accuracy_score(y_test, pred_lgbm)
```




    0.7527314548591144




```python
_ = lgbm.plot_importance(clf_lgbm)
```


    
![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_model_files/spaceship_titanic_model_22_0.png)
    

## 마무리
EDA와 전처리를 굉장히 만족스럽게했지만, 기대했던 성능에 미치지는 못했다..  
베스트 스코어는 역시 `lgbm`을 사용한 경우로, 74점정도임

![png](/assets/images/sourceImg/spaceship_titanic/spaceship_titanic_kaggle_score.png)