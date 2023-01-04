---
title: DACON Wine 데이터 머신러닝으로 분류하기
date: 2022-06-21T16:18:40.230Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - Pandas
  - sklearn
  - lightGBM
  - Seaborn
  - matplot
  - Numpy
---

# DACON Wine
Kaggle wine에서는 마땅한 데이터 셋을 찾지 못해서.. DACON의 Wine 데이터 셋을 이용  
[데이터 셋 출처](https://dacon.io/competitions/open/235610/data)

### 전처리 및 EDA

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from lightgbm import plot_importance
```

### Data Load


```python
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train.shape, test.shape
```




    ((5497, 14), (1000, 13))




```python
display(train.sample(3))
display(test.sample(3))
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
      <th>index</th>
      <th>quality</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2274</th>
      <td>2274</td>
      <td>5</td>
      <td>7.5</td>
      <td>0.490</td>
      <td>0.19</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>10.0</td>
      <td>44.0</td>
      <td>0.99570</td>
      <td>3.39</td>
      <td>0.54</td>
      <td>9.7</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4452</th>
      <td>4452</td>
      <td>6</td>
      <td>7.2</td>
      <td>0.605</td>
      <td>0.02</td>
      <td>1.9</td>
      <td>0.096</td>
      <td>10.0</td>
      <td>31.0</td>
      <td>0.99500</td>
      <td>3.46</td>
      <td>0.53</td>
      <td>11.8</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>4500</td>
      <td>7</td>
      <td>6.8</td>
      <td>0.180</td>
      <td>0.30</td>
      <td>12.8</td>
      <td>0.062</td>
      <td>19.0</td>
      <td>171.0</td>
      <td>0.99808</td>
      <td>3.00</td>
      <td>0.52</td>
      <td>9.0</td>
      <td>white</td>
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
      <th>index</th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>354</th>
      <td>354</td>
      <td>8.3</td>
      <td>0.18</td>
      <td>0.30</td>
      <td>1.1</td>
      <td>0.033</td>
      <td>20.0</td>
      <td>57.0</td>
      <td>0.99109</td>
      <td>3.02</td>
      <td>0.51</td>
      <td>11.0</td>
      <td>white</td>
    </tr>
    <tr>
      <th>795</th>
      <td>795</td>
      <td>5.5</td>
      <td>0.24</td>
      <td>0.32</td>
      <td>8.7</td>
      <td>0.060</td>
      <td>19.0</td>
      <td>102.0</td>
      <td>0.99400</td>
      <td>3.27</td>
      <td>0.31</td>
      <td>10.4</td>
      <td>white</td>
    </tr>
    <tr>
      <th>197</th>
      <td>197</td>
      <td>7.6</td>
      <td>0.25</td>
      <td>0.34</td>
      <td>1.3</td>
      <td>0.056</td>
      <td>34.0</td>
      <td>176.0</td>
      <td>0.99434</td>
      <td>3.10</td>
      <td>0.51</td>
      <td>9.5</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>


### 기본 정보


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5497 entries, 0 to 5496
    Data columns (total 14 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   index                 5497 non-null   int64  
     1   quality               5497 non-null   int64  
     2   fixed acidity         5497 non-null   float64
     3   volatile acidity      5497 non-null   float64
     4   citric acid           5497 non-null   float64
     5   residual sugar        5497 non-null   float64
     6   chlorides             5497 non-null   float64
     7   free sulfur dioxide   5497 non-null   float64
     8   total sulfur dioxide  5497 non-null   float64
     9   density               5497 non-null   float64
     10  pH                    5497 non-null   float64
     11  sulphates             5497 non-null   float64
     12  alcohol               5497 non-null   float64
     13  type                  5497 non-null   object 
    dtypes: float64(11), int64(2), object(1)
    memory usage: 601.4+ KB
    


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   index                 1000 non-null   int64  
     1   fixed acidity         1000 non-null   float64
     2   volatile acidity      1000 non-null   float64
     3   citric acid           1000 non-null   float64
     4   residual sugar        1000 non-null   float64
     5   chlorides             1000 non-null   float64
     6   free sulfur dioxide   1000 non-null   float64
     7   total sulfur dioxide  1000 non-null   float64
     8   density               1000 non-null   float64
     9   pH                    1000 non-null   float64
     10  sulphates             1000 non-null   float64
     11  alcohol               1000 non-null   float64
     12  type                  1000 non-null   object 
    dtypes: float64(11), int64(1), object(1)
    memory usage: 101.7+ KB
    

- index 구분자
- quality 품질
- fixed acidity 산도
- volatile acidity 휘발성산
- citric acid 시트르산
- residual sugar 잔당 : 발효 후 와인 속에 남아있는 당분
- chlorides 염화물
- free sulfur dioxide 독립 이산화황
- total sulfur dioxide 총 이산화황
- density 밀도
- pH 수소이온농도
- sulphates 황산염
- alcohol 도수
- type 종류

`quality`가 분류해야할 타겟 값  



```python
_ = train.hist(bins=50, figsize=(12, 10))
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_10_0.png?raw=true)
    


### 결측치 확인


```python
_ = sns.heatmap(train.isnull()).set_title("Train")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_12_0.png?raw=true)
    



```python
_ = sns.heatmap(test.isnull()).set_title("Test")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_13_0.png?raw=true)
    


결측치는 존재하지 않음

### 시각화


```python
_ = sns.countplot(data=train, x="type").set_title("와인 종류별 개수")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_16_0.png?raw=true)
    



```python
_ = sns.countplot(data=train, x="quality", hue="type").set_title("와인 종류별 품질 분포")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_17_0.png?raw=true)
    



```python
_ = sns.scatterplot(data=train, x="pH", y="alcohol", hue="type")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_18_0.png?raw=true)
    


일반적으로 레드 와인의 산도가 화이트 와인에 비해 높아 보임

### 범주형 데이터 수치형 데이터로 변경하기
`type`의 경우 범주형 데이터이므로 수치형 데이터로 전환함  
`white=1`, `red=0`



```python
# train
temp = pd.get_dummies(train["type"], drop_first=True)
train = pd.concat([train, temp], axis=1).copy()
train.drop(columns="type", inplace=True)
```


```python
# test
temp = pd.get_dummies(test["type"], drop_first=True)
test = pd.concat([test, temp], axis=1).copy()
test.drop(columns="type", inplace=True)
```


```python
label_name = "quality"

features_names = train.columns.tolist()
features_names.remove("index")
features_names.remove(label_name)
```


```python
X_train, X_test, y_train, y_test = train_test_split(train[features_names], train[label_name], test_size=0.2)

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")
```

    X_train: (4397, 12)
    y_train: (4397,)
    X_test: (1100, 12)
    y_test: (1100,)
    

### Train - Decision Tress


```python
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train, y_train)

pred_dt = model_dt.predict(X_test)
```


```python
accuracy_score(pred_dt, y_test)
```




    0.61



### Decision Tree 분석


```python
model_dt.feature_importances_
```




    array([0.07201916, 0.11946309, 0.06761771, 0.08113314, 0.08731109,
           0.08870931, 0.11023723, 0.07477896, 0.07667382, 0.08779586,
           0.13392146, 0.00033917])




```python
plt.figure(figsize=(12, 8))
_ = plot_tree(model_dt, max_depth=4, feature_names=features_names, filled=True)
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_29_0.png?raw=true)
    



```python
_ = sns.barplot(x=model_dt.feature_importances_, y=model_dt.feature_names_in_)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_30_0.png?raw=true)
    


### Train - Random Forest


```python
model_rf = RandomForestClassifier()

model_rf.fit(X_train, y_train)

pred_rf = model_rf.predict(X_test)
```


```python
accuracy_score(pred_rf, y_test)
```




    0.6663636363636364



### Random Forest 분석


```python
model_rf.feature_importances_
```




    array([0.07451421, 0.09884907, 0.08039004, 0.08426116, 0.08436724,
           0.08455709, 0.09202456, 0.10306862, 0.08115849, 0.08710219,
           0.1264419 , 0.00326544])




```python
plt.figure(figsize=(12, 8))
_ = sns.barplot(x=model_rf.feature_importances_, y=model_rf.feature_names_in_)
plt.show()
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_36_0.png?raw=true)
    


### Train - LightGBM


```python
lgbm_wrapper = LGBMClassifier(n_estimators=400)
evals = [(X_test, y_test)]

lgbm_wrapper.fit(X_train, y_train, eval_metric="logloss", eval_set=evals, verbose=False)
```



    LGBMClassifier(n_estimators=400)




```python
preds = lgbm_wrapper.predict(X_test)

accuracy_score(y_test, preds)
```




    0.6436363636363637



### LightGBM 분석


```python
fig, ax = plt.subplots(figsize=(10, 12))
_ = plot_importance(lgbm_wrapper, ax=ax)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/dacon_wine_41_0.png?raw=true)
    


[DACON](https://dacon.io/competitions/open/235610/leaderboard) 제출시 `LightGBM`으로 0.671을 달성했다.  
테스트시에는, `RF` 모델의 성능이 더 좋았지만, 제출시에는 `LightGBM`이 성능이 더 좋았음  

![img](https://github.com/nuyhc/github.io.archives/blob/main/dacon_wine_files/DACON.PNG?raw=true)
