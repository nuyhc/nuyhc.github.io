---
title: DACON 쇼핑몰 지점별 매출액 예측 경진대회 2 모델링 (회귀 모델)
date: 2022-07-19T14:33:03.247Z

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
## Model - Regression
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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgbm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")
```

### Data Load


```python
train = pd.read_csv("data/pre_train.csv")
test = pd.read_csv("data/pre_test.csv")

train.shape, test.shape
```




    ((6255, 18), (180, 17))



### 인코딩


```python
train = pd.get_dummies(train, columns=["Unemployment"])
test = pd.get_dummies(test, columns=["Unemployment"])
```

### 데이터셋 나누기


```python
label = "Weekly_Sales_log1p"
features = train.columns.tolist()
features.remove(label)
features.remove("id")
features.remove("Date")
features.remove("Weekly_Sales")
features.remove("Temperature")
features
```




    ['Store',
     'Fuel_Price',
     'Promotion1',
     'Promotion2',
     'Promotion3',
     'Promotion4',
     'Promotion5',
     'IsHoliday',
     'Year',
     'Month',
     'Day',
     'Cel',
     'Unemployment_High',
     'Unemployment_Low',
     'Unemployment_Middle']




```python
X_train, X_val, y_train, y_val = train_test_split(train[features], train[label], test_size=0.2)

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_val.shape}\ny_val: {y_val.shape}")
```

    X_train: (5004, 15)
    y_train: (5004,)
    X_test: (1251, 15)
    y_val: (1251,)
    

### Model
#### Random Forest


```python
reg_rf = RandomForestRegressor()

pred_rf = reg_rf.fit(X_train, y_train).predict(X_val)

mean_squared_error(y_val, pred_rf)
```




    0.013858126953444934




```python
_ = sns.barplot(x=reg_rf.feature_importances_, y=reg_rf.feature_names_in_)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/sale_forecast_shopping_mall_model_files/sale_forecast_shopping_mall_model_11_0.png?raw=true)
    


#### Catboost


```python
reg_cb = CatBoostRegressor(verbose=0)

pred_cb = reg_cb.fit(X_train, y_train).predict(X_val)

mean_squared_error(y_val, pred_cb)
```




    0.006702764929665284



#### XGBoost


```python
reg_xgb = XGBRegressor()

pred_xgb = reg_xgb.fit(X_train, y_train).predict(X_val)

mean_squared_error(y_val, pred_xgb)
```




    0.007517993250503564




```python
_ = xgb.plot_importance(reg_xgb)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/sale_forecast_shopping_mall_model_files/sale_forecast_shopping_mall_model_16_0.png?raw=true)
    


#### LGBM


```python
reg_lgbm = LGBMRegressor()

pred_lgbm = reg_lgbm.fit(X_train, y_train).predict(X_val)

mean_squared_error(y_val, pred_lgbm)
```




    0.009255447062714393




```python
_ = lgbm.plot_importance(reg_lgbm)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/sale_forecast_shopping_mall_model_files/sale_forecast_shopping_mall_model_19_0.png?raw=true)
    


#### Stacking


```python
pred_stacking = (0.25*pred_rf) + (0.25*pred_cb) + (0.25*pred_xgb) + (0.25*pred_lgbm)

np.sqrt(mean_squared_error(y_val, pred_stacking))
```




    0.08181326715540312



### Submit


```python
sub = pd.read_csv("data/sample_submission.csv")
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
      <th>id</th>
      <th>Weekly_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sub["Weekly_Sales"] = np.expm1((0.25*reg_rf.predict(test[features])) + (0.25*reg_cb.predict(test[features])) + (0.25*reg_xgb.predict(test[features])) + (0.25*reg_lgbm.predict(test[features])))
sub.to_csv("sub_stacking_rmse_08_excID.csv", index=False)
```


```python
sub["Weekly_Sales"] = np.expm1(reg_cb.predict(test[features]))
sub.to_csv("sub_cat_rmse_007_excID.csv", index=False)
```

### 마무리
DACON 제출시 22만점 정도나오는데, 4~6만점대 점수를 가진 사람들의 공통적인 특징을 보면 지점별로 예측을 했다는 점이다.  
지점별로 예측을한 과정을 이해할 수는 없지만, 나중에 시간을 내서 성능 개선을 시도해봐야겠다.  
처음으로 참가해보는 실시간 대회인데, 등수는 100등 밖으로 밀려났다.