---
title: Stacking Ensemble 모델
date: 2022-07-04T14:33:28.392Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - Pandas
  - Numpy
---

[참고 | 파이썬 머신러닝 완벽 가이드]()  

# Stacking Ensemble
스태킹(Stacking)은 개별적인 여러 알고리즘을 서로 결합해 예측 결과를 도출한다는 점에서 배깅(Bagging) 및 부스팅(Boosting)과 공통점을 가지지만,  
개별 알고리즘으로 예측한 데이터를 기반으로 다시 예측을 수행한다는 차이가 있음  
-> 개별 알고리즘의 예측 결과 데이터 세트를 최종적인 메타 데이터 세트로 만들어 별도의 ML 알고리즘으로 최종 학습을 수행하고 테스트 데이터를 기반으로 다시 최종 예측을 수행

### 스태킹 모델은 두 종류의 모델이 필요  
1. 개별적인 **기반 모델**
2. 개별 기반 모델의 예측 데이터를 학습 데이터로 만들어 학습하는 최종 **메타 모델**

여러 개별 모델의 예측 데이터를 각각 스태킹 형태로 결합해, 최종 메타 모델의 학습용과 테스트용 피처 데이터 세트를 만들어야 함  

일반 상황에서는 잘 사용하지 않고, 조금이라도 성능을 올려야하는 경우 사용  
일반적으로 성능이 비슷한 모델을 결합해 좀 더 나은 성능 향상을 도출하기 위해 적용


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


```python
cancer_data = load_breast_cancer()

X_data = cancer_data.data
y_label = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.2)
```

### 개별 모델 생성


```python
clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_rf = RandomForestClassifier(n_estimators=100)
clf_dt = DecisionTreeClassifier()
clf_ada = AdaBoostClassifier(n_estimators=100)
```

### 메타 모델 생성


```python
clf_lr_final = LogisticRegression(C=10)
```

### 개별 모델 학습 및 예측


```python
clf_knn.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)
clf_dt.fit(X_train, y_train)
clf_ada.fit(X_train, y_train)
```




    AdaBoostClassifier(n_estimators=100)




```python
pred_knn = clf_knn.predict(X_test)
pred_rf = clf_rf.predict(X_test)
pred_dt = clf_dt.predict(X_test)
pred_ada = clf_ada.predict(X_test)

print(f"knn acc: {accuracy_score(y_test, pred_knn):.4f}\nrf acc: {accuracy_score(y_test, pred_rf):.4f}\ndt acc: {accuracy_score(y_test, pred_dt):.4f}\nada acc: {accuracy_score(y_test, pred_ada):.4f}")
```

    knn acc: 0.9649
    rf acc: 0.9737
    dt acc: 0.9386
    ada acc: 0.9912
    

### 메타 모델 학습 및 예측
개별 알고리즘(개별 모델)으로부터 예측된 예측값을 컬럼 레벨로 옆으로 붙여 피처 값으로 만들어, 최종 메타 모델의 학습 데이터로 사용  
반환된 예측 데이터 세트는 1차원 `ndarray`이므로, 예측 결과를 행 형태로 붙인 뒤, 행과 열을 바꿈


```python
X_test.shape
```




    (114, 30)




```python
pred = np.array([pred_knn, pred_rf, pred_dt, pred_ada])
pred.shape
```




    (4, 114)




```python
pred = np.transpose(pred)
pred.shape
```




    (114, 4)




```python
clf_lr_final.fit(pred, y_test)
final = clf_lr_final.predict(pred)

accuracy_score(y_test, final)
```




    0.9912280701754386



## CV 세트 기반의 스태킹
오버피팅 문제를 개선  
1. 각 개별 모델별 원본 학습/테스트 데이터를 예측한 결과 값을 기반으로 메타 모델을 위한 학습/테스트용 데이터 생성
2. 생성한 모든 데이터(학습/테스트)를 스태킹 해, 메타 데이터를 학습 및 테스트


```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
```


```python
def get_stacking_base_datasets(model, X_train_n, y_train_n, X_test_n, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=False)
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    print(model.__class__.__name__, "model start")
    
    for folder_counter, (train_idx, valid_idx) in enumerate(kf.split(X_train_n)):
        X_tr = X_train_n[train_idx]
        y_tr = y_train_n[train_idx]
        X_te = X_train_n[valid_idx]
        
        model.fit(X_tr, y_tr)
        train_fold_pred[valid_idx, :] = model.predict(X_te).reshape(-1, 1)
        test_pred[:, folder_counter] = model.predict(X_test_n)
    
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    
    return train_fold_pred, test_pred_mean
```


```python
knn_train, knn_test = get_stacking_base_datasets(clf_knn, X_train, y_train, X_test, 7)
rf_train, rf_test = get_stacking_base_datasets(clf_rf, X_train, y_train, X_test, 7)
dt_train, dt_test = get_stacking_base_datasets(clf_dt, X_train, y_train, X_test, 7)
ada_train, ada_test = get_stacking_base_datasets(clf_ada, X_train, y_train, X_test, 7)
```

    KNeighborsClassifier model start
    RandomForestClassifier model start
    DecisionTreeClassifier model start
    AdaBoostClassifier model start
    


```python
Stack_final_X_train = np.concatenate((knn_train, rf_train, dt_train, ada_train), axis=1)
Stack_final_X_test = np.concatenate((knn_test, rf_test, dt_test, ada_test), axis=1)

Stack_final_X_train.shape, Stack_final_X_test.shape
```




    ((455, 4), (114, 4))




```python
clf_lr_final.fit(Stack_final_X_train, y_train)
stack_final = clf_lr_final.predict(Stack_final_X_test)

accuracy_score(y_test, stack_final)
```




    0.9912280701754386



스태킹을 이루는 모델은 최적으로 파라미터를 튜닝한 상태에서 스태킹 모델을 생성하는 것이 일반적임  

