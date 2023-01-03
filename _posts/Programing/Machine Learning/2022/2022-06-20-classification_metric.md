---
title: 분류 모델에서의 평가 지표
date: 2022-06-19T15:30:11.223Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - Pandas
  - sklearn
  - Tutorial
---

# 분류 모델에서의 평가
## 분류의 평가 지표
- 정확도 (Accuracy)
- 오차행렬 (Confusion Matrix)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1 score
- ROC AUC

### 간단한 분류 모델
결정 트리를 이용한 모델


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```


```python
df_pima = pd.read_csv("http://bit.ly/data-diabetes-csv")

label_feature = "Outcome"
feature_name = df_pima.columns.tolist()
feature_name.remove(label_feature)

X_train, X_test, y_train, y_test = train_test_split(df_pima[feature_name], df_pima[label_feature], test_size=0.2, shuffle=False, random_state=42)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### 정확도 (Accuracy)
실제 데이터에서 예측 데이터가 얼마나 같은지를 판단하는 지표  
$ 정확도 = {예측 결과가 동일한 데이터 건수} \over {전체 예측 데이터 건수} $  
직관적으로 모델의 성능을 평가 할 수 있지만, 성능을 왜곡할 가능성도 있음  
    - 데이터가 지나치게 잘 구성되어 있는 경우
    - 데이터가 불균형한 경우



```python
from sklearn.metrics import accuracy_score

accuracy_score(y_pred, y_test)
```




    0.7142857142857143



### 오차행렬 (Confusion Matrix)
이진 분류에서 성능 지표로 사용 됨  
예측을 수행하면서 얼마나 헷갈리고 있는지도 함께 보여줌  
- `True/False` `Positive/Negative`를 이용한 방식


```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred, y_test)
```




    array([[76, 21],
           [23, 34]], dtype=int64)



| TN | FP |
| --- | --- |
| FN | TP |

형태로, `TN=76, FP=21, FN=23, TP=34`임  
- TN (True-Neagtive): 예측 값은 0, 실제 값은 0
- FP (False-Positive): 예측 값은 1, 실제 값은 0
- FN (False-Neagative): 예측 값은 0, 실제 값은 1
- TP (True-Positive): 예측 값은 1, 실제 값은 1

간단하게 생각하면, 앞에 `T`가 붙은 경우에만 예측 값이 실제 값과 동일하다는 의미  
위 오차 행렬을 해석하면, 154개의 예측 중 `TN+TP = 76+34 = 110`개만 맞췄다는 것을 의미


```python
(110/154) * 100
```




    71.42857142857143



위와 같이 정확도도 구할 수 있음

### 정밀도와 재현율 (Precision and Recall)
Positive 데이터 세트의 예측 성능에 좀 더 초첨을 맞춘 평가 지표  
#### 정밀도 (Precision)
양성 예측도라고도 불림  
실제 음성인 값을 양성으로 판단하면 안되는 경우 중요 (스팸 메일 분류 등)  
$ Precision = {TP \over (FP + TP)} $

#### 재현율 (Recall)
민감도(sensitivity) 또는 TRP(True Positive Rate)라고도 불림  
실제 양성인 값을 음성으로 판단하면 안되는 경우 중요 (암 판단 모델 등)  
$ Recall = {TP \over (FN + TP)} $  

민감도와 대응하는 지표로 TNR(True Negative Rate)라 불리는 특이성(specificity)가 있음  
$ TNR = {TN \over (FP + TN)} $


```python
from sklearn.metrics import precision_score, recall_score

print(f"Precision: {precision_score(y_pred, y_test)}\nRecall: {recall_score(y_pred, y_test)}")
```

    Precision: 0.6181818181818182
    Recall: 0.5964912280701754
    


```python
precision = 34 / (21+34)
recall = 34 / (23+34)

print(f"Precision: {precision}\nRecall: {recall}")
```

    Precision: 0.6181818181818182
    Recall: 0.5964912280701754
    

정밀도와 재현율은 상호 보완적 관계로, 한가지가 증가하면 다른 한가지는 감소함 -> **정밀도/재현율 트레이드오프(trade-off)**

### F1 score
정밀도와 재현율을 결합한 지표로, 정밀도와 재현율이 어느 한쪽으로 치우치지 않는 수치를 나타낼 때 상대적으로 높은 값을 가짐  
$ F1 = {2 \over {1 \over Recall} + {1 \over Precision}} = {2 \times {Precision \times Recall \over {Precision + Recall}}} $


```python
from sklearn.metrics import f1_score

f1_score(y_pred, y_test)
```




    0.607142857142857




```python
2 / ((1/recall) + (1/precision))
```




    0.6071428571428572



### ROC 곡선과 AUC
#### ROC (Receiver Operation Characteristic Curve)
수신자 판단 곡선으로, FPR이 변할 때 TPR이 어떻게 변하는지 나타낸 곡선 (`x=FPR, y=TPR`)  
임계값(threshold) 개념이 적용ㄷ함


```python
from sklearn.metrics import roc_curve

fpr, tpr, thr = roc_curve(y_pred, y_test)
```


```python
fpr
```




    array([0.        , 0.21649485, 1.        ])




```python
tpr
```




    array([0.        , 0.59649123, 1.        ])




```python
thr
```




    array([2, 1, 0], dtype=int64)



#### AUC (Area Under Curve)
ROC 곡선 밑의 면적을 구한 것으로, 일반적으로 1에 가까울수록 좋음


```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_pred, y_test)
```




    0.6899981913546753


