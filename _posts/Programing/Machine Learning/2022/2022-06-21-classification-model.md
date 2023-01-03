---
title: 머신러닝 주요 분류 모델 정리
date: 2022-06-20T16:23:11.519Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - Pandas
  - sklearn
  - XGBoost
  - lightGBM
  - Seaborn
  - matplot
  - Numpy
  - Tutorial
---

머신러닝 주요 분류 모델들을 예제와 함께 정리해봤습니다.

# 분류 모델 (Classification)
분류는 지도 학습의 대표적인 유형으로, 데이터의 피처와 레이블값을 학습해 모델을 생성하고,  
생성된 모델에 새로운 데이터를 넣어 미지의 레이블 값을 예측하는 것

## 머신러닝의 분류 알고리즘
- 나이브 베이즈(Naive Bayes)
- 로지스틱 회귀(Logistic Regression)
- 결정 트리(Decision Tree)
- 서포트 벡터 머신(SVM, Support Vector Machine)
- 최소 근접 알고리즘(Nearest Neighbor)
- 신경망(NN, Neural Network)
- 앙상블(Ensemble)

다양한 분류 알고리즘이 있지만, 주요한 몇개의 알고리즘만 다뤄 봅니다.  
정형 데이터의 경우, 앙상블 알고리즘이 매우 높은 성능을 보여 선호되는 추세입니다.  

## 앙상블 (Ensemble)
배깅(Bagging)과 부스팅(Boosting) 방식으로 분류 됨
- 배깅
  - 랜덤 포레스트(Random Forest)
- 부스팅
  - 그래디언트 부스팅(Gradient Boosting)
  - XgBoost
  - LightGBM

앙상블은 기본적으로 서로 다르거나 같은 알고리즘을 단순하게 결합한 방식이지만, 이런 앙상블을 앙상블끼리 결합한 **스태킹(Stacking)**도 있음

## 결정 트리 (Decision Tree)
머신 러닝 알고리즘 직관적으로 이해하기 쉬운 알고리즘으로, `if-else`를 이용한 방식이라고 생각하면 됨  
데이터의 어떤 기준을 바탕으로 규칙을 만드는지에 따라 성능이 달라지고, 특성(feature)이 많으면 모델이 복잡해져 Overfitting의 가능성이 높아짐  

### 지니 불순도와 엔트로피 (Gini and Entropy)
- 지니 불순도
  - 0에 가까워질수록 평등 (데이터가 잘 분류 됨)
  - 1에 가까워질수록 불평등 (데이터가 잘 분류 되지 못함 -> 불순도가 높다)
- 엔트로피
  - 혼잡도를 나타내는 개념으로, 데이터가 섞여있으면 엔트로피가 높다

결정 트리가 분기를하는 방식은 `지니`와 `엔트로피`를 이용함 (모델 생성시 선택해줘야 함)  

### 장점과 단점
- 장점
  - 쉽고 직관적
  - 데이터의 가공도가 영향을 미치지 않음
- 단점
  - 오버피팅 가능성이 높음

### 주요 파라미터
- `min_samples_split`:
  - 분할하기 위한 최소한의 샘플 수 (오버피팅을 제어하는데 사용)
  - `default=2`, 작게 설정할 수록 가지 수가 많아져 오버피팅 가능성이 높아짐
- `min_samples_leaf`:
  - 말단 노드가 되기 위한 최소한의 샘플 데이터 수
  - 오버피팅 제어하는데 사용
- `max_features`:
  - 최적의 분할을 위해 고려하는 최대 특성 개수
  - `default=None`으로 모든 특성을 고려함
  - `int`형 지정시 개수, `float`형 지정시 비율
  - `sqrt, auto, log` 옵션이 존재
- `max_depth`:
  - 트리의 최대 깊이를 규정
  - `default=None`으로 완벽하게 클래스가 결정될 때까지 분기
- `max_leaf_nodes`: 말단 노드의 최대 개수


```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

clf_dt = DecisionTreeClassifier(random_state=42)

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

clf_dt.fit(X_train, y_train)
```




    DecisionTreeClassifier(random_state=42)




```python
plt.figure(figsize=(12, 8))
_ = plot_tree(clf_dt, filled=True, feature_names=iris_data.feature_names)
plt.show()
```


    
![png](/assets/images/sourceImg/classification_files/classification_2_0.png)
    



```python
_ = sns.barplot(x=clf_dt.feature_importances_, y=iris_data.feature_names)
```


    
![png](/assets/images/sourceImg/classification_files/classification_3_0.png)
    


결정트리는 화이트 박스 모델이라고도하며, 내부에서 어떤 방식으로 데이터를 분할하는지 쉽게 알 수 있음

### 과적합 (Overfitting)
2개의 특성과 3개의 클래스를 갖는 더미 데이터를 이용해 시각화  
3개의 클래스는 서로 다른 색으로 시각화되었음




```python
from sklearn.datasets import make_classification

plt.title("Features-2-Class-3")
X_features, y_labels = make_classification(n_features=2, n_redundant=0, n_informative=2, n_classes=3, n_clusters_per_class=1)
_ = plt.scatter(X_features[:, 0], X_features[:, 1], marker="o", c=y_labels, s=25, edgecolors="k")
```


    
![png](/assets/images/sourceImg/classification_files/classification_5_0.png)
    



```python
def visualize_boundary(model, X, y):
    fig,ax = plt.subplots()
    # 학습 데이타 scatter plot으로 나타내기
    _ = ax.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap='rainbow', edgecolor='k',
               clim=(y.min(), y.max()), zorder=3)
    _ = ax.axis('tight')
    _ = ax.axis('off')
    xlim_start , xlim_end = ax.get_xlim()
    ylim_start , ylim_end = ax.get_ylim()
    # 호출 파라미터로 들어온 training 데이타로 model 학습 . 
    model.fit(X, y)
    # meshgrid 형태인 모든 좌표값으로 예측 수행. 
    xx, yy = np.meshgrid(np.linspace(xlim_start,xlim_end, num=200),np.linspace(ylim_start,ylim_end, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # contourf() 를 이용하여 class boundary 를 visualization 수행. 
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap='rainbow', clim=(y.min(), y.max()),
                           zorder=1)
```


```python
clf_dt = DecisionTreeClassifier().fit(X_features, y_labels)
visualize_boundary(clf_dt, X_features, y_labels)
```

    C:\Users\spec3\AppData\Local\Temp\ipykernel_21892\4204971820.py:17: UserWarning: The following kwargs were not used by contour: 'clim'
      contours = ax.contourf(xx, yy, Z, alpha=0.3,
    


    
![png](/assets/images/sourceImg/classification_files/classification_7_1.png)
    



```python
# 과적합 방지
clf_dt = DecisionTreeClassifier(min_samples_leaf=6).fit(X_features, y_labels)
visualize_boundary(clf_dt, X_features, y_labels)
```

    C:\Users\spec3\AppData\Local\Temp\ipykernel_21892\4204971820.py:17: UserWarning: The following kwargs were not used by contour: 'clim'
      contours = ax.contourf(xx, yy, Z, alpha=0.3,
    


    
![png](/assets/images/sourceImg/classification_files/classification_8_1.png)
    


## 앙상블 (Ensemble)
여러 개의 분류기(Classifier)를 생성하고, 각각의 예측을 결합해 최종 예측을 도출하는 기법  

### 학습 유형
1. 보팅(voting)
   1. 하드 보팅(Hard voting): 다수결로 결정된 예측값을 이용
   2. 소프트 보팅(Soft voting): 모든 예측값의 평균을 이용
2. 배깅(bagging): 랜덤 포레스트 (Random Forest)
3. 부스팅(boosting): 그래디언트 부스트, XGBoost, LightGBM
4. 스태킹(stacking)

보팅과 배깅은 여러개의 분류기가 투표를 통해 최종 예측 결과를 결정하는 방식  
보팅은 각각의 분류기가 다르고, 배깅은 각각의 분류기가 같은 알고리즘을 사용함  
  
부스팅은, 여러 개의 분류기가 순차적으로 학습을 수행하지만, 앞의 분류기에서 틀린 문제를 다음 분류기에서 맞출 수 있도록  
가중치(weight)를 부여하는 방식  
  
스태킹은, 여러개의 다른 모델의 예측 결과를 다시 학습 데이터로 만들어 다른 모델로 재학습시켜 결과를 예측하는 방식

#### 부트스트래핑(Bootstrapping) 분할 방식
배깅 방식에서, 각각의 분류기가 데이터를 샘플링해서 추출하는 과정  
각각의 분류기에서 나온 결과를 보팅(voting)을 통해 최종 예측값을 도출함  

### 보팅 분류기 (Voting Classifier)
로지스틱 회귀와 KNN을 기반으로 만든 보팅 분류기 - 위스콘신 유방암 데이터 셋


```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

df.sample(5)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>18.65</td>
      <td>17.60</td>
      <td>123.70</td>
      <td>1076.0</td>
      <td>0.10990</td>
      <td>0.16860</td>
      <td>0.19740</td>
      <td>0.10090</td>
      <td>0.1907</td>
      <td>0.06049</td>
      <td>...</td>
      <td>22.82</td>
      <td>21.32</td>
      <td>150.60</td>
      <td>1567.0</td>
      <td>0.1679</td>
      <td>0.5090</td>
      <td>0.73450</td>
      <td>0.23780</td>
      <td>0.3799</td>
      <td>0.09185</td>
    </tr>
    <tr>
      <th>264</th>
      <td>17.19</td>
      <td>22.07</td>
      <td>111.60</td>
      <td>928.3</td>
      <td>0.09726</td>
      <td>0.08995</td>
      <td>0.09061</td>
      <td>0.06527</td>
      <td>0.1867</td>
      <td>0.05580</td>
      <td>...</td>
      <td>21.58</td>
      <td>29.33</td>
      <td>140.50</td>
      <td>1436.0</td>
      <td>0.1558</td>
      <td>0.2567</td>
      <td>0.38890</td>
      <td>0.19840</td>
      <td>0.3216</td>
      <td>0.07570</td>
    </tr>
    <tr>
      <th>235</th>
      <td>14.03</td>
      <td>21.25</td>
      <td>89.79</td>
      <td>603.4</td>
      <td>0.09070</td>
      <td>0.06945</td>
      <td>0.01462</td>
      <td>0.01896</td>
      <td>0.1517</td>
      <td>0.05835</td>
      <td>...</td>
      <td>15.33</td>
      <td>30.28</td>
      <td>98.27</td>
      <td>715.5</td>
      <td>0.1287</td>
      <td>0.1513</td>
      <td>0.06231</td>
      <td>0.07963</td>
      <td>0.2226</td>
      <td>0.07617</td>
    </tr>
    <tr>
      <th>395</th>
      <td>14.06</td>
      <td>17.18</td>
      <td>89.75</td>
      <td>609.1</td>
      <td>0.08045</td>
      <td>0.05361</td>
      <td>0.02681</td>
      <td>0.03251</td>
      <td>0.1641</td>
      <td>0.05764</td>
      <td>...</td>
      <td>14.92</td>
      <td>25.34</td>
      <td>96.42</td>
      <td>684.5</td>
      <td>0.1066</td>
      <td>0.1231</td>
      <td>0.08460</td>
      <td>0.07911</td>
      <td>0.2523</td>
      <td>0.06609</td>
    </tr>
    <tr>
      <th>320</th>
      <td>10.25</td>
      <td>16.18</td>
      <td>66.52</td>
      <td>324.2</td>
      <td>0.10610</td>
      <td>0.11110</td>
      <td>0.06726</td>
      <td>0.03965</td>
      <td>0.1743</td>
      <td>0.07279</td>
      <td>...</td>
      <td>11.28</td>
      <td>20.61</td>
      <td>71.53</td>
      <td>390.4</td>
      <td>0.1402</td>
      <td>0.2360</td>
      <td>0.18980</td>
      <td>0.09744</td>
      <td>0.2608</td>
      <td>0.09702</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
clf_lr = LogisticRegression(max_iter=10000)
clf_knn = KNeighborsClassifier(n_neighbors=8)

clf_vo = VotingClassifier(estimators=[("LR", clf_lr), ("KNN", clf_knn)], voting="soft")

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)
```


```python
clf_vo.fit(X_train, y_train)

pred_vo = clf_vo.predict(X_test)

accuracy_score(y_test, pred_vo)
```




    0.9649122807017544




```python
# 로지스틱 회귀
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
accuracy_score(y_test, pred_lr)
```




    0.956140350877193




```python
# KNN
clf_knn.fit(X_train, y_train)
pred_knn = clf_knn.predict(X_test)
accuracy_score(y_test, pred_knn)
```




    0.956140350877193



### 랜덤 포레스트 (Random Forest)
여러개의 결정 트리(Decision Tree)를 분류기로 사용하는 알고리즘  - 위스콘신 유방암 데이터 셋


```python
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

clf_rf = RandomForestClassifier(random_state=42)

clf_rf.fit(X_train, y_train)

pred_rf = clf_rf.predict(X_test)

accuracy_score(y_test, pred_rf)
```




    0.9649122807017544



#### 랜덤 포레스트 하이퍼파라미터 튜닝
- `n_estimators`: 결정트리의 개수, `default=10`
- `max_features`: 결정트리와 동일, `default=auto`

#### 그리디 서치를 이용한 튜닝


```python
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100],
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [8, 12, 18],
    "min_samples_split": [8, 16, 20]
}

clf_rf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid_cv = GridSearchCV(clf_rf, param_grid=params, cv=3, n_jobs=-1)
grid_cv.fit(X_train, y_train)
```




    GridSearchCV(cv=3, estimator=RandomForestClassifier(n_jobs=-1, random_state=42),
                 n_jobs=-1,
                 param_grid={'max_depth': [6, 8, 10, 12],
                             'min_samples_leaf': [8, 12, 18],
                             'min_samples_split': [8, 16, 20],
                             'n_estimators': [100]})




```python
grid_cv.best_params_
```




    {'max_depth': 6,
     'min_samples_leaf': 8,
     'min_samples_split': 8,
     'n_estimators': 100}




```python
grid_cv.best_score_
```




    0.9384512606018357



설정해준 범위에서, `max_depth=6, min_samples_leaf=8, min_samples_split=8, n_estimators=100`에서 성능이 가장 높았음  
하이퍼퍼라미터 튜닝 이전이 성능이 더 좋은 듯.. (simple is best?!)


```python
clf_rf = grid_cv.best_estimator_
clf_rf.fit(X_train, y_train)

ftr_importance = pd.Series(clf_rf.feature_importances_, index=cancer.feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
_ = sns.barplot(x=ftr_importance, y=ftr_importance.index)
plt.show()
```


    
![png](/assets/images/sourceImg/classification_files/classification_22_0.png)
    


### GBM (Gradinet Boosting Machine)
부스팅 알고리즘은 여러 개의 약한 학습기(week learner)를 순차적으로 **학습-예측**하면서 잘못 예측한 데이터에 가중치를 부여해 오류를 개선해 나감  
이런 학습 방법은 `AdaBoost(Adaptive boosting)`가 대표적이지만, GBM은 가중치를 주는 방식이 경사 하강법(Gradient Descent)임  

#### 경사 하강법 (Gradient Descent)
$ 오류값 = 실제값 - 예측값 $  
예측 함수를 $ F(x) $라하면, $ h(x) = y - F(x) $가 됨  
이 오류식인 $ h(x) $를 최소화하는 방향성을 가지고 반복적으로 가중치를 업데이트하는 것이 경사 하강법임  

**반복 수행을 통해 오류를 최소화할 수 있도록 가중치를 업데이트하는 방법**

위스콘신 유방암 데이터 셋


```python
from sklearn.ensemble import GradientBoostingClassifier
import time

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

start_time = time.time()

clf_gb = GradientBoostingClassifier(random_state=42)

clf_gb.fit(X_train, y_train)

pred_gb = clf_gb.predict(X_test)

time.time() - start_time
```




    0.2740490436553955




```python
accuracy_score(y_test, pred_gb)
```




    0.956140350877193



#### GBM 하이퍼파라미터 튜닝
주요 파라미터는 모델에서 거의 동일함  
- `loss`: 경사 하강법에서 사용할 비용 함수, `default="deviance"` (비용함수와 손실함수는 추후에 정리할 예정)  
- `learning_rate`: 학습 진행시 적용하는 학습률로, $ \alpha $로 표시함
  - `default=0.1`, `0~1`사이의 값을 지정 가능
  - `n_estimators`와 상호 보완적으로 조합해 사용
- `n_estimators`: weak learner의 개수, `default=100`
- `subsample`: weak learner가 학습에 사용하는 데이터의 샘플링 비율, `default=1`

#### 그리디 서치를 이용한 튜닝



```python
params = {
    "n_estimators": [100, 500],
    "learning_rate": [0.05, 1]
}

grid_cv = GridSearchCV(clf_gb, param_grid=params, cv=3, verbose=1)
grid_cv.fit(X_train, y_train)
```

    Fitting 3 folds for each of 4 candidates, totalling 12 fits
    




    GridSearchCV(cv=3, estimator=GradientBoostingClassifier(random_state=42),
                 param_grid={'learning_rate': [0.05, 1],
                             'n_estimators': [100, 500]},
                 verbose=1)




```python
grid_cv.best_estimator_
```




    GradientBoostingClassifier(learning_rate=0.05, n_estimators=500,
                               random_state=42)




```python
grid_cv.best_score_
```




    0.9560241663762055



### XGBoost (eXtra Gradient Boost)
트리 기반의 앙상블 알고리즘 모델  
분류에 있어서, 일반적으로 가장 좋은 성능을 가짐  
- `GBM`대비 빠른 수행 시간
- 규제(Regularization): 규제를 적용해 과적합을 방지 가능
- 가지 치기(Tree Prunung): 이득이 없는 분할을 없애, 분할 수를 줄임
- 교차 검증 내장: 최적화된 반복 횟수를 가질 수 있음, 조기 중단(early stopping) 지원
- 결측치 제거

#### XGBoost 주요 하이퍼파라미터
1. 일반: 변경할 일이 거의 없음
   1. `booster`: `gbtree(tree based model)`(기본) 또는 `gblinear(linear model)` 선택
   2. `silent`: 출력 메시지를 나타내고 싶지 않으면 1
   3. `nthread`: 실행 스레드 개수, `default=all`
2. 부스터: 트리 최적화, 부스팅, 규제 등
   1. `eta (learning_rate)`: 학습률 조절, `default=0.3`
   2. `num_boost_rounds`: `n_estimators`
   3. `min_chilid_weight`: 추가 분기를 위해 필요한 데이터들의 가중치 총합, 클수록 분할이 적음, `default=1`
   4. `gamma (min_split_loss)`: 분기 결정에 사용할 최소 손실 감소, 지정값보다 클 경우 분기, `default=0`
   5. `max_depth`: `default=6`
   6. `sub_sample`: 데이터 샘플링 비율, `default=1`
   7. `colsample_bytree`: `max_features`와 유사, `default=1`
   8. `lambda (reg_lambda)`: R2 규제 적용값, `default=1`
   9. `alpha (reg_alpha)`: R1 규제 적용값, `default=1`
   10. `scale_pos_weight`: 비대칭한 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터, `default=0`
3. 학습 태스크: 학습 수행 시의 객체 함수, 평가를 위한 지표 등
   1. `objective`: 최솟값을 가져야할 손실 함수 지정 (이진/다중 분류에 따라 달라짐)
      1. `logistic`: 이진 분류일 경우
      2. `softmax` / `softprob`: 다중 분류일 경우
   2. `eval_metric`: 검증에 사용되는 함수
      1. `rmse`: (기본) 회귀
      2. `error`: 분류
      3. `mae, logloss, merror, mlogloss, auc`가 있음

#### Note!!!! 과적합 문제가 발생했다면
뛰어난 알고리즘일수록 파라미터 튜닝을 할 필요가 적어짐 -> 튜닝의 영향이 크지 않기때문  
- `eta` 값을 낮추고 `n_estimators`값을 올려줌
- `max_depth` 값을 낮춤
- `min_child_weight` 값을 높임
- `gamma` 값을 높임
- `sub_sample`과 `colsample_bytree` 조정

#### 조기 종료 (Early Stopping)
반복된 횟수만큼 학습을 진행하면서, 지정된 횟수만큼 성능 개선이 일어나지 않으면 학습을 종료하는 기법


```python
# XGBoost 설치
# !conda install -c anaconda py-xgboost
```


```python
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance
```


```python
xgb.__version__
```




    '1.5.0'




```python
# 위스콘신 유방암 데이터 셋
datasets = load_breast_cancer()
X_features = datasets.data
y_label = datasets.target

df_cancer = pd.DataFrame(data=X_features, columns=datasets.feature_names)
df_cancer["target"] = y_label
df_cancer.sample(5)
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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80</th>
      <td>11.45</td>
      <td>20.97</td>
      <td>73.81</td>
      <td>401.5</td>
      <td>0.11020</td>
      <td>0.09362</td>
      <td>0.045910</td>
      <td>0.022330</td>
      <td>0.1842</td>
      <td>0.07005</td>
      <td>...</td>
      <td>32.16</td>
      <td>84.53</td>
      <td>525.1</td>
      <td>0.1557</td>
      <td>0.1676</td>
      <td>0.17550</td>
      <td>0.06127</td>
      <td>0.2762</td>
      <td>0.08851</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>12.86</td>
      <td>18.00</td>
      <td>83.19</td>
      <td>506.3</td>
      <td>0.09934</td>
      <td>0.09546</td>
      <td>0.038890</td>
      <td>0.023150</td>
      <td>0.1718</td>
      <td>0.05997</td>
      <td>...</td>
      <td>24.82</td>
      <td>91.88</td>
      <td>622.1</td>
      <td>0.1289</td>
      <td>0.2141</td>
      <td>0.17310</td>
      <td>0.07926</td>
      <td>0.2779</td>
      <td>0.07918</td>
      <td>1</td>
    </tr>
    <tr>
      <th>498</th>
      <td>18.49</td>
      <td>17.52</td>
      <td>121.30</td>
      <td>1068.0</td>
      <td>0.10120</td>
      <td>0.13170</td>
      <td>0.149100</td>
      <td>0.091830</td>
      <td>0.1832</td>
      <td>0.06697</td>
      <td>...</td>
      <td>22.88</td>
      <td>146.40</td>
      <td>1600.0</td>
      <td>0.1412</td>
      <td>0.3089</td>
      <td>0.35330</td>
      <td>0.16630</td>
      <td>0.2510</td>
      <td>0.09445</td>
      <td>0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>10.08</td>
      <td>15.11</td>
      <td>63.76</td>
      <td>317.5</td>
      <td>0.09267</td>
      <td>0.04695</td>
      <td>0.001597</td>
      <td>0.002404</td>
      <td>0.1703</td>
      <td>0.06048</td>
      <td>...</td>
      <td>21.18</td>
      <td>75.39</td>
      <td>437.0</td>
      <td>0.1521</td>
      <td>0.1019</td>
      <td>0.00692</td>
      <td>0.01042</td>
      <td>0.2933</td>
      <td>0.07697</td>
      <td>1</td>
    </tr>
    <tr>
      <th>532</th>
      <td>13.68</td>
      <td>16.33</td>
      <td>87.76</td>
      <td>575.5</td>
      <td>0.09277</td>
      <td>0.07255</td>
      <td>0.017520</td>
      <td>0.018800</td>
      <td>0.1631</td>
      <td>0.06155</td>
      <td>...</td>
      <td>20.20</td>
      <td>101.60</td>
      <td>773.4</td>
      <td>0.1264</td>
      <td>0.1564</td>
      <td>0.12060</td>
      <td>0.08704</td>
      <td>0.2806</td>
      <td>0.07782</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



target의 경우 악성(malignant)=0, 양성(benign)=1


```python
# target 분포
df_cancer["target"].value_counts()
```




    1    357
    0    212
    Name: target, dtype: int64




```python
X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}")
```

    X_train: (455, 30)
    y_train: (455,)
    X_test: (114, 30)
    y_test: (114,)
    

- `XGBoost`의 경우, 학습용과 테스트용 데이터 세트를 위해 별도의 객체인 `DMatrix`를 생성해야 함  
- `DMatrix`는 주로 `numpy` 입력 파라미터를 받아서 만들어지는 `XGBoost`의 전용 데이터 세트로, 주요 입력 파라미터는 `data`와 `label`임  
- `DMatrix`로는 `numpy` 외에 `libsvm txt`, `xgboost bin buffer`를 변환 할 수 있음


```python
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)
```


```python
params = {"max_depth": 3,
          "eta": 0.1,
          "object": "binary:logistic",
          "eval_metirc": "logloss", # 주로 error나 logloss를 사용
          "early_stoppings": 100
          }

num_rounds = 400
```


```python
wlist = [(dtrain, "train"), (dtest, "eval")]
# evals에 학습 데이터 셋과 평가 데이터 셋을 넣어주면,
# 평가를 eval 데이터 셋에 수행하면서 조기 종료를 적용 가능
# 조기 종료를 사용하기 위해서는 필수적인 과정임
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist, early_stopping_rounds=100)
```

    
    
    [0]	train-rmse:0.45455	eval-rmse:0.45819
    [1]	train-rmse:0.41436	eval-rmse:0.41937
    [2]	train-rmse:0.37866	eval-rmse:0.38837
    [3]	train-rmse:0.34590	eval-rmse:0.35947
    [4]	train-rmse:0.31705	eval-rmse:0.33606
     ...
    [122]	train-rmse:0.03378	eval-rmse:0.17839
    [123]	train-rmse:0.03334	eval-rmse:0.17841
    [124]	train-rmse:0.03330	eval-rmse:0.17843
    [125]	train-rmse:0.03306	eval-rmse:0.17843
    [126]	train-rmse:0.03283	eval-rmse:0.17844
    

버전 업데이트로 사용법이 바뀐건지.. 몇개의 하이퍼파라미터 설정이 적용되지 않았음.. 당장 중요한건 아니라 넘어감 


```python
pred_probs =xgb_model.predict(dtest)
preds = [1 if x>0.5 else 0 for x in pred_probs]
```

`XGBoost`의 경우 예측값 자체를 반화하는 것이 아니라, 예측 결과를 추정할 수 있는 확률 값을 반환함  
0.5보다 높으면 1(양성), 그렇지 않으면 0(악성)을 반환하게 추가함


```python
accuracy_score(y_test, preds)
```




    0.956140350877193




```python
fig, ax = plt.subplots(figsize=(10, 12))
_ = plot_importance(xgb_model, ax=ax)
```


    
![png](/assets/images/sourceImg/classification_files/classification_46_0.png)
    


#### XGBoost 교차 검증
`GridSearchCV`와 유사하게 교차 검증 수행 후 최적 파라미터를 구할 수 있는 `cv()`를 제공  
자세한 사항은 [공식 문서](https://xgboost.readthedocs.io/en/stable/python/python_api.html)를 참고하자

### Scikit Learn 래퍼 XGBoost
`XGBoost`를 사이킷 런과 유사하게 사용하기 위해 개발됨  
`XGBClassifier`와 `XGBRegressor`가 지원됨


```python
from xgboost import XGBClassifier

xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train, y_train)

pred_xgb = xgb_wrapper.predict(X_test)
pred_xgb_proba = xgb_wrapper.predict_proba(X_test)
```


```python
pred_xgb_proba[:2]
```




    array([[4.6873093e-03, 9.9531269e-01],
           [9.9985588e-01, 1.4412223e-04]], dtype=float32)




```python
accuracy_score(y_test, pred_xgb)
```




    0.9736842105263158



`XGBoost`는 과거에 비해 변경 사항이 조금 많은거 같음..

### LightGBM
`XGBoost`와 같은 부스팅 알고리즘으로, `XGBoost`에 비해 빠르고 리소스를 적게 사용하며, 카테고리형 특성을 자동으로 변경해준다는 장점이 있음  
반면, 데이터 양이 적은 경우 오버피팅 발생 가능성이 높다는 단점이 있음(일반적으로 10k개 이하)  

**리프 중심 트리 분할(Leaf Wise)** 방식을 사용함 -> 기존 트리 기반 알고리즘들은 깊이를 효과적으로 줄이기 위해 균형 트리 분할(Level Wise) 방식을 사용했음  
- `LightGBM`은 트리의 균형을 맞추지 않고, 최대 손실 값(max delta loss)을 가지는 리프 노드를 계속 분할하는 방식

#### LightGBM 주요 하이퍼파라미터
- `num_iterations (n_estimators)`: 반복 수행하려는 트리의 개수, `default=100`
- `learning_rate`: 학습률, `default=0.1`
- `max_depth`: 0보다 작은 값을 지정하면 제한이 없음, `default=-1`
- `min_data_in_leaf (min_child_samples)`: `min_smaples_leaf`와 동일, `default=20`
- `num_leaves`: 하나의 트리가 가질 수 있는 최대 리프 개수, `default=31`
- `boosting`
  - `gbdt`: (기본) 일반적인 그래디언트 부스팅 결정 트리
  - `rf`: 랜덤 포레스트
- `bagging_fraction (sub_sample)`: 데이터 샘플링 비율, `default=1.0`
- `feature_fraction (colsample_bytree)`: 개별 트리 학습시 무작위로 선택되는 특성의 비율, `default=1.0`
- `lambda_l2 (reg_lambda)`: L2 규제값, `default=0.0`
- `lambda_l1 (reg_alpha)`: L1 규제값, `default=0.0`

- Learning Task
  - `objective`: 최솟값을 가져야 할 손실함수


```python
# LightGBM 설치
# !conda install -c conda-forge lightgbm
```


```python
from lightgbm import LGBMClassifier

datasets = load_breast_cancer()

features = datasets.data
target = datasets.target

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

lgbm_wrapper = LGBMClassifier(n_estimators=400)

evals = [(X_test, y_test)]
```


```python
# 조기 종료 사용법은 XGBoost와 동일
lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=False)
```




    LGBMClassifier(n_estimators=400)




```python
preds = lgbm_wrapper.predict(X_test)
pred_probs = lgbm_wrapper.predict_proba(X_test)
```


```python
pred_probs[:2]
```




    array([[0.01109023, 0.98890977],
           [0.99328587, 0.00671413]])




```python
accuracy_score(y_test, preds)
```




    0.9736842105263158




```python
from lightgbm import plot_importance

fig, ax = plt.subplots(figsize=(10, 12))
_ = plot_importance(lgbm_wrapper, ax=ax)
```


    
![png](/assets/images/sourceImg/classification_files/classification_58_0.png)
    

