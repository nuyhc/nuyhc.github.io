---
title: 회귀 (Regression) 정리
date: 2022-07-25T14:19:04.136Z

categories:
  - Programming
  - Machine Learning
tags:
  - sklearn
  - Tutorial
---

# 회귀(Regession)
통계학에서의 회귀 -> 여러 개의 독립 변수와 한 개의 종속 변수 간의 상관 관계를 모델링하는 기법  
독립 변수의 값에 여향을 미치는 **회귀 계수(Regression Coefficient)**  
  
머신러닝의 회귀는, 주어진 피처(특징/특성)와 결정 값 데이터 기반에서 학습을 통해 최적의 회귀 계수를 찾아내는 것  

## 대표적인 회귀 모델
- 일반 선형 회귀
- 릿지(Ridge): 선형 회귀에 L2 규제를 추가한 회귀 모델
- 라쏘(Lasso): 선형 회귀에 L1 규제를 추가한 회귀 모델
- ElasticNet: L1과 L2 규제를 결합한 회귀 모델

#### L1 규제와 L2 규제
- L1 규제: 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 선택되지 않게하는 것
- L2 규제: 상대적으로 큰 회귀 계수의 영향력을 감소 시키기 위해, 회귀 계수값을 더 작게 만드는 것

## 경사 하강법(Gradient Descent)
$RSS(w_0, w_1) = {1 \over N} \sum^N_{i=1} (y_i - (w_0 + w_1 \times x_i))^2$  
실제 값과 회귀 모델의 차이에 따른 오류 값을 잔차라하며, 최적의 회귀 모델을 만든다는 것은 잔차(오류 값)를 최소화하는 모델을 만드는 것  
보통 오류 합을 계산할 때는 절댓값을 취해 더하거나(MAS), 오류 값의 제곱을 더하는 방식(RSS, Residual Sum of Square)을 취함  
회귀에서 RSS는 비용(cost)이며, 이를 비용 함수(cost func.)라고 함  

#### 비용 함수와 손실 함수
- 하나의 관측치 단위 -> 비용 함수 (cost func.)
- 배치(batch) 단위 -> 손실 함수 (loss func.)


경사 하강법은 비용 함수가 최소가 되는 W 값을 찾는 효율적인 방법임  
업데이트시, 편미분한 결과를 더하는데 편미분한 결과에 학습률(learning rate)를 곱해서 더함


```python
import numpy as np
import matplotlib.pyplot as plt

# y = 4x + 6 근사
X = 2*np.random.rand(100, 1)
y = 6 + 4 * X+np.random.randn(100, 1)

_ = plt.scatter(X, y)
```


    
![png](/assets/images/sourceImg/2022-07-25-regression_model_files/2022-07-25-regression_model_1_0.png)
    



```python
# Cost func.
def get_cost(y, y_pred):
    return np.sum(np.square(y-y_pred))/len(y)

# w1과 w0를 업데이트하는 함수
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)
    y_pred = np.dot(X, w1.T)+w0
    diff = y-y_pred
    
    w0_factors = np.ones((len(y), 1))
    w1_update = -(2/len(y)*learning_rate*np.dot(X.T, diff))
    w0_update = -(2/len(y)*learning_rate*np.dot(w0_factors.T, diff))
    
    return w1_update, w0_update

# 경사 하강법
def GDS(X, y, iters=10000):
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    for _ in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1, w0
```


```python
w1, w0 = GDS(X, y, 1000)
y_pred = w1[0, 0]*X+w0
print(f"w1: {w1[0, 0]:.3f}\nw0: {w0[0, 0]:.3f}\ncost: {get_cost(y, y_pred):.3f}")
```

    w1: 3.999
    w0: 5.921
    cost: 1.165
    

근사한 값과 유사한 값이 나옴 (`w1, w0`)


```python
_ = plt.scatter(X, y)
_ = plt.plot(X, y_pred)
```


    
![png](/assets/images/sourceImg/2022-07-25-regression_model_files/2022-07-25-regression_model_5_0.png)
    


경사 하강법은 모든 학습 데이터에 대해 반복적으로 비용 함수 최소화를 위한 값을 업데이트하기 때문에, **확률적 경사 하강법(Stochastic Gradient Descent)** 을 이용


```python
def SGD(X, y, batch_size=10, iters=1000):
    w0 = np.zeros((1, 1))
    w1 = np.zeros((1, 1))
    prev_cost = 100000
    iter_idx = 0
    for idx in range(iters):
        # 배치 사이즈 만큼 랜덤하게 샘플링
        stochastic_random_idx = np.random.permutation(X.shape[0])
        sample_X = X[stochastic_random_idx[0:batch_size]]
        sample_y = y[stochastic_random_idx[0:batch_size]]
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    return w1, w0
```


```python
w1, w0 = SGD(X, y, 1000)
y_pred = w1[0, 0]*X+w0
print(f"w1: {w1[0, 0]:.3f}\nw0: {w0[0, 0]:.3f}\ncost: {get_cost(y, y_pred):.3f}")
```

    w1: 3.999
    w0: 5.921
    cost: 1.165
    

## 회귀 평가 지표
$MAE(Mean Absolute Error) = {1 \over n} \sum^n_{i=1}|{Y_i} - \hat {Y_i}|$  
$MSE(Mean Squared Error) = {1 \over n} \sum^n_{i=1} ({Y_i} - \hat {Y_i})^2$  
$RMSE(Root MSE) = \sqrt {MSE}$  
$R^2 = {예측값 var \over 실제값 var}$

## 다항 회귀
다항 회귀는 비선형 회귀가 아니라 선형 회귀임  
`sklearn`에서는 비선형 함수를 선형 모델에 적용시키는 방법을 사용해 적용


```python
from sklearn.preprocessing import PolynomialFeatures

X = np.arange(4).reshape(2, 2)
X
```




    array([[0, 1],
           [2, 3]])




```python
poly = PolynomialFeatures(degree=2)
poly.fit(X)
poly_ftr = poly.transform(X)
poly_ftr
```




    array([[1., 0., 1., 0., 0., 1.],
           [1., 2., 3., 4., 6., 9.]])



$[x_1, x_2]  -> [1, x_1, x_2, x_1^2, x_1x_2, {x_2}^2]$

## 규제 선형 모델
과소/과대 적합 방지를 위해 규제가 필요함  
$비용 함수 목표 = Min(RSS(W) + alpha * ||W||^2_2)$  
`alpha`는 학습 데이터가 적합 정도와 회귀 계수 값의 크기 제어를 수행하는 튜닝 파라미터
- L1 규제: $ alpha*||W||_1 $
- L2 규제: $ alpha*||W||^2_2 $
