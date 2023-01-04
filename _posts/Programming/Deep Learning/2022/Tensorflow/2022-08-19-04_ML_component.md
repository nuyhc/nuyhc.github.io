---
title: "[TF로 정리하는 DL] 4. ML Component"
date: 2022-08-19T13:00:16.398Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - Tensorflow
  - Tutorial
---

# 머신 러닝의 기본 요소
## 머신 러닝의 네 가지 분류
### 지도 학습 (Supervised Learning)
샘플 데이터가 주어지면 알고 있는 타깃에 입력 데이터를 매핑하는 방법을 학습  
대부분 분류와 회귀로 구성되지만 예외 사항도 있음  
- 시퀀스 생성(sequence generation): 사진이 주어지면 이를 설명하는 캡션을 생성
- 구문 트리(syntax tree) 예측: 문장이 주어지면 분해된 구문 트리를 예측
- 물체 감지(object detection)
- 이미지 분할(image segmentation)

### 비지도 학습 (Unsupervised Learning)
입력 데이터에 대한 흥미로운 변환을 찾음  
- 차원축소(dimensionality reduction)
- 군집(clustering)

### 자기 지도 학습(Self-Supervised Learning)
지도 학습의 특별한 경우이지만, 사람이 만든 레이블을 사용하지 않음  
-> 학습 과정에서 사람이 개입하지 않음  
-> 경험적인 알고리즘(heuristic algorithm)
- 오토인코더(autoencoder)

### 강화 학습(Reinforcement Learning)
환경에 대한 정보를 받아 보상을 최대화하는 행동을 선택하도록 학습

## 머신러닝 모델 평가
머신러닝의 목표는 처음 본 데이터에서 잘 작동하는 **일반화** 된 모델을 얻는 것
### 훈련, 검증, 테스트 세트
모델 평가의 핵심은 가용한 데이터를 항상 훈련, 검증, 테스트 3개의 세트로 나누는 것  
#### 정보 누설(Information Leak)
검증 세트의 모델 성능에 기반하여 모델의 하이퍼파라미터를 조정할 때마다 검증 데이터에 관한 정보가 모델로 새는 것  
-> 하이퍼파라미터를 조정할 때마다 검증 데이터에 관한 정보가 모델로 새는 것 -> 영향을 줌  
- 홀드아웃 검증(hold-out validation)
- K-겹 교차 검증(K-fold cross-validation)
- 셔플링을 사용한 K-겹 교차 검증(iterated K-fold cross-validation)
##### 홀드아웃 검증(hold-out validation)
데이터의 일정량을 테스트 세트로 떼어 놓고, 남은 데이터에서 훈련하고 테스트 세트로 평가


```python
num_validation_samples = 10000

np.random.shuffle(data)
# 검증 데이터
validation_data = data[:num_validation_samples]
# 훈련용 데이터
data = data[num_validation_samples:]
training_data = data[:]
# 모델을 생성하고,
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# 모델 튜닝 진행 후,
# 훈련과 평가 반복

# 튜닝 종료후, 모든 데이터를 이용해 훈련을 다시 시킴
model.train(np.concatenate([training_data, validation_data]))
test_score = model.evaluate(test_data)
```

##### K-겹 교차 검증(K-fold cross-validation)
데이터를 동일한 크기를 가진 K개로 분할하고, 각 분할에 대해 K-1개로 훈련을하고 나머지로 검증을 함


```python
k = 4
num_validation_samples = len(data)//4

np.random.shuffle(data)

validation_score = []
for fold in range(k):
    validation_data = data[num_validation_samples*fold:num_validation_samples*(fold+1)]
    training_data = data[:num_validation_samples*fold] + data[num_validation_samples*(fold+1):]
    
    model = get_model()
    model.train(training_data)
    validation_score.append(model.evaluate(validation_data))

validation_score = np.average(validation_score)

# 최종 훈련
model = get_model()
model.train(data)

test_score = model.evaluate(test_data)
```

##### 셔플링을 사용한 반복 K-겹 교차 검증
가용 데이터가 적고 가능한 정확하게 모델을 평가하고자하는 경우에 사용  
K-겹 교차 검증을 여러 번 적용하되 K개의 분할로 나누기 전에 매번 데이터를 무작위로 섞음 -> 최종 점수는 모든 평균 값

## 데이터 전처리, 특성 공학, 특성 학습
### 신경망을 위한 데이터 전처리
데이터 전처리 목적은 주어진 원본 데이터를 신경망에 적용하기 쉽도록 만드는 것  
- 벡터화(vectorization)
- 정규화(normalization)
- 결측치 처리
- 특성 추출

## 과대적합과 과소적합
머신러닝의 근본적인 이슈는 최적화와 일반화 사이의 줄다기  
- 최적화(optimization): 가능한 훈련 데이터에서 최고 성능을 얻으려고 모델을 조정하는 과정
- 일반화(generalization): 훈련된 모델이 이전에 본 적 없는 데이터에서 얼마나 잘 수행되는지

훈련 및 검증 데이터(한본 본 데이터)에서는 성능이 좋으나 테스트 데이터에서는 성능이 좋지 않음 -> 과대적합(overfitting)  
훈련 및 검증 데이터, 테스트 데이터에서 모두 성능이 좋지 않음 -> 과소적합(underfitting)

### 네트워크 크기 축소
오버피팅을 막는 가장 단순한 방법

### 가중치 규제(Weight Requlrization)
오컴의 면도날(Occam's razor) -> 어떤 것에 대한 두 가지 설명이 있다면 더 적은 가정이 필요한 간단한 설명이 옳을 것이라는 이론  
간단한 모델 -> 파라미터 값 분포의 엔트로피가 작은 모델 -> 과대적합을 완하하기 위한 일반적인 방법으로 모델 복잡도에 제한을 두어 가중치가 작은 값을 가지도록 규제하는 것  
- L1 규제: 가중치의 절대값에 비례하는 비용이 추가
- L2 규제: 가중치의 제곱에 비례하는 비용이 추가, 가중치 감쇠(weight decay)라고도 함


```python
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```


```python
import numpy as np

def vectorize_sequence(sequence, dimension=10000):
    result = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        result[i, sequence] = 1.
    return result

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L2
import tensorflow as tf

model = Sequential()
# 모델에 L2 규제 추가
model.add(Dense(units=16, kernel_regularizer=L2(0.001), activation="relu", input_shape=(10000, )))
model.add(Dense(units=16, kernel_regularizer=L2(0.001)))
model.add(Dense(units=1, activation="sigmoid"))
```

### 드롭아웃(dropout)
신경망을 위해 사용되는 규제 기법 중에서 가장 효과적이고 널리 사용되는 방법 중 하나  
훈련하는 동안 무작위로 층의 일부 출력 특성을 제외시킴  
-> 테스트 단계에서는 적용되지 않음


```python
from tensorflow.keras.layers import Dropout

model = Sequential()
# 모델에 드롭아웃 추가
model.add(Dense(16, activation="relu", input_shape=(10000, )))
model.add(Dropout(0.5))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
```

## 머신러닝 작업 흐름
1. 문제 정의와 데이터 수집
2. 지표 선택
3. 평가 방법 선택
4. 모델 생성
5. 모델 규제와 하이퍼파라미터 튜닝

|문제 유형|출력층 활성화 함수|손실함수|
|------|---|---|
|이진 분류|시그모이드|binary_crossentropy|
|단일 레이블 다중 분류|소프트맥스|categorical_crossentropy|
|다중 레이블 다중 분류|시그모이드|binary_crossentropy|
|임의 값에 대한 회귀| |mse|
|0과 1 사이 값에 대한 회귀|시그모이드|mse 또는 binary_crossentropy|
