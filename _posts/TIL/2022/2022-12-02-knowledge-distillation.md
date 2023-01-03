---
title: 지식 증류 학습
date: 2022-12-02T13:49:14.206Z

categories:
  - TIL
---

## 지식 증류 학습 (Knowledge Distillation)
학습된 모델로부터 지식을 추출하는 학습 방법

### 1. 지식 증류 학습?
- 지식 증류를 처음으로 소개한 논문은 **모델 배포** 측면에서 지식 증류의 필요성을 찾고 있음
- 복잡한 모델이 학습한 **일반화 능력**을 단순한 모델에 전달해주는 것
- Teacher Model과 Studnet Model로 표현

### 2. Distilling the Knowledge in a Neural Network
- 지식 증류 학습이 처음 소개된 논문 (NIPS 2014 workshop)
  - 앙상블과 같은 복잡한 모델을 다량의 유저에게 배포하는 것은 하드웨어적으로 힘든 일
  - 앙상블이 가진 지식을 단일 모델로 전달해주는 기존 기법이 있으나 그 보다 더 일반적인 연구를 함
  - MNIST 데이터를 사용해, 큰 모델로부터 증류된 지식을 작은 모델로 잘 전달되는지 확인
  - 앙상블 학습 시간을 단축하는 새로운 방법의 앙상블을 제시

### HOW?
#### Soft Label
신경망의 마지막 활성 함수를 이용해, 각 클래스의 확률값을 뱉어냄  
-> 출력값의 분포를 좀 더 soft하게 만들면, 모델이 가진 지식이라고 볼 수 있음  

- 기존 출력 : $q_i = {exp(z_i) \over \sum_j(z_j)}$
- soft output : $q_i = {exp(z_i/T) \over \sum_j(z_j)}$

`T`를 온도(temperature)라고 표현했고, 낮아지면 hard하게, 높아지면 soft하게 만듬

#### Distillation Loss
- 큰 모델(T)의 지식을 어떻게 작은 모델(S)에게 넘길 수 있나?
  - 큰 모델(T)을 학습 시킨 후, 작은 모델(S)을 손실함수를 통해 학습

$$L = {\sum_{(x,y) \in D}L_{KD}(S(x,\theta_s,\tau), T(x,\theta_T, \tau))+\lambda L_{CE}(\hat y_s, y)}$$

- T: Teacher model
- S: Student model
- (x, y): 이미지와 레이블
- $\theta$: 학습 파라미터
- $\tau$: temperature
- Cross Entropy Loss와 Distillation Loss로 구성