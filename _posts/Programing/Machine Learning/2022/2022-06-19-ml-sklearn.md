---
title: "머신러닝과 Scikit Learn 기본"
date: 2022-06-19T09:22:38.015Z

categories:
  - Programming
  - Machine Learning
tags:
  - Machine Learning
  - Pandas
  - Seaborn
  - matplot
  - sklearn
  - Tutorial
---


# 머신러닝과 사이킷 런
## 머신러닝
프로그램의 수정 없이 데이터를 기반으로 패턴을 학습하고 결과를 예측하는 알고리즘 기법
### 분류
1. 지도 학습 (Supervised Learning)
   1. 분류
   2. 회귀
   3. 추천 시스템
   4. 시각/음성 감자/인지
   5. 텍스트 분석 / NLP
2. 비지도 학습 (Un-supervised Learning)
   1. 클러스터링 (군집화)
   2. 차원축소 (PCA)
   3. 강화 학습
3. 강화 학습 (Reinforcemnet Learning)
강화 학습은 비지도 학습 방식이지만, 별개의 분야로 보는 견해가 많음  
동적 계획법 기반의 알고리즘

## 사이킷 런 (Scikit-learn)
파이썬에서 가장 인기있는 머신 러닝 라이브러리  

### 기본적인 절차
1. 데이터 세트 분리
2. 모델 학습
3. 예측 수행
4. 평가

### 기본적인 프레임워크
- Estimator: 기본적인 분류(Classification)와 회귀(Regressor) 알고리즘을 합쳐 놓음
- fit(): 학습을 시키는 메서드
- predict(): 학습된 모델에 예측(분류나 회귀)을 시키는 매서드
#### Estimator 알고리즘 분류
- Classifier
  - DecistionTree
  - RandomForest
  - GradientBoosting
  - GaussianNB
  - SVC
- Regressor
  - Linear
  - Ridge
  - Lasso
  - RandomForest
  - GradientBoosting

### 주요 모듈
- `sklearn.datasets`: 예제로 제공하는 데이터 세트
- `sklearn.preprocessing`: 데이터 전처리에 필요한 기능 모음 (인코딩, 정규화, 스케일링)
- `sklearn.feature_selection`: 특성을 우선 순위대로 선택하는 작업을 수행하는 기능 모음
- `sklearn.feature_extraction`: 벡터화된 특성을 추출하는 기능 모음
- `sklearn.decomposition`: 차원 축소 기능 모음
- `sklearn.model_selection`: 교차 검증, 데이터 분리, 하이퍼파라미터 튜닝과 관련된 기능 모음
- `sklearn.metrics`: 성능 측정 방식 모음
- 머신러닝 알고리즘
  - `sklearn.ensemble`: 앙상블 알고리즘 모음
  - `sklearn.linear_model`: 선형 회귀 알고리즘 모음
  - `sklearn.naive_bayes`: 나이브 베이즈 기반 알고리즘 모음
  - `sklearn.neighbors`: 최근접 이웃 알고리즘 모음
  - `sklearn.svm`: 서포트 백터 머신 알고리즘 모음
  - `sklearn.tree`: 트리 기반 알고리즘 모음
  - `sklearn.cluster`: 클러스터링 알고리즘 모음

### 교차 검증 (Cross Validation)
훈련 세트를 여러개의 훈련세트로 나누고, 훈련과 테스트를 반복하며 하이퍼파라미터 튜닝을 할 수 있도록 도움을 줌  
모델의 편향(overfitting/underfitting)을 방지해줌  

GridSearchCV나 RandomSearchCV를 함께 사용해 하이퍼파라미터 튜닝도 함께 진행함

### 대표적인 평가 방식
1. 정확도 (Accuracy)
2. 오차행렬 (Confusion Matrix)
3. 정밀도 (Precision)
4. 재현율 (Recall)
5. F1 score
6. ROC AUC

