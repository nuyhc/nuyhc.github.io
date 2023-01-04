---
title: "[PyTorch] Model Checkpoint"
date: 2022-08-04T12:17:47.900Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Model CheckPoint
## 모델 저장하고 불러오기
모델의 상태를 유지(persist)하기 위해 모델을 저장하고 불러와 모델의 예측을 실행하는 방법


```python
import torch
import torchvision.models as models
```

### 모델 가중치 저장하고 불러오기
PyTorch 모델은 학습한 매개변수를 `state_dict`라고 불리는 내부 상태 사전(internal state dictionary)에 저장  
-> `torch.save` 메소드를 사용해 저장(persist)


```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

모델 가중치를 불러오기 위해서는, 먼저 동일한 모델의 인스턴스(instance)를 생성한 다음에 `load_state_dict()` 메소드를 사용해 매개변수들을 불러옴


```python
model = models.vgg16()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()
```

> 추론(inference)을 하기 전에 `model.eval()` 메소드를 호출하여 드롭아웃(dropout)과 배치 정규화(batch normalization)를 평가 모드(evaluation mode)로 설정해야 함  
> 그렇지 않으면 일관성 없는 추론 결과가 생성

### 모델의 형태를 포함하여 저장하고 불러오기
모델의 가중치를 불러올 때, 신경망의 구조를 정의하기 위해 모델 클래스를 먼저 생성(instaniate)해야 했음  
-> 이 클래스의 구조를 모델과 함께 저장하고 싶으면, `model`을 저장 함수에 전달


```python
torch.save(model, "model.pth")
```


```python
# 저장한 모델 불러오기
model = torch.load("model.pth")
```
