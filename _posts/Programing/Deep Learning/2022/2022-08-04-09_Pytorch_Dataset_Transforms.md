---
title: "[PyTorch] Transform"
date: 2022-08-04T00:07:43.049Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Transform (변형)
데이터가 항상 학습에 필요한 최종 처리가 된 형태로 제공되지 않음  
-> 변형(transform)을 해서 데이터를 조작하고 학습에 적합하게 만듬  

모든 TorchVision 데이터셋들은 변형 로직을 갖는, 호출 가능한 객체(callable)를 받는 매개변수 두개를 갖음  
- 특징(feature): `transform`
- 정답(label): `target_transform`


```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float.scatter_(0, torch.tensor(y), value=1)))
)
```

`ToTensor()`는 PIL Image나 NumPy `ndarray`를 `FloatTensor`로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)  
`Lambda` 변형은 사용자 정의 람다 함수를 적용
