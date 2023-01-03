---
title: "[PyTorch] nn.Module"
date: 2022-08-03T00:19:08.650Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# nn.Module
# 신경망 모델 구성하기
신경망은 데이터에 대한 연산을 수행하는 계층(layer)/모듈(module)로 구성  
`torch.nn` 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공(PyTorch의 모든 모듈은 `nn.Module`의 하위 클래스(subclass))

## FashionMNIST 데이터셋 이미지 분류


```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

### 클래스 정의
신경망 모델을 `nn.Moudle`의 하위 클래스로 정의하고, `__init__`에서 신경망 계층들을 초기화  
`nn.Module`을 상속 받은 모든 클래스는 `forward` 메소드에 입력 데이터에 대한 연산들을 구현


```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

#### 모델 구조(Model Structure)


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

    Using cpu device
    


```python
model = NeuralNetwork().to(device)
print(model)
```

    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    


```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted Class: {y_pred}")
```

    Predicted Class: tensor([3])
    

### 모델 계층(Layer)
FashionMNIST 모델의 계층들을 살펴보기 위해, 28*28 크기의 이미지 3개로 구성된 미니배치를 가져와 신경망을 통과할 때 어떤 일이 발생하는지 알아봄


```python
input_image = torch.rand(3, 28, 28)
print(input_image.size())
```

    torch.Size([3, 28, 28])
    

#### nn.Flatten
`nn.Flatten` 계층을 초기화해 각 28*28의 2D 이미지를, 784 픽셀 값을 갖는 연속된 배열로 변환


```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

    torch.Size([3, 784])
    

#### nn.Linear
선형 계층은 저장된 가중치(weight)와 편향(bias)을 사용해 입력에 선형 변환(linear transformation)을 적용하는 모듈


```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

    torch.Size([3, 20])
    

#### nn.ReLU
비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듬  
비선형 활성화는 선형 변환 후에 적용되어 비선형성(nonlinearity)을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 도움


```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

    Before ReLU: tensor([[-0.2651, -0.5887, -0.4864,  0.1549,  0.4564, -0.4323, -0.2960, -0.3037,
              0.5260, -0.1421, -0.5691, -0.4042,  0.3189,  0.0369,  0.0328,  0.2001,
             -0.4554,  0.0499,  0.4366,  0.5875],
            [ 0.2950, -0.3271, -0.3917,  0.0481,  0.2082, -0.3425, -0.0608, -0.7253,
             -0.0373, -0.0830, -0.6450, -0.5228,  0.0753,  0.1797, -0.0839,  0.4757,
             -0.2450,  0.2931,  0.2034,  0.3196],
            [ 0.1540, -0.6391, -0.6594,  0.1505,  0.2341, -0.5107, -0.0886, -0.2877,
              0.1785,  0.1132, -0.3678, -0.4322,  0.4038,  0.3021, -0.2347,  0.2109,
             -0.2043,  0.1801,  0.4550,  0.5256]], grad_fn=<AddmmBackward0>)
    
    
    After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.1549, 0.4564, 0.0000, 0.0000, 0.0000, 0.5260,
             0.0000, 0.0000, 0.0000, 0.3189, 0.0369, 0.0328, 0.2001, 0.0000, 0.0499,
             0.4366, 0.5875],
            [0.2950, 0.0000, 0.0000, 0.0481, 0.2082, 0.0000, 0.0000, 0.0000, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0753, 0.1797, 0.0000, 0.4757, 0.0000, 0.2931,
             0.2034, 0.3196],
            [0.1540, 0.0000, 0.0000, 0.1505, 0.2341, 0.0000, 0.0000, 0.0000, 0.1785,
             0.1132, 0.0000, 0.0000, 0.4038, 0.3021, 0.0000, 0.2109, 0.0000, 0.1801,
             0.4550, 0.5256]], grad_fn=<ReluBackward0>)
    

#### nn.Sequential
`nn.Sequential`은 순서를 갖는 모듈의 컨테이너  
데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달, 순차 컨테이너(sequential containenr)를 사용하여 아래의 `seq_modules`와 같은 신경망을 만들 수 있음


```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)
```

#### nn.Softmax
신경망의 마지막 선형 계층은 nn.Softmax 모듈에 전달될 logits를 반환  
logits는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록 [0, 1] 범위로 비례해 조정(scale)


```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

#### 모델 매개변수
신경망 내부의 많은 계층들은 매개변수화(parameterize)됨  
즉, 학습 중에 최적화되는 가중치와 편향과 연관지어 짐  
`nn.Module`을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적(track)되며, 모델의 `parameters()` 및 `named_parameters()` 메소드로 모든 매개변수에 접근 가능  

각 매개변수들을 순회(iterate)하며, 매개변수의 크기와 값을 출력


```python
print(f"Model structure: {model}")
print("--------------------------------------------")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}")
```

    Model structure: NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    --------------------------------------------
    Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values: tensor([[-1.8840e-02, -3.0724e-02, -1.5696e-03,  ...,  3.3835e-02,
              1.0598e-02,  2.9609e-04],
            [-3.4447e-02, -1.5560e-02,  6.4176e-05,  ..., -1.6390e-02,
             -3.9231e-03,  4.5546e-03]], grad_fn=<SliceBackward0>)
    Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values: tensor([-0.0249,  0.0345], grad_fn=<SliceBackward0>)
    Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values: tensor([[-0.0359,  0.0187,  0.0137,  ...,  0.0227, -0.0354, -0.0129],
            [ 0.0169, -0.0083, -0.0046,  ...,  0.0166,  0.0015,  0.0325]],
           grad_fn=<SliceBackward0>)
    Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values: tensor([-0.0280, -0.0275], grad_fn=<SliceBackward0>)
    Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values: tensor([[-0.0063,  0.0275,  0.0014,  ...,  0.0205,  0.0098,  0.0348],
            [ 0.0125, -0.0432,  0.0020,  ..., -0.0373,  0.0205, -0.0032]],
           grad_fn=<SliceBackward0>)
    Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values: tensor([-0.0336,  0.0014], grad_fn=<SliceBackward0>)
    
