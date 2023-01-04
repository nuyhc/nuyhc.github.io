---
title: "[PyTorch] Autograd"
date: 2022-08-03T00:19:58.208Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Autograd
## TORCH.AUTOGRAD를 사용한 자동 미분
신경망을 학습할 때 가장 자주 사용되는 알고리즘 **역전파**  
매개변수(모델 가중치)는 주어진 매개변수에 대한 손실 함수의 **변화도(gradient)** 에 따라 조정  
이러한 변화도를 계산하기 위해 PyTorch에는 `torch.autograd`라고 불리는 자동 미분 엔진이 내장되어있음  
-> 모든 계산 그래프에 대한 변화도의 자동 계산을 지원  

입력 x, 매개변수 w와 b , 그리고 일부 손실 함수가 있는 가장 간단한 단일 계층 신경망을 가정


```python
import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

## Tensor, Function과 연산 그래프(Computational Graph)

![png](https://tutorials.pytorch.kr/_images/comp-graph.png)

이 신경망에서, w와 b는 최적화를 해야 하는 매개변수  
-> 손실 함수의 변화도를 계산할 수 있어야 함

> `requires_grad`의 값은 텐서를 생성할 때 설정하거나, 나중에 `x.requires_grad_(True)` 메소드를 사용해 설정 가능

연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 사실 `Function` 클래스의 객체  
이 객체는 순전파 방향으로 함수를 계산하는 방법과, 역방향 전파 단계에서 도함수(derivative)를 계산하는 방법을 알고있음  
역방향 전파 함수에 대한 참조(reference)는 텐서의 `grad_fn` 속성에 저장됨


```python
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
```

    Gradient function for z = <AddBackward0 object at 0x0000014B498C2D90>
    Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x0000014B498C2760>
    

## 변화도(Gradient) 계산하기
신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수(derivative)를 계산해야 함  
즉, x와 y의 일부 고정값에서 $\partial loss \over \partial w$ 와 $\partial loss \over \partial b$ 가 필요


```python
loss.backward()
print(w.grad)
print(b.grad)
```

    tensor([[0.0859, 0.0468, 0.2186],
            [0.0859, 0.0468, 0.2186],
            [0.0859, 0.0468, 0.2186],
            [0.0859, 0.0468, 0.2186],
            [0.0859, 0.0468, 0.2186]])
    tensor([0.0859, 0.0468, 0.2186])
    

> 연산 그래프의 잎(leaf) 노드들 중 `requires_grad` 속성이 `True`로 설정된 노드들의 `grad` 속성만 구할 수 있음 (모든 노드에서 변화도가 기록되는 것은 아님)  
> 성능상의 이유로, 주어진 그래프에서의 `backward`를 사용한 변화도 계산은 한 번만 수행할 수 있음 (여러번 수행해야하는 경우, `retrain_graph=True`를 전달해야 함)

## 변화도 추적 멈추기
기본적으로, `requires_grad=True`인 모든 텐서들은 연산 기록을 추적하고 변화도 계산을 지원  
그러나 모델을 학습한 뒤 입력 데이터를 단순히 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우에는, 이러한 추적이나 지원이 필요하지 않을 수 있음  
연산 코드들을 `torch.no_grad()` 블록으로 둘러싸서 연산 추적을 멈출 수 있음


```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

    True
    False
    


```python
# 동일한 메소드
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

    False
    

변화도 추적을 멈춰야하는 이유
- 신경망의 일부 매개변수를 **고정된 매개변수(frozen parameter)** 로 표시 -> 사전 학습된 신경망을 미세조정 할 때 유용
- 변화도를 추적하지 않는 텐서의 연산이 더 효율적 -> 순전파 단계만 수행할 때 연산 속도가 향상

[추가로 볼만한 정리 글](https://nuyhc.github.io/deep%20learning/02_Pytorch_Introduction_Variable/)
