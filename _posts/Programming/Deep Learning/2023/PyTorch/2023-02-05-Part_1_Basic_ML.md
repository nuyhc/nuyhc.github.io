---
title: "[BC] 모두를 위한 딥러닝 1 - Basic ML"
date: 2023-02-05T14:34:34.605Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Part-1 Basic ML

## Lab-01-1 Tensor Manipulation 1
- 텐서(Tensor)
- 넘파이(NumPy)
- 텐서 조작(Tensor Manipulation)
- 브로드캐스팅(Broadcasting)

### Tensor
- 2D Tensor: $|t| = (batch size, \ dim)$
- 3D Tensor: $|t| = (batch size, \ width, \ height)$ (vision)
- 3D Tensor: $|t| = (batch size, \ length, \ dim)$ (NLP, Seq.)


```python
import torch
import numpy as np
```

#### 1D Array with NumPy


```python
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
```

    [0. 1. 2. 3. 4. 5. 6.]
    


```python
print(f"Rank of t: {t.ndim}")
print(f"Shape of t: {t.shape}")
```

    Rank of t: 1
    Shape of t: (7,)
    


```python
# Element
print(t[0], t[1], t[-1])
# Slicing
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
```

    0.0 1.0 6.0
    [2. 3. 4.] [4. 5.]
    [0. 1.] [3. 4. 5. 6.]
    

#### 2D Array with NumPy


```python
t = np.array(
    [[1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.],
     [10., 11., 12.]]
)

print(t)
```

    [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]
     [10. 11. 12.]]
    


```python
print(f"Rank of t: {t.ndim}")
print(f"Shape of t: {t.shape}")
```

    Rank of t: 2
    Shape of t: (4, 3)
    

#### 1D Array with PyTorch


```python
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)
```

    tensor([0., 1., 2., 3., 4., 5., 6.])
    


```python
# Rank
print(f"Rank: {t.dim()}")
# Shape
print(f"Shape: {t.shape}")
print(f"Shape: {t.size()}")
# Element
print(t[0], t[1], t[-1])
# Slicing
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
```

    Rank: 1
    Shape: torch.Size([7])
    Shape: torch.Size([7])
    tensor(0.) tensor(1.) tensor(6.)
    tensor([2., 3., 4.]) tensor([4., 5.])
    tensor([0., 1.]) tensor([3., 4., 5., 6.])
    

#### 2D Array with PyTorch


```python
t = torch.FloatTensor(
    [[1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.],
     [10., 11., 12.]]
)

print(t)
```

    tensor([[ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])
    


```python
# Rank
print(f"Rank: {t.dim()}")
# Shape
print(t.size())
# Slicing
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])
```

    Rank: 2
    torch.Size([4, 3])
    tensor([ 2.,  5.,  8., 11.])
    torch.Size([4])
    tensor([[ 1.,  2.],
            [ 4.,  5.],
            [ 7.,  8.],
            [10., 11.]])
    

### Broadcasting
다른 크기의 행렬을 연산 할 때 적용되는 기능


```python
# same shape
m1 = torch.FloatTensor([[3, 3]]) # (1, 2)
m2 = torch.FloatTensor([[2, 2]]) # (1, 2)

print(m1.shape, m2.shape)
print(m1+m2)
```

    torch.Size([1, 2]) torch.Size([1, 2])
    tensor([[5., 5.]])
    


```python
# Vec + scaler
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])

print(m1.shape, m2.shape)
print(m1+m2)
```

    torch.Size([1, 2]) torch.Size([1])
    tensor([[4., 5.]])
    


```python
# 2*1 vec + 1*2 vec
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])

print(m1.shape, m2.shape)
print(m1+m2)
```

    torch.Size([1, 2]) torch.Size([2, 1])
    tensor([[4., 5.],
            [5., 6.]])
    

$$
\begin{bmatrix}
1 & 2
\end{bmatrix}
+
\begin{bmatrix}
3 \\
4
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 \\
1 & 2
\end{bmatrix}
+
\begin{bmatrix}
3 & 3 \\
4 & 4
\end{bmatrix}
=
\begin{bmatrix}
4 & 5 \\
5 & 6
\end{bmatrix}
$$

### Multiplication vs Matrix Multiplication
- 딥러닝은 행렬곱을 굉장히 많이 사용하는 알고리즘


```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])

print(f"Shape of Matrix 1: {m1.shape}") # 2 x 2
print(f"Shape of Matrix 2: {m2.shape}") # 2 x 1
```

    Shape of Matrix 1: torch.Size([2, 2])
    Shape of Matrix 2: torch.Size([2, 1])
    


```python
print(m1)
print("-"*10)
print(m2)
```

    tensor([[1., 2.],
            [3., 4.]])
    ----------
    tensor([[1.],
            [2.]])
    


```python
m1.matmul(m2)
```




    tensor([[ 5.],
            [11.]])




```python
m1*m2
```




    tensor([[1., 2.],
            [6., 8.]])



#### !!NOTE!!
- `np.matmul(a, b)` == `a@b`: 2D의 행렬곱
- `np.dot(a, b)`:
  - 1D의 내적
  - 2D의 행렬곱
  - nD의 경우, 첫 행렬의 마지막 축과 두 번째 행렬의 -2번째 축의 내적


```python
a = np.array(
    [[1, 2],
     [3, 4]]
)

b = np.array(
    [[1, 2],
     [3, 4]]
)
```


```python
np.matmul(a, b)
```




    array([[ 7, 10],
           [15, 22]])




```python
np.dot(a, b)
```




    array([[ 7, 10],
           [15, 22]])




```python
# point-wise
a*b
```




    array([[ 1,  4],
           [ 9, 16]])




```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 3])
```


```python
np.matmul(a, b), np.dot(a, b)
```




    (14, 14)




```python
a*b
```




    array([1, 4, 9])



### Mean


```python
t = torch.FloatTensor([1, 2])
print(t.mean())
```

    tensor(1.5000)
    


```python
t = torch.LongTensor([1, 2])
try: print(t.mean())
except Exception as e:
    print(e)
```

    mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
    

평균(Mean)은 정수형 텐서로는 못구함


```python
t = torch.FloatTensor(
    [[1, 2],
     [3, 4]]
)
print(t)
```

    tensor([[1., 2.],
            [3., 4.]])
    

### Sum


```python
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
```

    tensor([[1., 2.],
            [3., 4.]])
    


```python
print(t.sum()) # 10
print(t.sum(dim=0)) # 4, 6
print(t.sum(dim=1)) # 3, 7
```

    tensor(10.)
    tensor([4., 6.])
    tensor([3., 7.])
    

### Max and Argmax


```python
t.max() # 4
```




    tensor(4.)




```python
# Returns max and argmax
print(t.max(dim=0))
```

    torch.return_types.max(
    values=tensor([3., 4.]),
    indices=tensor([1, 1]))
    

## Lab-01-2 Tensor Manipulation 2
- 텐서(Tensor)
- 넘파이(NumPy)
- 텐서 조작(Tensor Manipulation)
- View, Squeeze, UnSqueeze, Type Casting, Concatenate, Stacking, In-place Operation

### View (Reshape)
- `numpy.reshape`와 유사


```python
t = np.array(
    [[[0, 1, 2],
      [3, 4, 5]],
     
     [[6, 7, 8],
      [9, 10, 11]]]
)

ft = torch.FloatTensor(t)

print(ft.shape)
```

    torch.Size([2, 2, 3])
    


```python
print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)
```

    tensor([[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.],
            [ 9., 10., 11.]])
    torch.Size([4, 3])
    


```python
print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)
```

    tensor([[[ 0.,  1.,  2.]],
    
            [[ 3.,  4.,  5.]],
    
            [[ 6.,  7.,  8.]],
    
            [[ 9., 10., 11.]]])
    torch.Size([4, 1, 3])
    

### Squeeze
- 특정 차원의 엘리먼트가 1인 경우, 해당 차원을 지워줌
- `.squeeze(dim=?)`: ? 차원에 엘리먼트가 1인 경우가 있으면, 해당 차원을 지워줌


```python
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    


```python
print(ft.squeeze())
print(ft.squeeze().shape)
```

    tensor([0., 1., 2.])
    torch.Size([3])
    

### Unsqueeze
- 차원은 추가


```python
ft = torch.Tensor([0, 1, 2])

print(ft)
print(ft.shape)
```

    tensor([0., 1., 2.])
    torch.Size([3])
    


```python
print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)
```

    tensor([[0., 1., 2.]])
    torch.Size([1, 3])
    


```python
print(ft.view(1, -1))
print(ft.view(1, -1).shape)
```

    tensor([[0., 1., 2.]])
    torch.Size([1, 3])
    


```python
print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    


```python
print(ft.view(-1, 1))
print(ft.view(-1, 1).shape)
```

    tensor([[0.],
            [1.],
            [2.]])
    torch.Size([3, 1])
    

### Type Casting


```python
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
```

    tensor([1, 2, 3, 4])
    tensor([1., 2., 3., 4.])
    


```python
# 마스킹하는 경우 사용 가능
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())
```

    tensor([1, 0, 0, 1], dtype=torch.uint8)
    tensor([1, 0, 0, 1])
    tensor([1., 0., 0., 1.])
    

### Concatenate
- 이어 붙이기


```python
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))
```

    tensor([[1., 2.],
            [3., 4.],
            [5., 6.],
            [7., 8.]])
    tensor([[1., 2., 5., 6.],
            [3., 4., 7., 8.]])
    

### Stacking


```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=0))
print(torch.stack([x, y, z], dim=1))
```

    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    tensor([[1., 4.],
            [2., 5.],
            [3., 6.]])
    tensor([[1., 2., 3.],
            [4., 5., 6.]])
    

### Ones and Zeros


```python
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
```

    tensor([[0., 1., 2.],
            [2., 1., 0.]])
    


```python
print(torch.ones_like(x))
print(torch.zeros_like(x))
```

    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    

같은 디바이스에 텐서를 생성하게 됨

### In-place Opertaion


```python
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)
```

    tensor([[2., 4.],
            [6., 8.]])
    tensor([[1., 2.],
            [3., 4.]])
    tensor([[2., 4.],
            [6., 8.]])
    tensor([[2., 4.],
            [6., 8.]])
    

## Lab-02 Linear regression
- 선형회귀(Linear Regression)
- 평균 제곱 오차(MSE)
- 경사하강법(Gradient Descent)

### Data Definition
| Hours(x) | Points(y) |
| :---: | :---: |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | ? |


```python
# train data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```

### Hypothesis
$y = Wx + b$ <- 선형회귀


```python
# weight
W = torch.zeros(1, requires_grad=True) 
# bias
b = torch.zeros(1, requires_grad=True)

hypothesis = x_train*W + b
```

`W`와 `b`를 0으로 초기화하고, 이를 학습 시키는 것이 목적  

학습을 위해 `requires_grad=True`로 학습할 것이라고 명시

### Compute loss
- Mean Squared Error (MSE)
  - $cost(W, b) = {1\over m} \sum^m_{i=1} (H(x^{(i)}) - y^{(i)})^2$  


```python
cost = torch.mean((hypothesis-y_train)**2)
```

### Gradient Descent
1. `zero_grad()`로 gradient 초기화
2. `backward()`로 gradient 계산
3. `step`으로 개선 


```python
optimizer = torch.optim.SGD([W, b], lr=0.01)

optimizer.zero_grad() # 1
cost.backward() # 2
optimizer.step() # 3
```

### Full Training Code


```python
# train data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# weight
W = torch.zeros(1, requires_grad=True) 
# bias
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs+1):
    hypothesis = x_train*W + b
    cost = torch.mean((hypothesis-y_train)**2)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%100==0: print(f"{cost.item():.3f}")
```

    0.048
    0.030
    0.018
    0.011
    0.007
    0.004
    0.003
    0.002
    0.001
    0.001
    

## Lab-03 Deeper Look at GD
- 가설 함수(Hypothesis Function)
- 평균 제곱 오차(MSE)
- 경사하강법(Gradient Descent)


```python
# Simpler Hypothesis
hypothesis = x_train * W
```


```python
# Dummpy Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
```

### Cost function: Intuition
- 실제값과 모델 값이 얼마나 다른지 나타내는 값 -> Cost
- 위 모델에서는, W=1일 때 cost=0

### Gradient Descent
- 목표는 cost 함수를 최소화하는 것 -> 미분 이용

$$
\nabla W = {\delta cost \over \delta W} = {{2 \over m} \sum^m_{i=1}(Wx^{(i)}-y^{(i)})x^{(i)}} \\
W : = W - \alpha \nabla W
$$



```python
# Full Code
# Dummpy Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1)
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = x_train*W
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train)**2)
    gradient = torch.sum((W*x_train - y_train)*x_train)
    
    print(f"{epoch:04d}/{nb_epochs} W: {W.item():.3f}, Cost: {cost.item():.6f}")
    
    # 개선
    W -= lr*gradient
```

    0000/10 W: 0.000, Cost: 4.666667
    0001/10 W: 1.400, Cost: 0.746666
    0002/10 W: 0.840, Cost: 0.119467
    0003/10 W: 1.064, Cost: 0.019115
    0004/10 W: 0.974, Cost: 0.003058
    0005/10 W: 1.010, Cost: 0.000489
    0006/10 W: 0.996, Cost: 0.000078
    0007/10 W: 1.002, Cost: 0.000013
    0008/10 W: 0.999, Cost: 0.000002
    0009/10 W: 1.000, Cost: 0.000000
    0010/10 W: 1.000, Cost: 0.000000
    

`torch.optim`으로도 gradient descent를 할 수 있음
- Optimizer 정의
- `optimizer.zero_grad()`로 gradient를 0으로 초기화
- `cost.backward()`로 gradient 계산
- `optimizer.step()`으로 gradient descent


```python
# optimizer 설정
optimizer = torch.optim.SGD([W], lr=0.15)
# cost로 H(x) 개선
optimizer.zero_grad()
cost.backward()
optimizer.step()
```


```python
from torch import optim

# Full Code
# Dummpy Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
# 모델 초기화
W = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W], lr=0.15)

nb_epochs = 10
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = x_train*W
    # cost gradient 계산
    cost = torch.mean((hypothesis - y_train)**2)
    
    print(f"{epoch:04d}/{nb_epochs} W: {W.item():.3f}, Cost: {cost.item():.6f}")
    
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

    0000/10 W: 0.000, Cost: 4.666667
    0001/10 W: 1.400, Cost: 0.746667
    0002/10 W: 0.840, Cost: 0.119467
    0003/10 W: 1.064, Cost: 0.019115
    0004/10 W: 0.974, Cost: 0.003058
    0005/10 W: 1.010, Cost: 0.000489
    0006/10 W: 0.996, Cost: 0.000078
    0007/10 W: 1.002, Cost: 0.000013
    0008/10 W: 0.999, Cost: 0.000002
    0009/10 W: 1.000, Cost: 0.000000
    0010/10 W: 1.000, Cost: 0.000000
    

## Lab-04-1 Multivariable Linear regression
- 다항 선형 회귀(Multivariable Linear regression)
- 가설 함수(Hypothesis Function)
- 평균 제곱 오차(MSE)
- 경사하강법(Gradient descent)


```python
# Data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = x_train@W # matmul
    # cost
    cost = torch.mean((hypothesis - y_train)**2)
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch:4d}/{nb_epochs} hypothesis: {hypothesis.squeeze().detach()}, Cost: {cost.item():.6f}")
```

    Epoch:    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]), Cost: 29661.800781
    Epoch:    1/20 hypothesis: tensor([67.2544, 80.8363, 79.6489, 86.7360, 61.6571]), Cost: 9299.174805
    Epoch:    2/20 hypothesis: tensor([104.9088, 126.0952, 124.2429, 135.2979,  96.1780]), Cost: 2916.123535
    Epoch:    3/20 hypothesis: tensor([125.9906, 151.4351, 149.2102, 162.4868, 115.5059]), Cost: 915.234863
    Epoch:    4/20 hypothesis: tensor([137.7938, 165.6226, 163.1889, 177.7094, 126.3274]), Cost: 288.017517
    Epoch:    5/20 hypothesis: tensor([144.4020, 173.5660, 171.0154, 186.2322, 132.3864]), Cost: 91.404106
    Epoch:    6/20 hypothesis: tensor([148.1017, 178.0136, 175.3972, 191.0040, 135.7788]), Cost: 29.771265
    Epoch:    7/20 hypothesis: tensor([150.1728, 180.5038, 177.8504, 193.6755, 137.6784]), Cost: 10.450775
    Epoch:    8/20 hypothesis: tensor([151.3322, 181.8982, 179.2239, 195.1712, 138.7421]), Cost: 4.393974
    Epoch:    9/20 hypothesis: tensor([151.9812, 182.6790, 179.9928, 196.0086, 139.3379]), Cost: 2.494854
    Epoch:   10/20 hypothesis: tensor([152.3443, 183.1163, 180.4233, 196.4774, 139.6716]), Cost: 1.899049
    Epoch:   11/20 hypothesis: tensor([152.5475, 183.3613, 180.6642, 196.7398, 139.8586]), Cost: 1.711788
    Epoch:   12/20 hypothesis: tensor([152.6610, 183.4986, 180.7991, 196.8867, 139.9635]), Cost: 1.652612
    Epoch:   13/20 hypothesis: tensor([152.7244, 183.5756, 180.8745, 196.9688, 140.0224]), Cost: 1.633556
    Epoch:   14/20 hypothesis: tensor([152.7597, 183.6188, 180.9167, 197.0148, 140.0556]), Cost: 1.627118
    Epoch:   15/20 hypothesis: tensor([152.7792, 183.6432, 180.9402, 197.0405, 140.0744]), Cost: 1.624588
    Epoch:   16/20 hypothesis: tensor([152.7900, 183.6569, 180.9534, 197.0548, 140.0850]), Cost: 1.623325
    Epoch:   17/20 hypothesis: tensor([152.7958, 183.6647, 180.9606, 197.0628, 140.0912]), Cost: 1.622444
    Epoch:   18/20 hypothesis: tensor([152.7989, 183.6693, 180.9647, 197.0672, 140.0948]), Cost: 1.621667
    Epoch:   19/20 hypothesis: tensor([152.8004, 183.6719, 180.9669, 197.0696, 140.0970]), Cost: 1.620959
    Epoch:   20/20 hypothesis: tensor([152.8011, 183.6736, 180.9680, 197.0709, 140.0984]), Cost: 1.620234
    

### nn.Module


```python
import torch.nn as nn

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 입력 차원, 출력 차원
        
    def forward(self, x): # hypothesis 계산
        return self.linear(x)

    # gradient 계산은 토치가 알아서 해줌 (backward())

model = MultivariateLinearRegressionModel()    
hypothesis = model(x_train)
```

### F
- 다양한 코스트 함수를 사용 가능


```python
import torch.nn.functional as F

# cost = F.mse_loss(pred, y_train)
```


```python
# Full Code
# Data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델 초기화
model = MultivariateLinearRegressionModel()

# optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = model(x_train)
    # cost 계산
    cost = F.mse_loss(hypothesis, y_train)
    
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch:4d}/{nb_epochs} hypothesis: {hypothesis.squeeze().detach()}, Cost: {cost.item():.6f}")
```

    Epoch:    0/20 hypothesis: tensor([-32.4972, -42.8243, -40.2378, -44.5123, -32.9395]), Cost: 44579.566406
    Epoch:    1/20 hypothesis: tensor([49.9533, 56.2768, 57.4073, 61.8209, 42.6500]), Cost: 13977.326172
    Epoch:    2/20 hypothesis: tensor([ 96.1139, 111.7601, 112.0750, 121.3528,  84.9702]), Cost: 4385.147461
    Epoch:    3/20 hypothesis: tensor([121.9571, 142.8235, 142.6814, 154.6825, 108.6641]), Cost: 1378.507080
    Epoch:    4/20 hypothesis: tensor([136.4254, 160.2150, 159.8166, 173.3424, 121.9298]), Cost: 436.082703
    Epoch:    5/20 hypothesis: tensor([144.5252, 169.9521, 169.4099, 183.7894, 129.3571]), Cost: 140.680481
    Epoch:    6/20 hypothesis: tensor([149.0596, 175.4039, 174.7807, 189.6381, 133.5158]), Cost: 48.085285
    Epoch:    7/20 hypothesis: tensor([151.5979, 178.4564, 177.7875, 192.9126, 135.8444]), Cost: 19.059713
    Epoch:    8/20 hypothesis: tensor([153.0185, 180.1657, 179.4708, 194.7457, 137.1485]), Cost: 9.959645
    Epoch:    9/20 hypothesis: tensor([153.8135, 181.1229, 180.4131, 195.7719, 137.8790]), Cost: 7.105104
    Epoch:   10/20 hypothesis: tensor([154.2581, 181.6591, 180.9405, 196.3463, 138.2883]), Cost: 6.208321
    Epoch:   11/20 hypothesis: tensor([154.5067, 181.9596, 181.2356, 196.6678, 138.5179]), Cost: 5.925124
    Epoch:   12/20 hypothesis: tensor([154.6455, 182.1281, 181.4008, 196.8478, 138.6468]), Cost: 5.834256
    Epoch:   13/20 hypothesis: tensor([154.7227, 182.2227, 181.4931, 196.9484, 138.7193]), Cost: 5.803680
    Epoch:   14/20 hypothesis: tensor([154.7656, 182.2760, 181.5446, 197.0046, 138.7603]), Cost: 5.792007
    Epoch:   15/20 hypothesis: tensor([154.7892, 182.3060, 181.5734, 197.0360, 138.7836]), Cost: 5.786251
    Epoch:   16/20 hypothesis: tensor([154.8020, 182.3232, 181.5894, 197.0535, 138.7970]), Cost: 5.782393
    Epoch:   17/20 hypothesis: tensor([154.8088, 182.3330, 181.5982, 197.0632, 138.8049]), Cost: 5.779080
    Epoch:   18/20 hypothesis: tensor([154.8121, 182.3388, 181.6030, 197.0686, 138.8096]), Cost: 5.775957
    Epoch:   19/20 hypothesis: tensor([154.8136, 182.3424, 181.6056, 197.0715, 138.8127]), Cost: 5.772896
    Epoch:   20/20 hypothesis: tensor([154.8141, 182.3446, 181.6069, 197.0730, 138.8147]), Cost: 5.769830
    

## Lab-04-2 Loading Data
- 다항 선형 회귀 (Multivariable Linear regression)
- 미니배치 경사하강법(Minibatch Gradient descnet)
- Dataset, DataLoader

### Minibatch Gradient Descent
- 전체 데이터를 균일하게 나눠서 학습
- 각 미니 배치의 코스트를 구하고 GD 수행
  - 업데이트가 빠름
  - 전체 데이터를 쓰지 않아서 잘못된 방향으로 업데이트를 할 수도 있음

### PyTorch Dataset
- 다음 메서드들은 필수로 구현되어야 함
  - `__len__()` : 데이터셋의 총 데이터 수
  - `__getitem__()` : 어떤 인덱스를 받았을 때, 그에 상응하는 입출력 데이터 반환


```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y
    
dataset = CustomDataset()
```

### PyTorch DataLoader
- 각 미니 배치의 크기를 지정해줘야 하고, 통상적으로 2의 제곱수를 이용


```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```


```python
# Full Code
nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        # H(x) 계산
        pred = model(x_train)
        # cost 계산
        cost = F.mse_loss(pred, y_train)
        # 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:4d}/{nb_epochs} Batch {batch_idx+1}/{len(dataloader)} Cost: {cost.item():.6f}")
```

    Epoch    0/20 Batch 1/3 Cost: 1.868474
    Epoch    0/20 Batch 2/3 Cost: 12.439695
    Epoch    0/20 Batch 3/3 Cost: 12.434620
    Epoch    1/20 Batch 1/3 Cost: 2.737696
    Epoch    1/20 Batch 2/3 Cost: 15.813776
    Epoch    1/20 Batch 3/3 Cost: 4.588157
    Epoch    2/20 Batch 1/3 Cost: 5.603372
    Epoch    2/20 Batch 2/3 Cost: 4.540079
    Epoch    2/20 Batch 3/3 Cost: 14.736433
    Epoch    3/20 Batch 1/3 Cost: 9.144573
    Epoch    3/20 Batch 2/3 Cost: 4.510876
    Epoch    3/20 Batch 3/3 Cost: 2.529641
    Epoch    4/20 Batch 1/3 Cost: 7.655476
    Epoch    4/20 Batch 2/3 Cost: 1.510289
    Epoch    4/20 Batch 3/3 Cost: 13.989696
    Epoch    5/20 Batch 1/3 Cost: 7.586789
    Epoch    5/20 Batch 2/3 Cost: 3.379308
    Epoch    5/20 Batch 3/3 Cost: 12.388711
    Epoch    6/20 Batch 1/3 Cost: 6.161662
    Epoch    6/20 Batch 2/3 Cost: 7.593049
    Epoch    6/20 Batch 3/3 Cost: 9.640903
    Epoch    7/20 Batch 1/3 Cost: 3.702758
    Epoch    7/20 Batch 2/3 Cost: 9.231374
    Epoch    7/20 Batch 3/3 Cost: 12.365843
    Epoch    8/20 Batch 1/3 Cost: 3.565463
    Epoch    8/20 Batch 2/3 Cost: 9.194556
    Epoch    8/20 Batch 3/3 Cost: 4.596266
    Epoch    9/20 Batch 1/3 Cost: 9.076134
    Epoch    9/20 Batch 2/3 Cost: 5.187035
    Epoch    9/20 Batch 3/3 Cost: 1.384005
    Epoch   10/20 Batch 1/3 Cost: 2.869235
    Epoch   10/20 Batch 2/3 Cost: 7.266636
    Epoch   10/20 Batch 3/3 Cost: 11.544607
    Epoch   11/20 Batch 1/3 Cost: 8.700336
    Epoch   11/20 Batch 2/3 Cost: 6.612643
    Epoch   11/20 Batch 3/3 Cost: 6.908180
    Epoch   12/20 Batch 1/3 Cost: 3.930895
    Epoch   12/20 Batch 2/3 Cost: 9.338434
    Epoch   12/20 Batch 3/3 Cost: 5.180389
    Epoch   13/20 Batch 1/3 Cost: 5.364339
    Epoch   13/20 Batch 2/3 Cost: 8.799165
    Epoch   13/20 Batch 3/3 Cost: 1.258289
    Epoch   14/20 Batch 1/3 Cost: 6.465598
    Epoch   14/20 Batch 2/3 Cost: 5.129382
    Epoch   14/20 Batch 3/3 Cost: 12.905280
    Epoch   15/20 Batch 1/3 Cost: 6.058489
    Epoch   15/20 Batch 2/3 Cost: 3.494545
    Epoch   15/20 Batch 3/3 Cost: 11.230926
    Epoch   16/20 Batch 1/3 Cost: 7.892144
    Epoch   16/20 Batch 2/3 Cost: 5.982709
    Epoch   16/20 Batch 3/3 Cost: 2.968175
    Epoch   17/20 Batch 1/3 Cost: 12.357420
    Epoch   17/20 Batch 2/3 Cost: 8.819189
    Epoch   17/20 Batch 3/3 Cost: 0.288896
    Epoch   18/20 Batch 1/3 Cost: 6.975581
    Epoch   18/20 Batch 2/3 Cost: 7.287281
    Epoch   18/20 Batch 3/3 Cost: 2.553432
    Epoch   19/20 Batch 1/3 Cost: 6.105959
    Epoch   19/20 Batch 2/3 Cost: 3.924425
    Epoch   19/20 Batch 3/3 Cost: 14.760810
    Epoch   20/20 Batch 1/3 Cost: 3.494778
    Epoch   20/20 Batch 2/3 Cost: 7.753602
    Epoch   20/20 Batch 3/3 Cost: 12.892018
    

## Lab-05 Logistic Regression
- 로지스틱 회귀(Logistic Regression)
- 가설(Hypothesis)
- 손실함수(Cost Function)
- 평가(Evaluation)

### Logistic Regression
- Hypothesis $$H(x) = {1 \over 1 + e^{-W^TX}}$$
- Cost $$cost(W) = -{1 \over m} \sum ylog(H(x)) + (1-y)(log(1-H(x)))$$
- $H(x) = P(x=1;W) = 1 - P(x=0;W)$
- Weight Update via Gradient Descent
  - $W \ := W - \alpha {\delta \over \delta W} cost(W) \ = W - \alpha \nabla_W cost(W)$


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```




    <torch._C.Generator at 0x1cd30089650>




```python
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)
```

    torch.Size([6, 2])
    torch.Size([6, 1])
    

### Computing the Hypothesis


```python
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```


```python
# hypothesis = 1 / (1 + torch.exp(-(x_train@W +b)))
hypothesis = torch.sigmoid(x_train@W+b)
```


```python
print(hypothesis)
print(hypothesis.shape)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<SigmoidBackward0>)
    torch.Size([6, 1])
    

### Computing the Cost Function


```python
print(hypothesis)
print(y_train)
```

    tensor([[0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000],
            [0.5000]], grad_fn=<SigmoidBackward0>)
    tensor([[0.],
            [0.],
            [0.],
            [1.],
            [1.],
            [1.]])
    


```python
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
```

    tensor([[0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931],
            [0.6931]], grad_fn=<NegBackward0>)
    


```python
cost = losses.mean()
print(cost)
```

    tensor(0.6931, grad_fn=<MeanBackward0>)
    


```python
F.binary_cross_entropy(hypothesis, y_train)
```




    tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward0>)




```python
# Train
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # Cost 계산
    hypothesis = torch.sigmoid(x_train@W+b)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%100==0:
        print(f"Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item()}")
```

    Epoch    0/1000 Cost: 0.6931471824645996
    Epoch  100/1000 Cost: 0.134722039103508
    Epoch  200/1000 Cost: 0.08064313977956772
    Epoch  300/1000 Cost: 0.05790001526474953
    Epoch  400/1000 Cost: 0.0452997200191021
    Epoch  500/1000 Cost: 0.037260960787534714
    Epoch  600/1000 Cost: 0.03167249262332916
    Epoch  700/1000 Cost: 0.027555925771594048
    Epoch  800/1000 Cost: 0.024394338950514793
    Epoch  900/1000 Cost: 0.021888310089707375
    Epoch 1000/1000 Cost: 0.019852152094244957
    

### Evaluation


```python
hypothesis = torch.sigmoid(x_train@W+b)
print(hypothesis[:5])
```

    tensor([[2.7648e-04],
            [3.1608e-02],
            [3.8977e-02],
            [9.5622e-01],
            [9.9823e-01]], grad_fn=<SliceBackward0>)
    


```python
pred = hypothesis >= torch.FloatTensor([0.5])
print(pred[:5])
```

    tensor([[False],
            [False],
            [False],
            [ True],
            [ True]])
    


```python
pred[:5].float()
```




    tensor([[0.],
            [0.],
            [0.],
            [1.],
            [1.]])




```python
print(y_train[:5])
```

    tensor([[0.],
            [0.],
            [0.],
            [1.],
            [1.]])
    


```python
pred[:5].float() == y_train[:5]
```




    tensor([[True],
            [True],
            [True],
            [True],
            [True]])



### Higher Implementation with Class


```python
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))
```


```python
model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 100
for epoch in range(nb_epochs+1):
    # Cost 계산
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%10==0:
        pred = hypothesis >= torch.FloatTensor([0.5])
        correct_pred = pred.float()==y_train
        acc = correct_pred.sum().item() / len(correct_pred)
        print(f"Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f} ACC: {acc:2.2f}%")
```

    Epoch    0/100 Cost: 1.958243 ACC: 0.50%
    Epoch   10/100 Cost: 0.673601 ACC: 0.50%
    Epoch   20/100 Cost: 0.501478 ACC: 0.67%
    Epoch   30/100 Cost: 0.423174 ACC: 0.67%
    Epoch   40/100 Cost: 0.359201 ACC: 0.83%
    Epoch   50/100 Cost: 0.304263 ACC: 0.83%
    Epoch   60/100 Cost: 0.255134 ACC: 0.83%
    Epoch   70/100 Cost: 0.210761 ACC: 1.00%
    Epoch   80/100 Cost: 0.174970 ACC: 1.00%
    Epoch   90/100 Cost: 0.153584 ACC: 1.00%
    Epoch  100/100 Cost: 0.141670 ACC: 1.00%
    

## Lab-06 Softmax Classification
- 소프트맥스(Softmax)
- 크로스 엔트로피(Cross Entropy)

### Discrete Probability Distribution
- 이산 확률 분포

### Softmax
$$P(class=i) = {e^i \over \sum e^i}$$


```python
z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
```

    tensor([0.0900, 0.2447, 0.6652])
    


```python
hypothesis.sum()
```




    tensor(1.)



### Cross Entropy
- 주어진 2개의 확률 분포가 얼마나 동일한지를 나타내는 수치


```python
# Cross Entropy Loss (Low-level)
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
```

    tensor([[0.1706, 0.2296, 0.1470, 0.3028, 0.1500],
            [0.2751, 0.1902, 0.1453, 0.2445, 0.1450],
            [0.2647, 0.1978, 0.2163, 0.1413, 0.1800]], grad_fn=<SoftmaxBackward0>)
    


```python
y = torch.randint(5, (3, )).long()
print(y)
```

    tensor([3, 0, 1])
    


```python
# One-Hot
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
```




    tensor([[0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.]])




```python
cost = (y_one_hot*-torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
```

    tensor(1.3687, grad_fn=<MeanBackward0>)
    


```python
torch.log(F.softmax(z, dim=1))
```




    tensor([[-1.7686, -1.4713, -1.9176, -1.1947, -1.8968],
            [-1.2907, -1.6598, -1.9291, -1.4087, -1.9311],
            [-1.3293, -1.6207, -1.5312, -1.9570, -1.7147]], grad_fn=<LogBackward0>)




```python
torch.log_softmax(z, dim=1)
```




    tensor([[-1.7686, -1.4713, -1.9176, -1.1947, -1.8968],
            [-1.2907, -1.6598, -1.9291, -1.4087, -1.9311],
            [-1.3293, -1.6207, -1.5312, -1.9570, -1.7147]],
           grad_fn=<LogSoftmaxBackward0>)




```python
F.nll_loss(F.log_softmax(z, dim=1), y)
```




    tensor(1.3687, grad_fn=<NllLossBackward0>)




```python
# Cross Entropy = log_softmax + nll_loss
F.cross_entropy(z, y)
```




    tensor(1.3687, grad_fn=<NllLossBackward0>)




```python
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
```


```python
# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # Cost 계산 1
    # hypothesis = F.softmax(x_train@W+b, dim=1)
    # y_one_hot = torch.zeros_like(hypothesis)
    # y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    # cost = (y_one_hot*-torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
    
    # Cost 계산 2
    z = x_train@W+b
    cost = F.cross_entropy(z, y_train)
    
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%100==0:
        print(f"Epoch: {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}")

```

    Epoch:    0/1000 Cost: 1.098612
    Epoch:  100/1000 Cost: 0.761050
    Epoch:  200/1000 Cost: 0.689991
    Epoch:  300/1000 Cost: 0.643229
    Epoch:  400/1000 Cost: 0.604117
    Epoch:  500/1000 Cost: 0.568256
    Epoch:  600/1000 Cost: 0.533922
    Epoch:  700/1000 Cost: 0.500291
    Epoch:  800/1000 Cost: 0.466908
    Epoch:  900/1000 Cost: 0.433507
    Epoch: 1000/1000 Cost: 0.399962
    


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    pred = model(x_train)
    # cost 계산
    cost = F.cross_entropy(pred, y_train)
    # 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch%100==0:
        print(f"Epoch: {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}")
```

    Epoch:    0/1000 Cost: 1.263907
    Epoch:  100/1000 Cost: 0.673919
    Epoch:  200/1000 Cost: 0.586287
    Epoch:  300/1000 Cost: 0.530028
    Epoch:  400/1000 Cost: 0.484597
    Epoch:  500/1000 Cost: 0.444543
    Epoch:  600/1000 Cost: 0.407506
    Epoch:  700/1000 Cost: 0.372013
    Epoch:  800/1000 Cost: 0.336894
    Epoch:  900/1000 Cost: 0.301193
    Epoch: 1000/1000 Cost: 0.265083
    

## Lab-07-1 Tips
- 최대 가능도 추정(Maximum Likehood Estimation)
- 과적합(Overfitting)과 정규화(Regurlarization)
- 훈련 세트와 테스트 세트
- 학습률(Learning Rate)
- 데이터 전처리(Preprocessing)

### Maximum Likelihood Estimation (MLE)
- 최대 가능도 추정

$$
K \sim B(n, \theta)
\\
\begin{align}
P(K=k) = \begin{pmatrix} n \\ k \end{pmatrix} \theta^k (1-\theta)^{n-k}
\\
{n! \over k!(n-k)!} \theta^k(1-\theta)^{n-k}
\end{align}
$$

- Obseravation: n=100, k=27
- $\theta$가 궁금한것임 (y가 최대가 되는곳)
  - $\theta$ = 0.27
  - 관측값을 가장 잘 설명하는 확률 분포 함수의 파라미터 ($\theta$ 찾기)
- 기울기를 통해 찾을 수 있음 (Gradient Descent/Ascent)
- Descent -> Local Minimum
- Ascent -> Local Maximum

### Overfitting
- 최소화하는 것이 중요
- Train / Validation(Devlopment) / Test 이용
- More Data, Less Features, Regularization으로 해결 가능

#### Regularization
- Early Stopping
- Reducing Network Size
- Weight Decay
- Dropout
- Batch Normalization


```python
x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

x_test = torch.FloatTensor([[2, 1, 1],
                            [3, 1, 2],
                            [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
```


```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()
```


```python
optimizer = optim.SGD(model.parameters(), lr=0.1)

def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        pred = model(x_train)
        cost = F.cross_entropy(pred, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}")
        
def test(model, optimizer, x_test, y_test):
    pred = model(x_test)
    pred_classes = pred.max(1)[1]
    correct_count = (pred_classes==y_test).sum().item()
    cost = F.cross_entropy(pred, y_test)
    
    print(f"Acc: {correct_count/len(y_test)*100}% Cost: {cost.item():.6f}")
    

```


```python
train(model, optimizer, x_train, y_train)
```

    Epoch:    0/20 Cost: 1.374710
    Epoch:    1/20 Cost: 1.203222
    Epoch:    2/20 Cost: 1.148050
    Epoch:    3/20 Cost: 1.134803
    Epoch:    4/20 Cost: 1.123535
    Epoch:    5/20 Cost: 1.113072
    Epoch:    6/20 Cost: 1.103134
    Epoch:    7/20 Cost: 1.093612
    Epoch:    8/20 Cost: 1.084436
    Epoch:    9/20 Cost: 1.075567
    Epoch:   10/20 Cost: 1.066976
    Epoch:   11/20 Cost: 1.058643
    Epoch:   12/20 Cost: 1.050551
    Epoch:   13/20 Cost: 1.042686
    Epoch:   14/20 Cost: 1.035039
    Epoch:   15/20 Cost: 1.027598
    Epoch:   16/20 Cost: 1.020354
    Epoch:   17/20 Cost: 1.013297
    Epoch:   18/20 Cost: 1.006421
    Epoch:   19/20 Cost: 0.999717
    


```python
test(model, optimizer, x_test, y_test)
```

    Acc: 100.0% Cost: 0.791553
    

### Learning Rate
- 너무 크면 발산하면서 cost가 늘어남(overshooting)


```python
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e5)

train(model, optimizer, x_train, y_train)
```

    Epoch:    0/20 Cost: 2.108304
    Epoch:    1/20 Cost: 1097902.000000
    Epoch:    2/20 Cost: 2111920.000000
    Epoch:    3/20 Cost: 661964.562500
    Epoch:    4/20 Cost: 1297857.500000
    Epoch:    5/20 Cost: 1356673.000000
    Epoch:    6/20 Cost: 1351027.000000
    Epoch:    7/20 Cost: 1419732.625000
    Epoch:    8/20 Cost: 915089.562500
    Epoch:    9/20 Cost: 873860.625000
    Epoch:   10/20 Cost: 1541607.500000
    Epoch:   11/20 Cost: 1604151.875000
    Epoch:   12/20 Cost: 727544.937500
    Epoch:   13/20 Cost: 1185001.250000
    Epoch:   14/20 Cost: 1106673.125000
    Epoch:   15/20 Cost: 818169.937500
    Epoch:   16/20 Cost: 2015089.500000
    Epoch:   17/20 Cost: 130845.273438
    Epoch:   18/20 Cost: 781876.250000
    Epoch:   19/20 Cost: 833235.625000
    

- 너무 작으면 cost가 거의 줄어들지 않음


```python
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-10)

train(model, optimizer, x_train, y_train)
```

    Epoch:    0/20 Cost: 1.513814
    Epoch:    1/20 Cost: 1.513814
    Epoch:    2/20 Cost: 1.513814
    Epoch:    3/20 Cost: 1.513814
    Epoch:    4/20 Cost: 1.513814
    Epoch:    5/20 Cost: 1.513814
    Epoch:    6/20 Cost: 1.513814
    Epoch:    7/20 Cost: 1.513814
    Epoch:    8/20 Cost: 1.513814
    Epoch:    9/20 Cost: 1.513814
    Epoch:   10/20 Cost: 1.513814
    Epoch:   11/20 Cost: 1.513814
    Epoch:   12/20 Cost: 1.513814
    Epoch:   13/20 Cost: 1.513814
    Epoch:   14/20 Cost: 1.513814
    Epoch:   15/20 Cost: 1.513814
    Epoch:   16/20 Cost: 1.513814
    Epoch:   17/20 Cost: 1.513814
    Epoch:   18/20 Cost: 1.513814
    Epoch:   19/20 Cost: 1.513814
    

- 적절한 숫자로 시작해, 발산하면 작게, 줄어들지 않으면 크게 조정할 필요가 있음


```python
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

train(model, optimizer, x_train, y_train)
```

    Epoch:    0/20 Cost: 2.779414
    Epoch:    1/20 Cost: 1.193390
    Epoch:    2/20 Cost: 1.033265
    Epoch:    3/20 Cost: 0.992758
    Epoch:    4/20 Cost: 0.972408
    Epoch:    5/20 Cost: 0.957914
    Epoch:    6/20 Cost: 0.945693
    Epoch:    7/20 Cost: 0.935023
    Epoch:    8/20 Cost: 0.925264
    Epoch:    9/20 Cost: 0.916228
    Epoch:   10/20 Cost: 0.907732
    Epoch:   11/20 Cost: 0.899703
    Epoch:   12/20 Cost: 0.892071
    Epoch:   13/20 Cost: 0.884799
    Epoch:   14/20 Cost: 0.877851
    Epoch:   15/20 Cost: 0.871207
    Epoch:   16/20 Cost: 0.864842
    Epoch:   17/20 Cost: 0.858740
    Epoch:   18/20 Cost: 0.852885
    Epoch:   19/20 Cost: 0.847262
    

### Data Preprocessing


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [74, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```


```python
# Standardization -> 정규 분포화
mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma
```


```python
print(norm_x_train)
```

    tensor([[-1.1118, -0.3758, -0.8398],
            [ 0.7412,  0.2778,  0.5863],
            [ 0.3706,  0.5229,  0.3486],
            [ 1.0191,  1.0948,  1.1409],
            [-1.0191, -1.5197, -1.2360]])
    


```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)
    
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-1)
```


```python
def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epoch in range(nb_epochs):
        pred = model(x_train)
        cost = F.mse_loss(pred, y_train)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:4d}/{nb_epochs} Cost: {cost.item():.6f}")
        

train(model, optimizer, x_train, y_train)
```

    Epoch    0/20 Cost: 5596.600098
    Epoch    1/20 Cost: 108613935104.000000
    Epoch    2/20 Cost: 2108445212777906176.000000
    Epoch    3/20 Cost: 40929755654583232237666304.000000
    Epoch    4/20 Cost: 794540324115572007583260136701952.000000
    Epoch    5/20 Cost: inf
    Epoch    6/20 Cost: inf
    Epoch    7/20 Cost: inf
    Epoch    8/20 Cost: inf
    Epoch    9/20 Cost: inf
    Epoch   10/20 Cost: inf
    Epoch   11/20 Cost: inf
    Epoch   12/20 Cost: nan
    Epoch   13/20 Cost: nan
    Epoch   14/20 Cost: nan
    Epoch   15/20 Cost: nan
    Epoch   16/20 Cost: nan
    Epoch   17/20 Cost: nan
    Epoch   18/20 Cost: nan
    Epoch   19/20 Cost: nan
    


```python
model = MultivariateLinearRegressionModel()

optimizer = optim.SGD(model.parameters(), lr=1e-1)

train(model, optimizer, norm_x_train, y_train)
```

    Epoch    0/20 Cost: 29726.589844
    Epoch    1/20 Cost: 18865.031250
    Epoch    2/20 Cost: 12026.861328
    Epoch    3/20 Cost: 7683.623535
    Epoch    4/20 Cost: 4913.682129
    Epoch    5/20 Cost: 3143.766846
    Epoch    6/20 Cost: 2011.853516
    Epoch    7/20 Cost: 1287.668335
    Epoch    8/20 Cost: 824.257935
    Epoch    9/20 Cost: 527.691528
    Epoch   10/20 Cost: 337.890778
    Epoch   11/20 Cost: 216.416229
    Epoch   12/20 Cost: 138.669388
    Epoch   13/20 Cost: 88.907555
    Epoch   14/20 Cost: 57.056488
    Epoch   15/20 Cost: 36.668449
    Epoch   16/20 Cost: 23.616732
    Epoch   17/20 Cost: 15.260473
    Epoch   18/20 Cost: 9.909460
    Epoch   19/20 Cost: 6.481894
    

전처리를 통해 모든 데이터를 보게 만듬

## Lab-07-2 MNIST Introduction
- MNIST
- torchvision
- Epoch
- Batch size
- Iteration

### torchvision
- 데이터셋과 모델 아키텍쳐, 전처리 등을 제공하는 패키지


```python
import torchvision.datasets as dsets
from torchvision import transforms

...
mnist_train = dsets.MNIST(root="./", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)
...
data_loader = torch.utils.DataLoader(Dataloader=mnist_train, batch_size=64, shuffle=True, drop_lats=True)
...
```

### torch.no_grad()
- 학습하지 않겠다라는 의미


```python
# Test
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)
    
    prd = linear(X_test)
    correct_pred = torch.argmax(pred, 1)==Y_test
    acc = correct_pred.float().mean()
    print(f"ACC: {acc.item()}")
```
