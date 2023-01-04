---
title: "[PyTorch] Gradeint Descent"
date: 2022-08-01T08:57:14.857Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Graident Descent
본질적으로, PyTorch에는 두가지 주요한 특성이 있음
- NumPy와 유사하지만 GPU 상에서 실행 가능한 n-차원 텐서(Tensor)
- 신경망을 구성하고 학습하는 과정에서의 자동 미분(Automatoc differentiation)

## sin(x) 근사하기
3차 다항식을 사용해, $y=sin(x)$에 근사(fit)하는 문제 다루기  
신경망은 4개의 매개변수를 가지며, 정답과 신경망이 예측한 결과 사이의 유클리드 거리(Euclidean distance)를 최소화하여 임의의 값을 근사할 수 있도록 **경사 하강법(gradient descent)** 을 사용하여 학습 

NumPy는 연산 그래프(computation graph)나 딥러닝, 변화도(gradient)에 대해서는 알지 못합니다. 하지만 NumPy 연산을 사용하여 신경망의 순전파 단계와 역전파 단계를 직접 구현함으로써, 3차 다항식이 사인(sine) 함수에 근사하도록 만들 수 있음

### NumPy


```python
import numpy as np
import math
# 무작위 입/출력 데이터 생성
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 무작위로 가중치 초기화
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    # 손실(loss)을 계산하고 출력
    loss = np.square(y_pred-y).sum()
    if t%100==99: print(t, loss)
    # 손실에 따른 a, b, c, d의 변화도(grdient)를 계산하고 역전파
    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()
    # 가중치 갱신
    a -= learning_rate*grad_a
    b -= learning_rate*grad_b
    c -= learning_rate*grad_c
    d -= learning_rate*grad_d
    
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')   
```

    99 3005.093739696107
    199 2053.496881150398
    299 1405.7224600064478
    399 964.2667462685014
    499 663.0733590447679
    599 457.3416379543572
    699 316.6539523622675
    799 220.3353372303261
    899 154.31726097340132
    999 109.01581916340615
    1099 77.89474199556972
    1199 56.49115946165688
    1299 41.75438824008516
    1399 31.596674177440697
    1499 24.5876074303758
    1599 19.746029990619565
    1699 16.39817128300095
    1799 14.080823421353184
    1899 12.475180918891025
    1999 11.361581406294208
    Result: y = 0.04631577819034255 + 0.8323458176324102 x + -0.00799024243570984 x^2 + -0.08986040910531393 x^3
    

### PyTorch


```python
import torch

dtype = torch.float
device = torch.device("cpu")

# 무작위로 입/출력 데이터를 생성
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
# 무작위로 가중치를 초기화
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    # 손실(loss)을 계산하고 출력
    loss = np.square(y_pred-y).sum()
    if t%100==99: print(t, loss)
    # 손실에 따른 a, b, c, d의 변화도(grdient)를 계산하고 역전파
    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred*x).sum()
    grad_c = (grad_y_pred*x**2).sum()
    grad_d = (grad_y_pred*x**3).sum()
    # 가중치 갱신
    a -= learning_rate*grad_a
    b -= learning_rate*grad_b
    c -= learning_rate*grad_c
    d -= learning_rate*grad_d
    
print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')   
```

    99 tensor(1712.0419)
    199 tensor(1177.8615)
    299 tensor(812.1266)
    399 tensor(561.4324)
    499 tensor(389.3970)
    599 tensor(271.2057)
    699 tensor(189.9148)
    799 tensor(133.9413)
    899 tensor(95.3582)
    999 tensor(68.7333)
    1099 tensor(50.3409)
    1199 tensor(37.6223)
    1299 tensor(28.8181)
    1399 tensor(22.7176)
    1499 tensor(18.4864)
    1599 tensor(15.5489)
    1699 tensor(13.5076)
    1799 tensor(12.0879)
    1899 tensor(11.0997)
    1999 tensor(10.4112)
    Result: y = 0.03805118799209595 + 0.8736276626586914 x + -0.006564462557435036 x^2 + -0.09573239833116531 x^3
    

## 새 autograd Function 정의
autograd의 기본 연산자는, 텐서를 조작하는 2개의 함수  
1. `forward` 함수는 입력 텐서로부터 출력 텐서를 계산
2. `backward` 함수는 어떤 스칼라 값에 대한 출력 텐서의 변화도(gradient)를 전달받고, 동일한 스칼라 값에 대한 입력 텐서의 변화도를 계산

`torch.autograd.Function`의 하위클래스(subclass)를 정의하고 `forward`와 `backward` 함수를 구현함으로써 사용자 정의 autograd 연산자를 손쉽게 정의할 수 있음  

$P_3(x) = {1 \over 2}(5x^3 - 3x)$ 3차 르장드르 다항식(Legendre Polynomial)


```python
from turtle import forward


class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*1.5*(5*input**2-1)

dtype = torch.float
device = torch.device("cpu")
# 입/출력 텐서 생성 - 추척은 하지 않음
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)
# 임의의 가중치
a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    # 사용자 정의 Function을 적용하기 위해 Function.apply 메소드를 사용합니다.
    # 여기에 'P3'라고 이름을 붙였습니다.
    P3 = LegendrePolynomial3.apply

    # 순전파 단계: 연산을 하여 예측값 y를 계산합니다;
    # 사용자 정의 autograd 연산을 사용하여 P3를 계산합니다.
    y_pred = a + b * P3(c + d * x)

    # 손실을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograd를 사용하여 역전파 단계를 계산합니다.
    loss.backward()

    # 경사하강법(gradient descent)을 사용하여 가중치를 갱신합니다.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 가중치 갱신 후에는 변화도를 직접 0으로 만듭니다.
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
```

    99 209.95834350585938
    199 144.66018676757812
    299 100.70249938964844
    399 71.03519439697266
    499 50.97850799560547
    599 37.403133392333984
    699 28.206867218017578
    799 21.973188400268555
    899 17.7457275390625
    999 14.877889633178711
    1099 12.931766510009766
    1199 11.610918045043945
    1299 10.714258193969727
    1399 10.10548210144043
    1499 9.692106246948242
    1599 9.411375045776367
    1699 9.220745086669922
    1799 9.091285705566406
    1899 9.003361701965332
    1999 8.943639755249023
    Result: y = -5.423830273798558e-09 + -2.208526849746704 * P3(1.3320399228078372e-09 + 0.2554861009120941 x)
    

## nn모듈
연산 그래프와 autograd는 복잡한 연산자를 정의하고 도함수(derivative)를 자동으로 계산하는 매우 강력한 패러다임(paradigm)임  
텐서플로우(Tensorflow)에서는, Keras 와 TensorFlow-Slim, TFLearn 같은 패키지들이 연산 그래프를 고수준(high-level)으로 추상화(abstraction)하여  
제공하므로 신경망을 구축하는데 유용  
-> 파이토치(PyTorch)에서는 nn 패키지가 동일한 목적으로 제공


```python
# 입/출력 텐서 생성
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p) # (2000, 1) shape / (3, ) shape

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# MSE
loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-6

for t in range(2000):
    # 순전파
    y_pred = model(xx)
    loss = loss_fn(y_pred, y)
    if t%100==99: print(t, loss.item())
    # 역전파를 시작하기 전에, 변화도를 0으로 만듬
    model.zero_grad()
    # 역전파
    loss.backward()
    # 경사 하강법을 사용해 가중치를 갱신
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate*param.grad

linear_layer = model[0]

print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

    99 982.2666015625
    199 653.8624267578125
    299 436.30999755859375
    399 292.17388916015625
    499 196.6658935546875
    599 133.37106323242188
    699 91.41831970214844
    799 63.606956481933594
    899 45.16706085205078
    999 32.938541412353516
    1099 24.827646255493164
    1199 19.44669532775879
    1299 15.876077651977539
    1399 13.506179809570312
    1499 11.932876586914062
    1599 10.888101577758789
    1699 10.194124221801758
    1799 9.73302173614502
    1899 9.426568031311035
    1999 9.222803115844727
    Result: y = -0.006762427743524313 + 0.8381737470626831 x + 0.0011666310019791126 x^2 + -0.09068938344717026 x^3
    

## optim



```python
# 입력값과 출력값을 갖는 텐서들을 생성합니다.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# 입력 텐서 (x, x^2, x^3)를 준비합니다.
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# nn 패키지를 사용하여 모델과 손실 함수를 정의합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    # 순전파 단계: 모델에 x를 전달하여 예측값 y를 계산합니다.
    y_pred = model(xx)

    # 손실을 계산하고 출력합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계 전에, optimizer 객체를 사용하여 (모델의 학습 가능한 가중치인) 갱신할
    # 변수들에 대한 모든 변화도(gradient)를 0으로 만듭니다. 이렇게 하는 이유는 기본적으로 
    # .backward()를 호출할 때마다 변화도가 버퍼(buffer)에 (덮어쓰지 않고) 누적되기
    # 때문입니다. 더 자세한 내용은 torch.autograd.backward에 대한 문서를 참조하세요.
    optimizer.zero_grad()

    # 역전파 단계: 모델의 매개변수들에 대한 손실의 변화도를 계산합니다.
    loss.backward()

    # optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
    optimizer.step()


linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

    99 1341.424560546875
    199 1132.794677734375
    299 941.76953125
    399 769.7745361328125
    499 616.8236694335938
    599 482.7627868652344
    699 367.20428466796875
    799 269.63458251953125
    899 190.11915588378906
    999 126.16220092773438
    1099 78.390625
    1199 45.023807525634766
    1299 24.294246673583984
    1399 13.696269989013672
    1499 9.783488273620605
    1599 8.96534252166748
    1699 8.909524917602539
    1799 8.921245574951172
    1899 8.922243118286133
    1999 8.919466018676758
    Result: y = 0.0004976235213689506 + 0.8562686443328857 x + 0.0004976146738044918 x^2 + -0.09383021295070648 x^3
    
