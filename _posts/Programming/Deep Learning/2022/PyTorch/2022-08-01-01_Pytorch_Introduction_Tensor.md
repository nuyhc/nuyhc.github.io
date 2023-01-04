---
title: "[PyTorch] Tensor"
date: 2022-08-01T08:53:53.146Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# TENSOR(텐서)
## 파이토치(PyTorch)가 무엇인가요?
Python 기반의 과학 연산 패키지로 다음 두 가지 목적으로 제공
- GPU 및 다른 가속기의 성능을 사용하기 위한 NumPy의 대체제 제공
- 신경망 구현에 유용한 자동 미분(automatic differntiation) 라이브러리 제공

## 텐서 (TENSOR)
텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조  
PyTorch에서는 텐서를 사용해 모델의 입/출력뿐만 아니라 매개변수를 부호화(encode)  

GPU나 다른 연산 가속을 위한 특수한 하드웨어에서 실행할 수 있다는 점을 제외하면, 텐서는 NumPy의 ndarray와 매우 유사  


```python
import torch
import numpy as np
```

### 텐서 초기화하기
#### 데이터로부터 직접 생성하기
데이터로부터 직접 텐서를 생성할 수 있고, 데이터의 자료형(data type)은 자동으로 유추


```python
data = [[1, 2],
        [3, 4]]

x_data = torch.tensor(data)
```


```python
x_data
```




    tensor([[1, 2],
            [3, 4]])



#### NumPy 배열로부터 생성하기


```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```


```python
x_np
```




    tensor([[1, 2],
            [3, 4]], dtype=torch.int32)



#### 다른 텐서로부터 생성하기
명시적으로 재정의(override)하지 않는다면, 인자로 주어진 텐서의 속성(`모양(shape), 자료형(datatype)`)을 유지


```python
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지
print(f"Ones Tensor:\n {x_ones}")
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씀
print(f"Random Tensor:\n {x_rand}")
```

    Ones Tensor:
     tensor([[1, 1],
            [1, 1]])
    Random Tensor:
     tensor([[0.5501, 0.1136],
            [0.8837, 0.9330]])
    

#### 무작위(random) 또는 상수(constant) 값을 사용하기
`shape`는 텐서의 차원(dimension)을 나타내는 튜플(tuple)로, 아래 함수들에서는 출력 텐서의 차원을 결정


```python
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

    Random Tensor: 
     tensor([[0.5479, 0.9156, 0.1940],
            [0.4972, 0.9222, 0.7007]]) 
    
    Ones Tensor: 
     tensor([[1., 1., 1.],
            [1., 1., 1.]]) 
    
    Zeros Tensor: 
     tensor([[0., 0., 0.],
            [0., 0., 0.]])
    

### 텐서의 속성(Attribute)
텐서의 속성은 텐서의 **모양(shape), 자료형(datatype) 및 어느 장치에 저장되는지**를 나타냄


```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

    Shape of tensor: torch.Size([3, 4])
    Datatype of tensor: torch.float32
    Device tensor is stored on: cpu
    

### 텐서 연산(Operation)
전치(transposing), 인덱싱(indexing), 슬라이싱(slicing), 수학 계산, 선형 대수, 임의 샘플링(random sampling) 등, 100가지 이상의 텐서 연산을 지원  
[텐서 연산 목록](https://pytorch.org/docs/stable/torch.html)

#### NumPy식의 표준 인덱싱과 슬라이싱


```python
tensor = torch.ones(4, 4)
print(tensor)
tensor[:, 1] = 0
print(tensor)
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])
    

#### 텐서 합치기
주어진 차원에 따라 일련의 텐서를 연결할 수 있음  


```python
tensor
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])




```python
torch.cat([tensor, tensor, tensor], dim=1)
```




    tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])




```python
torch.stack([tensor, tensor, tensor], dim=1)
```




    tensor([[[1., 0., 1., 1.],
             [1., 0., 1., 1.],
             [1., 0., 1., 1.]],
    
            [[1., 0., 1., 1.],
             [1., 0., 1., 1.],
             [1., 0., 1., 1.]],
    
            [[1., 0., 1., 1.],
             [1., 0., 1., 1.],
             [1., 0., 1., 1.]],
    
            [[1., 0., 1., 1.],
             [1., 0., 1., 1.],
             [1., 0., 1., 1.]]])



`cat`과 `stack`은 `pd.concat`에서 `axis` 옵션의 차이와 유사한듯
#### 텐서 곱하기
##### 요소별 곱(element-wise)


```python
tensor.mul(tensor)
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])




```python
tensor * tensor
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])



##### 행렬 곱(matrix multiplication)


```python
tensor.matmul(tensor.T)
```




    tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])




```python
tensor @ tensor.T
```




    tensor([[3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]])



### 바꿔치기(in-plcae) 연산
`_` 접미사를 갖는 연산들은 바꿔치기(in-place) 연산


```python
tensor
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])




```python
tensor.add(5)
tensor
```




    tensor([[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [1., 0., 1., 1.]])




```python
tensor.add_(5)
tensor
```




    tensor([[6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.],
            [6., 5., 6., 6.]])



### NumPy 변환 (Bridge)
CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경
#### Tensor -> NumPy


```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

    t: tensor([1., 1., 1., 1., 1.])
    n: [1. 1. 1. 1. 1.]
    

텐서의 변경 사항이 NumPy 배열에 반영


```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

    t: tensor([2., 2., 2., 2., 2.])
    n: [2. 2. 2. 2. 2.]
    

#### NumPy -> Tensor


```python
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")
```

    n: [1. 1. 1. 1. 1.]
    t: tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
    


```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

    t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    n: [2. 2. 2. 2. 2.]
    
