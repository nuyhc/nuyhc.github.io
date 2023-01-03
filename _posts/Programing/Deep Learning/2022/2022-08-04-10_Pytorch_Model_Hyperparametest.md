---
title: "[PyTorch] Hyperparameter"
date: 2022-08-04T12:16:17.294Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# Hyperparameter
## Hyperparameter vs Parameter
- Parameter
  - 모델 내부에서 결정되는 값
  - 데이터로부터 결정 됨
- Hyperparameter
  - 모델링시 사용자가 직접 세팅해주는 값
  - 최적의 값은 존재하지 않고, 휴리스틱하게 경험 법칙(rules of thumb)에 의해 결정되는 경우가 많음

## 모델 매개변수 최적화하기


```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
# Dataset / DataLoader
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# NN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

### 하이퍼파라미터(Hyperparameter)
하이퍼파라미터는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수  
서로 다른 하이퍼파라미터 값은 모델 학습과 수렴율(convergence rate)에 영향을 미칠 수 있음  

학습시에는 다음과 같은 하이퍼파라미터를 정의
- 에폭(epoch)
- 배치 크기(batch size)
- 학습률(learning rate)


```python
 epochs = 5
 batch_size = 64
 learning_rate = 1e-3
```

### 최적화 단계(Optimization Loop)
하이퍼파라미터를 설정한 뒤, 최적환 단계를 통해 모델을 학습하고 최적화 할 수 있음  
-> 최적화 단계의 각 반복(iteration)을 **에폭(epoch)** 이라고 부름  
- 학습 단계(train loop): 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴
- 검증/테스트 단계(validation/test loop): 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)

### 손실 함수(loss function)
학습용 데이터를 제공하면, 학습되지 않은 신경망은 정답을 제공하지 않을 확률이 높음  
손실 함수(loss function)는 획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarity)를 측정하며, 학습 중에 이 값을 최소화하려고 함  
주어진 데이터 샘플을 입력으로 계산한 예측과 정답을 비교해 손실(loss)을 계산  
- `nn.MSELoss`
- `nn.NLLLoss`
- `nn.CrossEntropyLoss`


```python
loss_fn = nn.CrossEntropyLoss()
```

### 옵티마이저(Optimizer)
최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정  
최적화 알고리즘은 이 과정이 수행되는 방식을 정의  
모든 최적환 절차(logic)는 `optimizer` 객체에 캡슐화(encapsulate) 됨


```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

학습 단계에서 최적화는 세단계로 이뤄짐  
- `optimizer.zero_grad()`를 호출하여 모델 매개변수의 변화도를 재설정 (기본적으로 변화도는 더해지기때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로 0으로 설정)
- `loss.backwards()`를 호출하여 예측 손실(prediction loss)을 역전파 (각 매개변수에 대한 손실의 변화도를 저장)
- 변화도 계산 이후, `optimizer.step()`을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정


```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```


```python
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

    Epoch 1
    -------------------------------
    loss: 2.301477  [    0/60000]
    loss: 2.289027  [ 6400/60000]
    loss: 2.273893  [12800/60000]
    loss: 2.267931  [19200/60000]
    loss: 2.254754  [25600/60000]
    loss: 2.221437  [32000/60000]
    loss: 2.229231  [38400/60000]
    loss: 2.194830  [44800/60000]
    loss: 2.192768  [51200/60000]
    loss: 2.154762  [57600/60000]
    Test Error: 
     Accuracy: 52.5%, Avg loss: 2.156031 
    
    Epoch 2
    -------------------------------
    loss: 2.169882  [    0/60000]
    loss: 2.153263  [ 6400/60000]
    loss: 2.104008  [12800/60000]
    loss: 2.118407  [19200/60000]
    loss: 2.072300  [25600/60000]
    loss: 2.009760  [32000/60000]
    loss: 2.038347  [38400/60000]
    loss: 1.962625  [44800/60000]
    loss: 1.962357  [51200/60000]
    loss: 1.881872  [57600/60000]
    Test Error: 
     Accuracy: 58.7%, Avg loss: 1.888902 
    
    ...
    
    Epoch 10
    -------------------------------
    loss: 0.825386  [    0/60000]
    loss: 0.890994  [ 6400/60000]
    loss: 0.671051  [12800/60000]
    loss: 0.867102  [19200/60000]
    loss: 0.768304  [25600/60000]
    loss: 0.757851  [32000/60000]
    loss: 0.847798  [38400/60000]
    loss: 0.809082  [44800/60000]
    loss: 0.823819  [51200/60000]
    loss: 0.779364  [57600/60000]
    Test Error: 
     Accuracy: 71.0%, Avg loss: 0.779402 
    
    Done!
    
