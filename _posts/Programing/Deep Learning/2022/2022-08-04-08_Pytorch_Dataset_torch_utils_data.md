---
title: "[PyTorch] Data"
date: 2022-08-04T00:06:53.020Z

categories:
  - Programming
  - Deep Learning
tags:
  - Deep Learning
  - PyTorch
  - Tutorial
---

# TORCH.UTILS.DATA
PyTorch 데이터 로딩 유틸리티의 핵심은 `torch.utils.data.DataLoader` 클래스임  
데이터 세트에 대해 반복 가능한 Python을 나타내며 다음을 지원함  
- map-스타일 및 반복 가능한 스타일 데이터 세트
- 데이터 로드 순서 사용자 정의
- 자동 배치
- 단일 및 다중 프로세스 데이터 로딩
- 자동 메모리 고정

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

## 데이터 세트 유형
`DataLoader` 생성자의 가장 중요한 파라미터는 `dataset` 데이터를 로드할 데이터세트 개체를 나타내는 파라미터임
1. map-style
2. iterable-style

### map-style
`__getitem__()` 및 `__len__()` 메서드를 구현하는 데이터 세트이며, 인덱스/키에서 데이터 샘플까지의 맵을 나타냄  
-> 예를 들어, 데이터 세트[idx]를 사용하여 액세스하면 디스크의 폴더에서 idx번째 이미지와 해당 레이블을 읽을 수 있음

### iterable-style
`__iter__()` 메서드를 구현하고 반복 가능한 개체를 나타내는 하위 클래스의 인스턴스  
랜덤 읽기가 비싸거나(리소스를 많이 사용하거나) 사용 가능성이 거의 없으며 배치 크기가 가져온 데이터에 의존하는 경우에 적합  
-> 원격 서버 또는 실시간으로 생성된 로그에서 읽은 데이터 스트림을 반환할 수 있음

## 데이터 로딩 순서 및 샘플러(Sampler)
iterable 스타일의 데이터 세트의 로드 순서는 사용자가 정의한 iterable에 의해 완전히 제어됨  
-> 청크 읽기 및 동적 배치 크기를 보다 쉽게 구현 가능  

###  map-style
`torch.tuils.data.Sampler` 클래스는 데이터 로드에 사용되는 인덱스/키의 시퀀스를 지정하는데 사용  
-> 데이터 세트에 대한 인덱스에 대해 반복 가능한 객체를 나타냄  
-> SGD의 일반적인 경우, `Sampler`에 인덱스 목록을 무작위로 치환하고 한 번에 하나씩 생성하거나 미니 배치 SGD에 대해 소수를 생성할 수 있음  
#### 일괄 및 비일괄 데이터 로드
`DataLoader`로 가져온 개별 데이터 샘플을,  
- batch_size
- drop_last
- batch_sampler
- collate_fn
들을 통해 일괄 처리로 자동 조합하는 것을 지원

#### 자동 일괄 처리(기본값)
가장 일반적인 예시로, 데이터의 미니 배치를 가져와서 배치된 샘플을 조합하는 것에 해당  
`batch_size`가 None이 아닌 경우, `DataLoader`는 개별 샘플 대신 일괄 샘플을 생성  
`Sampler`에서 인덱스를 사용해 샘플 목록을 가져온 후 `collate_fn` 인수로 전달된 함수는 샘플 목록을 일괄 처리하는데 사용

```python
for bs in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

`iterabel-style`의 경우 다음과 유사함
```python
dataset_iter = iter(dataset)
for bs in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in bs])
```
#### 자동 일괄 처리 비활성화
경우에 따라 사용자는 데이터세트 코드에서 수동으로 일괄 처리를 처리하거나 개별 샘플을 로드하기를 원할 수 있음  
-> 일괄 처리된 데이터를 직접 로드하는 것이 더 저렴할 수 있음  

-> 자동 일괄 처림를 사용하지 않고 `DataLoader`가 개체 `collate_fn`의 각 구성원을 직접 반환하도록 하는 것이 좋음  

`batch_size`와 `batch_sampler`가 모두 None인 경우 자동 일괄 처리가 비활성화 됨  
-> 자동 일괄 처리가 비활성화 되면 기본값은 단순히 넘파이 배열을 파이토치 텐서로 변환하고, 다른 모든 것은 그대로 유지

```python
for index in sampler:
    yield collate_fn(dataset[index])
```
`iterabel-style`의 경우 다음과 유사함
```python
for data in iter(dataset):
    yield collate_fn(data)
```
## collate_fn
`collate_fn`은 자동 일괄 처리가 활성화되거나 비활성화된 경우의 사용이 다름  
### 비활성화 된 경우
각 개별 데이터 샘플과 함께 호출되고 `DataLoader` 반복기에서 출력이 생성  
기본적으로 PyTorch 텐서의 NumPy 배열을 변환함  
### 활성화 된 경우
매번 데이터 샘플 목록과 함께 호출됨  
`DataLoader` 반복기에서 산출하기 위해 입력 샘플을 배치로 조합할 것으로 기대됨  
- 항상 새 차원을 일괄 처리 차원으로 추가
- NumPy 배열과 Python 숫자 값을 PyTorch 텐서로 자동 변환
- 데이터 구조를 유지
