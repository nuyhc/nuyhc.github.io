---
title: 판다스 요약 정리
date: 2022-05-14T15:57:58.739Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Tutorial
---

# Pandas Summary
- [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) 참고

## 모듈


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 1. 객체 생성 (Object Creation)
Pandas는 기본적으로 2개의 데이터 형식을 가지고 있음
1. Series
- 벡터 형식의 데이터
2. DataFrame
- 행렬 형식의 데이터

### Series와 DataFrame
Pandas는 값을 가지고 있는 리스트를 통해 Series를 만들고, 정수 인덱스를 기본으로 제공


```python
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)
```

    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64
    


```python
dates = pd.date_range("20220513", periods=6)
print(dates)
```

    DatetimeIndex(['2022-05-13', '2022-05-14', '2022-05-15', '2022-05-16',
                   '2022-05-17', '2022-05-18'],
                  dtype='datetime64[ns]', freq='D')
    

DataFrame은 행과 열을 가진 데이터 형식으로, 인덱스와 레이블이 있는 열을 가지고 있는 np 배열을 전달해 생성 할 수 있다.


```python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list("ABCD"))
print(df)
```

                       A         B         C         D
    2022-05-13 -0.364352 -1.806739 -0.605339 -0.259275
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498
    


```python
df2 = pd.DataFrame({'A': 1.,
              'B': pd.Timestamp("20220513"),
              'C': pd.Series(1, index=list(range(4)), dtype='float32'),
              'D': np.array([3]*4, dtype='int32'),
              'E': pd.Categorical(["test", "train", "test", "train"]),
              'F': "foo" })
print(df2)
```

         A          B    C  D      E    F
    0  1.0 2022-05-13  1.0  3   test  foo
    1  1.0 2022-05-13  1.0  3  train  foo
    2  1.0 2022-05-13  1.0  3   test  foo
    3  1.0 2022-05-13  1.0  3  train  foo
    

- `pd.Timestamp()`
  - para: year, month, day, hour, minute, second
  - 하나의 문자열로 넣어도 변환되는거 같음


```python
print(df2.dtypes)
```

    A           float64
    B    datetime64[ns]
    C           float32
    D             int32
    E          category
    F            object
    dtype: object
    

## 2. 데이터 확인하기 (Viewing Data)
### tail과 head
`tail`과 `head` 함수의 경우 인자로 숫자를 넣지 않을 경우, 기본값인 5로 처리됨


```python
print(df.tail())
```

                       A         B         C         D
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498
    


```python
print(df.head())
```

                       A         B         C         D
    2022-05-13 -0.364352 -1.806739 -0.605339 -0.259275
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702
    

### 인덱스(index), 열(column), numpy 데이터에 대한 세부 정보


```python
print(df.index)
```

    DatetimeIndex(['2022-05-13', '2022-05-14', '2022-05-15', '2022-05-16',
                   '2022-05-17', '2022-05-18'],
                  dtype='datetime64[ns]', freq='D')
    


```python
print(df.columns)
```

    Index(['A', 'B', 'C', 'D'], dtype='object')
    


```python
print(df.values)
```

    [[-0.36435205 -1.80673941 -0.60533949 -0.25927545]
     [ 1.79202234 -0.10922598 -0.78255824  0.95971119]
     [-0.03553265 -1.69541806 -0.26203292 -0.73922933]
     [-0.93887282  0.24653065 -1.11889869 -0.01798904]
     [ 0.33573461 -0.9354455  -0.21775696  0.198702  ]
     [-0.09965588 -0.01101404  0.5773068  -1.97349815]]
    

### 대략적인 통계적 정보 요약


```python
print(df.describe())
```

                  A         B         C         D
    count  6.000000  6.000000  6.000000  6.000000
    mean   0.114891 -0.718552 -0.401547 -0.305263
    std    0.924153  0.893167  0.585268  0.990971
    min   -0.938873 -1.806739 -1.118899 -1.973498
    25%   -0.298178 -1.505425 -0.738254 -0.619241
    50%   -0.067594 -0.522336 -0.433686 -0.138632
    75%    0.242918 -0.035567 -0.228826  0.144529
    max    1.792022  0.246531  0.577307  0.959711
    

### 데이터 전치


```python
print(df.T)
```

       2022-05-13  2022-05-14  2022-05-15  2022-05-16  2022-05-17  2022-05-18
    A   -0.364352    1.792022   -0.035533   -0.938873    0.335735   -0.099656
    B   -1.806739   -0.109226   -1.695418    0.246531   -0.935445   -0.011014
    C   -0.605339   -0.782558   -0.262033   -1.118899   -0.217757    0.577307
    D   -0.259275    0.959711   -0.739229   -0.017989    0.198702   -1.973498
    

### 축 별로 정렬


```python
df.sort_index(axis=0, ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-18</th>
      <td>-0.099656</td>
      <td>-0.011014</td>
      <td>0.577307</td>
      <td>-1.973498</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
      <td>-0.217757</td>
      <td>0.198702</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
      <td>-1.118899</td>
      <td>-0.017989</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
      <td>-0.605339</td>
      <td>-0.259275</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(axis=1, ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>C</th>
      <th>B</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.259275</td>
      <td>-0.605339</td>
      <td>-1.806739</td>
      <td>-0.364352</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>0.959711</td>
      <td>-0.782558</td>
      <td>-0.109226</td>
      <td>1.792022</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.739229</td>
      <td>-0.262033</td>
      <td>-1.695418</td>
      <td>-0.035533</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.017989</td>
      <td>-1.118899</td>
      <td>0.246531</td>
      <td>-0.938873</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.198702</td>
      <td>-0.217757</td>
      <td>-0.935445</td>
      <td>0.335735</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>-1.973498</td>
      <td>0.577307</td>
      <td>-0.011014</td>
      <td>-0.099656</td>
    </tr>
  </tbody>
</table>
</div>



### 값 별로 정렬


```python
df.sort_values(by='B')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
      <td>-0.605339</td>
      <td>-0.259275</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
      <td>-0.217757</td>
      <td>0.198702</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>-0.099656</td>
      <td>-0.011014</td>
      <td>0.577307</td>
      <td>-1.973498</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
      <td>-1.118899</td>
      <td>-0.017989</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 선택 (Selection)
### 데이터 얻기 (Getting)


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
      <td>-0.605339</td>
      <td>-0.259275</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
      <td>-1.118899</td>
      <td>-0.017989</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
      <td>-0.217757</td>
      <td>0.198702</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>-0.099656</td>
      <td>-0.011014</td>
      <td>0.577307</td>
      <td>-1.973498</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['A']
```




    2022-05-13   -0.364352
    2022-05-14    1.792022
    2022-05-15   -0.035533
    2022-05-16   -0.938873
    2022-05-17    0.335735
    2022-05-18   -0.099656
    Freq: D, Name: A, dtype: float64



행을 분할 할수도 있음


```python
df[0:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
      <td>-0.605339</td>
      <td>-0.259275</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['2022-05-13':'2022-05-15']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
      <td>-0.605339</td>
      <td>-0.259275</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
    </tr>
  </tbody>
</table>
</div>



### 라벨을 통한 선택


```python
print(dates)
```

    DatetimeIndex(['2022-05-13', '2022-05-14', '2022-05-15', '2022-05-16',
                   '2022-05-17', '2022-05-18'],
                  dtype='datetime64[ns]', freq='D')
    


```python
df.loc[dates[0]]
```




    A   -0.364352
    B   -1.806739
    C   -0.605339
    D   -0.259275
    Name: 2022-05-13 00:00:00, dtype: float64




```python
df.loc[:,['A','B']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>-0.364352</td>
      <td>-1.806739</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>-0.099656</td>
      <td>-0.011014</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc['2022-05-14':'2022-05-17', ['B','D']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-14</th>
      <td>-0.109226</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-1.695418</td>
      <td>-0.739229</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>0.246531</td>
      <td>-0.017989</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>-0.935445</td>
      <td>0.198702</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[dates[0], 'A']
```




    -0.3643520548458078



### 위치를 통한 선택



```python
df.iloc[3]
```




    A   -0.938873
    B    0.246531
    C   -1.118899
    D   -0.017989
    Name: 2022-05-16 00:00:00, dtype: float64




```python
df.iloc[3:5, 0:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[[1,2,4],[0,2]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.782558</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-0.262033</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.217757</td>
    </tr>
  </tbody>
</table>
</div>



- loc: 인덱스 명칭을 기준으로 값을 색인
- iloc: 인덱스 순서를 기준으로 값을 색인, range 값을 기준으로

`df.set_index[키워드]` 형식으로 인덱스를 설정할 수 있음

### 특정한 값 얻기


```python
df.iloc[1,1]
```




    -0.10922598205957576



### Boolean Indexing


```python
df[df['A']>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
      <td>-0.217757</td>
      <td>0.198702</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df>0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.959711</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>NaN</td>
      <td>0.246531</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.198702</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.577307</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df.copy()
df2['E'] = ["one", "one", "two", "three", "four", "three"]
print(df2)
```

                       A         B         C         D      E
    2022-05-13 -0.364352 -1.806739 -0.605339 -0.259275    one
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711    one
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229    two
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989  three
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702   four
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498  three
    


```python
df2[df2['E'].isin(["two", "four"])]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>-0.739229</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>0.335735</td>
      <td>-0.935445</td>
      <td>-0.217757</td>
      <td>0.198702</td>
      <td>four</td>
    </tr>
  </tbody>
</table>
</div>



### Setting
새 열을 설정하면 데이터가 인덱스 별로 자동 정렬


```python
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20220513", periods=6))
print(s1)
```

    2022-05-13    1
    2022-05-14    2
    2022-05-15    3
    2022-05-16    4
    2022-05-17    5
    2022-05-18    6
    Freq: D, dtype: int64
    


```python
df['F'] = s1
print(df)
```

                       A         B         C         D  F
    2022-05-13 -0.364352 -1.806739 -0.605339 -0.259275  1
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711  2
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229  3
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989  4
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702  5
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498  6
    


```python
df.at[dates[0], 'A'] = 0
print(df)
```

                       A         B         C         D  F
    2022-05-13  0.000000 -1.806739 -0.605339 -0.259275  1
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711  2
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229  3
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989  4
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702  5
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498  6
    


```python
df.iat[0,1] = 0
print(df)
```

                       A         B         C         D  F
    2022-05-13  0.000000  0.000000 -0.605339 -0.259275  1
    2022-05-14  1.792022 -0.109226 -0.782558  0.959711  2
    2022-05-15 -0.035533 -1.695418 -0.262033 -0.739229  3
    2022-05-16 -0.938873  0.246531 -1.118899 -0.017989  4
    2022-05-17  0.335735 -0.935445 -0.217757  0.198702  5
    2022-05-18 -0.099656 -0.011014  0.577307 -1.973498  6
    


```python
df.loc[:, 'D'] = np.array([5]*len(df))
print(df)
```

                       A         B         C  D  F
    2022-05-13  0.000000  0.000000 -0.605339  5  1
    2022-05-14  1.792022 -0.109226 -0.782558  5  2
    2022-05-15 -0.035533 -1.695418 -0.262033  5  3
    2022-05-16 -0.938873  0.246531 -1.118899  5  4
    2022-05-17  0.335735 -0.935445 -0.217757  5  5
    2022-05-18 -0.099656 -0.011014  0.577307  5  6
    


```python
df2 = df.copy()
df2[df2>0] = -df2
print(df2)
```

                       A         B         C  D  F
    2022-05-13  0.000000  0.000000 -0.605339 -5 -1
    2022-05-14 -1.792022 -0.109226 -0.782558 -5 -2
    2022-05-15 -0.035533 -1.695418 -0.262033 -5 -3
    2022-05-16 -0.938873 -0.246531 -1.118899 -5 -4
    2022-05-17 -0.335735 -0.935445 -0.217757 -5 -5
    2022-05-18 -0.099656 -0.011014 -0.577307 -5 -6
    

## 4. 결측치 (Missing Data)
Pandas는 결측치를 표한하기 위해 주로 np.nan 값을 사용

### Reindexing
지정된 축 상의 인덱스를 변경 / 추가 / 삭제 할 수 있고 데이터의 복사본은 반환


```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns)+['E'])
print(df1)
```

                       A         B         C  D  F   E
    2022-05-13  0.000000  0.000000 -0.605339  5  1 NaN
    2022-05-14  1.792022 -0.109226 -0.782558  5  2 NaN
    2022-05-15 -0.035533 -1.695418 -0.262033  5  3 NaN
    2022-05-16 -0.938873  0.246531 -1.118899  5  4 NaN
    


```python
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)
```

                       A         B         C  D  F    E
    2022-05-13  0.000000  0.000000 -0.605339  5  1  1.0
    2022-05-14  1.792022 -0.109226 -0.782558  5  2  1.0
    2022-05-15 -0.035533 -1.695418 -0.262033  5  3  NaN
    2022-05-16 -0.938873  0.246531 -1.118899  5  4  NaN
    

결측치를 가지고 있는 행들을 지움


```python
df1.dropna(how='any')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.605339</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>5</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



결측치를 채워 넣음


```python
df1.fillna(value=5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.605339</td>
      <td>5</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-0.782558</td>
      <td>5</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-0.035533</td>
      <td>-1.695418</td>
      <td>-0.262033</td>
      <td>5</td>
      <td>3</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-0.938873</td>
      <td>0.246531</td>
      <td>-1.118899</td>
      <td>5</td>
      <td>4</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



nan인 값에 boolean을 통한 표식
- 데이터프레임의 모든 값이 boolean 형태로 표시
- nan인 값에만 True가 표시되게 함


```python
pd.isna(df1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



## 5. 연산 (Operation)
### 통계 (Stats)
- 일반적으로 결측치를 제외한 후 연산
#### 평균
축 선택이 가능함


```python
df.mean() # default: 열(row)
```




    A    0.175616
    B   -0.417429
    C   -0.401547
    D    5.000000
    F    3.500000
    dtype: float64




```python
df.mean(1) # 행(col)
```




    2022-05-13    1.078932
    2022-05-14    1.580048
    2022-05-15    1.201403
    2022-05-16    1.437752
    2022-05-17    1.836506
    2022-05-18    2.293327
    Freq: D, dtype: float64



### pd.shift()
행의 위치를 일정 칸수씩 이동


```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates)
s
```




    2022-05-13    1.0
    2022-05-14    3.0
    2022-05-15    5.0
    2022-05-16    NaN
    2022-05-17    6.0
    2022-05-18    8.0
    Freq: D, dtype: float64




```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
s
```




    2022-05-13    NaN
    2022-05-14    NaN
    2022-05-15    1.0
    2022-05-16    3.0
    2022-05-17    5.0
    2022-05-18    NaN
    Freq: D, dtype: float64




```python
df.sub(s, axis='index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>-1.035533</td>
      <td>-2.695418</td>
      <td>-1.262033</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>-3.938873</td>
      <td>-2.753469</td>
      <td>-4.118899</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>-4.664265</td>
      <td>-5.935445</td>
      <td>-5.217757</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Apply
데이터에 함수를 적용


```python
df.apply(np.cumsum)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-05-13</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.605339</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2022-05-14</th>
      <td>1.792022</td>
      <td>-0.109226</td>
      <td>-1.387898</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2022-05-15</th>
      <td>1.756490</td>
      <td>-1.804644</td>
      <td>-1.649931</td>
      <td>15</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2022-05-16</th>
      <td>0.817617</td>
      <td>-1.558113</td>
      <td>-2.768829</td>
      <td>20</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2022-05-17</th>
      <td>1.153351</td>
      <td>-2.493559</td>
      <td>-2.986586</td>
      <td>25</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2022-05-18</th>
      <td>1.053696</td>
      <td>-2.504573</td>
      <td>-2.409279</td>
      <td>30</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.apply(lambda x: x.max()-x.min())
```




    A    2.730895
    B    1.941949
    C    1.696205
    D    0.000000
    F    5.000000
    dtype: float64



### 히스토그래밍



```python
s = pd.Series(np.random.randint(0, 7, size=10))
s
```




    0    6
    1    5
    2    5
    3    5
    4    1
    5    0
    6    5
    7    1
    8    1
    9    0
    dtype: int32




```python
print(s.value_counts())
```

    5    4
    1    3
    0    2
    6    1
    dtype: int64
    

### 문자열 메소드 (String Methods)
- 문자열의 패턴 일치 확인은 기본적으로 정규 표현식을 사용
- 몇몇 경우에는 항상 정규 표현식을 사용
- `str` 메서드를 이용
- [참고 문서](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html)


```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())
```

    0       a
    1       b
    2       c
    3    aaba
    4    baca
    5     NaN
    6    caba
    7     dog
    8     cat
    dtype: object
    

## 6. 병합 (Merge)
### 연결 (Concat)
결합 (join) / 병합 (merge) 형태의 연산에 대한 인덱스, 관계 대수 기능을 위한 다양한 형태의 논리를 포함한 객체를 손쉽게 결합할 수 있음


```python
df = pd.DataFrame(np.random.randn(10, 4))
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.434956</td>
      <td>1.443564</td>
      <td>0.644435</td>
      <td>1.496020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.837197</td>
      <td>-0.217102</td>
      <td>0.341705</td>
      <td>0.325993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.820944</td>
      <td>0.776307</td>
      <td>-1.041641</td>
      <td>0.713703</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.405343</td>
      <td>-1.151217</td>
      <td>-0.508435</td>
      <td>0.075641</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.487023</td>
      <td>0.446358</td>
      <td>1.297404</td>
      <td>1.569944</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.271487</td>
      <td>0.630323</td>
      <td>0.196226</td>
      <td>-0.552806</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.271090</td>
      <td>0.558662</td>
      <td>0.716677</td>
      <td>-1.491536</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.054462</td>
      <td>1.934392</td>
      <td>-0.031558</td>
      <td>-0.740327</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.840678</td>
      <td>0.847108</td>
      <td>0.349140</td>
      <td>-1.254350</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.857202</td>
      <td>-0.708943</td>
      <td>0.179937</td>
      <td>-0.259415</td>
    </tr>
  </tbody>
</table>
</div>




```python
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.434956</td>
      <td>1.443564</td>
      <td>0.644435</td>
      <td>1.496020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.837197</td>
      <td>-0.217102</td>
      <td>0.341705</td>
      <td>0.325993</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.820944</td>
      <td>0.776307</td>
      <td>-1.041641</td>
      <td>0.713703</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.405343</td>
      <td>-1.151217</td>
      <td>-0.508435</td>
      <td>0.075641</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.487023</td>
      <td>0.446358</td>
      <td>1.297404</td>
      <td>1.569944</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.271487</td>
      <td>0.630323</td>
      <td>0.196226</td>
      <td>-0.552806</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.271090</td>
      <td>0.558662</td>
      <td>0.716677</td>
      <td>-1.491536</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.054462</td>
      <td>1.934392</td>
      <td>-0.031558</td>
      <td>-0.740327</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.840678</td>
      <td>0.847108</td>
      <td>0.349140</td>
      <td>-1.254350</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.857202</td>
      <td>-0.708943</td>
      <td>0.179937</td>
      <td>-0.259415</td>
    </tr>
  </tbody>
</table>
</div>



### 결합 (Join)
SQL 방식으로 병합


```python
left = pd.DataFrame({'key':['foo', 'foo'], 'lval':[1, 2]})
right = pd.DataFrame({'key':['foo', 'foo'], 'rval':[4,5]})
```


```python
left
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>lval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
right
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>rval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.merge(left, right, on='key')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>lval</th>
      <th>rval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>foo</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>foo</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>foo</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
left = pd.DataFrame({'key':['foo', 'bar'], 'lval':[1, 2]})
right = pd.DataFrame({'key':['foo', 'bar'], 'rval':[4,5]})
```


```python
pd.merge(left, right, on='key')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>key</th>
      <th>lval</th>
      <th>rval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>foo</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bar</td>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### 추가 (Append)


```python
df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.925361</td>
      <td>1.590198</td>
      <td>-0.046855</td>
      <td>-1.486980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.970544</td>
      <td>-0.441473</td>
      <td>-0.058223</td>
      <td>-1.217569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.822900</td>
      <td>0.292400</td>
      <td>0.607530</td>
      <td>-1.003081</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.119498</td>
      <td>0.706862</td>
      <td>1.178307</td>
      <td>2.322978</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.532140</td>
      <td>2.189385</td>
      <td>-0.852177</td>
      <td>-0.113575</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.595338</td>
      <td>-2.415004</td>
      <td>1.331114</td>
      <td>1.422960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.103675</td>
      <td>-0.270880</td>
      <td>0.539115</td>
      <td>-0.082935</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.319083</td>
      <td>0.996896</td>
      <td>-0.603317</td>
      <td>0.223083</td>
    </tr>
  </tbody>
</table>
</div>




```python
s = df.iloc[3]
s
```




    A    0.119498
    B    0.706862
    C    1.178307
    D    2.322978
    Name: 3, dtype: float64




```python
df.append(s, ignore_index=True)
```

    C:\Users\spec3\AppData\Local\Temp/ipykernel_10056/4011806271.py:1: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df.append(s, ignore_index=True)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.925361</td>
      <td>1.590198</td>
      <td>-0.046855</td>
      <td>-1.486980</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.970544</td>
      <td>-0.441473</td>
      <td>-0.058223</td>
      <td>-1.217569</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.822900</td>
      <td>0.292400</td>
      <td>0.607530</td>
      <td>-1.003081</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.119498</td>
      <td>0.706862</td>
      <td>1.178307</td>
      <td>2.322978</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.532140</td>
      <td>2.189385</td>
      <td>-0.852177</td>
      <td>-0.113575</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.595338</td>
      <td>-2.415004</td>
      <td>1.331114</td>
      <td>1.422960</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.103675</td>
      <td>-0.270880</td>
      <td>0.539115</td>
      <td>-0.082935</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.319083</td>
      <td>0.996896</td>
      <td>-0.603317</td>
      <td>0.223083</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.119498</td>
      <td>0.706862</td>
      <td>1.178307</td>
      <td>2.322978</td>
    </tr>
  </tbody>
</table>
</div>



`append`는 추후 삭제될 예정이므로 `concat`을 사용하라고 함

## 7. 그룹화 (Grouping)
그룹화는 다음 단계 중 하나 이상을 포함하는 과정
1. 몇몇 기준에 따라 여러 그룹을 데이터를 분할 -> splitting
2. 각 그룹에 독립적으로 함수를 적용 -> applying
3. 결과물들을 하나의 데이터 구조로 결합 -> combining


```python
df = pd.DataFrame(
    {
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'two', 'three', 'two', 'two','one', 'three'],
        'C': np.random.randn(8),
        'D': np.random.randn(8)
    }
)

print(df)
```

         A      B         C         D
    0  foo    one -0.641923  0.427765
    1  bar    one -0.643603  0.494289
    2  foo    two -0.186318 -0.767165
    3  bar  three  0.588941 -0.066441
    4  foo    two -1.536313  0.290155
    5  bar    two  0.449030 -0.116639
    6  foo    one -0.956890  0.661999
    7  foo  three -0.016754  0.907288
    


```python
df.groupby('A').sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>D</th>
    </tr>
    <tr>
      <th>A</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bar</th>
      <td>0.394368</td>
      <td>0.311210</td>
    </tr>
    <tr>
      <th>foo</th>
      <td>-3.338199</td>
      <td>1.520043</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(['A', 'B']).sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>C</th>
      <th>D</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">bar</th>
      <th>one</th>
      <td>-0.643603</td>
      <td>0.494289</td>
    </tr>
    <tr>
      <th>three</th>
      <td>0.588941</td>
      <td>-0.066441</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.449030</td>
      <td>-0.116639</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">foo</th>
      <th>one</th>
      <td>-1.598813</td>
      <td>1.089764</td>
    </tr>
    <tr>
      <th>three</th>
      <td>-0.016754</td>
      <td>0.907288</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-1.722632</td>
      <td>-0.477010</td>
    </tr>
  </tbody>
</table>
</div>



## 8. 변형 (Reshaping)
### 스택 (Stack)


```python
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

print(tuples)
```

    [('bar', 'one'), ('bar', 'two'), ('baz', 'one'), ('baz', 'two'), ('foo', 'one'), ('foo', 'two'), ('qux', 'one'), ('qux', 'two')]
    


```python
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

print(index)
```

    MultiIndex([('bar', 'one'),
                ('bar', 'two'),
                ('baz', 'one'),
                ('baz', 'two'),
                ('foo', 'one'),
                ('foo', 'two'),
                ('qux', 'one'),
                ('qux', 'two')],
               names=['first', 'second'])
    


```python
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
print(df)
```

                         A         B
    first second                    
    bar   one     0.288745 -0.503898
          two    -0.024742  0.001157
    baz   one    -2.595053 -0.804304
          two    -0.405564 -0.110970
    foo   one     1.143463  0.600136
          two    -1.193027 -0.706129
    qux   one    -1.632288  0.230009
          two     0.391425 -0.299005
    


```python
df2 = df[:4]
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>0.288745</td>
      <td>-0.503898</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.024742</td>
      <td>0.001157</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>-2.595053</td>
      <td>-0.804304</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.405564</td>
      <td>-0.110970</td>
    </tr>
  </tbody>
</table>
</div>



stack() 메소드는 데이터프레임 열들의 계층을 **압축**


```python
stacked = df2.stack()
stacked
```




    first  second   
    bar    one     A    0.288745
                   B   -0.503898
           two     A   -0.024742
                   B    0.001157
    baz    one     A   -2.595053
                   B   -0.804304
           two     A   -0.405564
                   B   -0.110970
    dtype: float64




```python
stacked.unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>0.288745</td>
      <td>-0.503898</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.024742</td>
      <td>0.001157</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>-2.595053</td>
      <td>-0.804304</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.405564</td>
      <td>-0.110970</td>
    </tr>
  </tbody>
</table>
</div>



### 피봇 테이블 (Pivot Tables)


```python
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three']*3,
                   'B': ['A', 'B', 'C']*4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar']*2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>A</td>
      <td>foo</td>
      <td>-0.562920</td>
      <td>0.665956</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>B</td>
      <td>foo</td>
      <td>0.461604</td>
      <td>0.009933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>two</td>
      <td>C</td>
      <td>foo</td>
      <td>0.355507</td>
      <td>0.423206</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three</td>
      <td>A</td>
      <td>bar</td>
      <td>-0.388622</td>
      <td>0.509214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>B</td>
      <td>bar</td>
      <td>-0.655012</td>
      <td>0.015648</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>C</td>
      <td>bar</td>
      <td>0.285531</td>
      <td>0.566449</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>A</td>
      <td>foo</td>
      <td>-0.149651</td>
      <td>-0.964291</td>
    </tr>
    <tr>
      <th>7</th>
      <td>three</td>
      <td>B</td>
      <td>foo</td>
      <td>0.062204</td>
      <td>1.159678</td>
    </tr>
    <tr>
      <th>8</th>
      <td>one</td>
      <td>C</td>
      <td>foo</td>
      <td>-0.710424</td>
      <td>1.138363</td>
    </tr>
    <tr>
      <th>9</th>
      <td>one</td>
      <td>A</td>
      <td>bar</td>
      <td>-0.383573</td>
      <td>0.615988</td>
    </tr>
    <tr>
      <th>10</th>
      <td>two</td>
      <td>B</td>
      <td>bar</td>
      <td>0.220584</td>
      <td>-1.644652</td>
    </tr>
    <tr>
      <th>11</th>
      <td>three</td>
      <td>C</td>
      <td>bar</td>
      <td>-0.005331</td>
      <td>0.041637</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>bar</th>
      <th>foo</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>A</th>
      <td>-0.383573</td>
      <td>-0.562920</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.655012</td>
      <td>0.461604</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.285531</td>
      <td>-0.710424</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">three</th>
      <th>A</th>
      <td>-0.388622</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>NaN</td>
      <td>0.062204</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.005331</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">two</th>
      <th>A</th>
      <td>NaN</td>
      <td>-0.149651</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.220584</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>0.355507</td>
    </tr>
  </tbody>
</table>
</div>



## 9. 범주화 (Categoricals)


```python
df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>raw_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["grade"] = df["raw_grade"].astype("category")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>raw_grade</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>e</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["grade"]
```




    0    a
    1    b
    2    b
    3    a
    4    a
    5    e
    Name: grade, dtype: category
    Categories (3, object): ['a', 'b', 'e']




```python
# 의미 있는 이름 붙이기
df["grade"].cat.categories = ["very good", "good", "very bad"]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>raw_grade</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>e</td>
      <td>very bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 순서를 바꾸고 누락된 범주를 추가
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>raw_grade</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>e</td>
      <td>very bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["grade"]
```




    0    very good
    1         good
    2         good
    3    very good
    4    very good
    5     very bad
    Name: grade, dtype: category
    Categories (5, object): ['very bad', 'bad', 'medium', 'good', 'very good']



정렬은, 해당 범주에서 지정된 순서대로 배열됨


```python
df.sort_values(by="grade")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>raw_grade</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>e</td>
      <td>very bad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>b</td>
      <td>good</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>a</td>
      <td>very good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>a</td>
      <td>very good</td>
    </tr>
  </tbody>
</table>
</div>



## 10. 입/출력
### CSV 파일
- 쓰기: `df.to_csv("파일이름.csv")`
- 읽기: `pd.read_csv("파일이름.csv")`
### HDF5
- 쓰기: `df.to_hdf("파일이름.h5", 'df')`
- 읽기: `pd.read_hdf("파일이름.h5", 'df)`
### Excel
- 쓰기: `df.to_excel('파일이름.xlsx', sheet_name="시트 이름")`
- 읽기: `pd.read_excel("파일이름.xlsx", "시트 이름", index_col=None, na_values['NA'])`
