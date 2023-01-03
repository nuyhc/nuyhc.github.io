---
title: 통계 기초
date: 2022-06-05T14:13:49.980Z
img_path: /nuyhc.github.io/assets/img
categories:
  - Programming
  - DataScience
tags:
  - Pandas
---
 
# 데이터 분석을 위한 통계 기초

### 통계학을 배우는 이유
- 데이터 = 패턴 + 노이즈
  - 측정의 불완전성
  - 데이터에 포함된 잡음
  - 제 3의 변수
- 통계의 역할: 다양한 요소를 고려해 합리적 결론을 유도
- 통계학과 머신러닝의 차이
  - 근본적인 차이는 없음
  - 머신러닝 = 인공지능 + 통계학
  - 통계학은 모형의 타당성에, 머신러닝은 예측 성능에 좀 더 관심을 기울이는 경향이 있음

### 사례와 변수
#### 사례(case)
- 데이터 수집의 단위 (제품, 실험, 고객 등)
- 데이터를 표로 나타낼때는, 한 행(row)에 표시
#### 변수(variable)
- 사례에 따라 달라지는 특성 (만족도, 성능, 색상 등)
- 데이터를 표로 나타낼때는, 한 열(column)에 표시

### 모집단과 표본
- 모집단(population): 연구의 관심이 되는 집단 전체
- 표본(sample): 특정 연구에서 선택된 모집단의 부분 집합 (사례들의 집합)

기술 통계(descriptive)는 표본을 요약하고 묘사하며, 추론 통계(inferential)는 표본을 통해 모집단에 대한 추측을 함

### 범주형 변수 (categorical variable)
- 2개 이상의 범주(category)를 값으로 가지는 변수
  - 순서가 없는 범주
  - 순서가 있는 범주

### 연속형 변수 (continuous variable)
- 실수로 표현할 수 있는 변수
- 간격이 일정
- 계산이 의미가 있음

범주형과 연속형이 헛갈릴 때는, 앞에 **평균**이라는 단어를 붙여보면 됨 -> 평균 국적?x / 평균 나이o

### 기술 통계
- 중심 경향치: 데이터가 어디에 몰려있는가?
  - 평균
  - 중간값
  - 최빈값
- 분위수: 데이터에서 각각의 순위가 어느 정도인가?
- 변산성 측정치: 데이터가 어떻게 퍼져있는가?
  - 범위
  - IQR
  - 분산
  - 표준편차


```python
import pandas as pd
import seaborn as sns
df = pd.read_excel("data/car.xlsx")
df.head()
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
      <th>mileage</th>
      <th>model</th>
      <th>price</th>
      <th>year</th>
      <th>my_car_damage</th>
      <th>other_car_damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63608</td>
      <td>K3</td>
      <td>970</td>
      <td>2017</td>
      <td>0</td>
      <td>564596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>69336</td>
      <td>K3</td>
      <td>1130</td>
      <td>2015</td>
      <td>1839700</td>
      <td>1140150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36000</td>
      <td>K3</td>
      <td>1380</td>
      <td>2016</td>
      <td>446520</td>
      <td>2244910</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19029</td>
      <td>K3</td>
      <td>1390</td>
      <td>2017</td>
      <td>889000</td>
      <td>4196110</td>
    </tr>
    <tr>
      <th>4</th>
      <td>97090</td>
      <td>K3</td>
      <td>760</td>
      <td>2015</td>
      <td>2339137</td>
      <td>2029570</td>
    </tr>
  </tbody>
</table>
</div>



#### 중심 경향치
##### 평균 (mean)
- 평균은 극단값의 영향을 크게 받음  
$\bar X =  \frac{1}{N} \sum_{i=1}^{N}X_i$


```python
df["price"].mean()
```




    853.6605839416059



##### 중간값 (중위수, median)
- 값들을 크기 순으로 정렬했을 때 중간에 위치한 값
- 값이 짝수개인 경우, 가운데 두 값의 평균
- 히스토그램으로 그리면 쉽게 볼 수 있음


```python
df["price"].median()
```




    805.0




```python
_ = sns.histplot(data=df, x="price", bins=30)
```


    
![png](static_files/static_8_0.png)
    


##### 최빈값 (mode)
- 가장 많은 사례에서 관찰된 값
- 연속형 변수보다는 **범주형** 변수에서 유용
- 연속형 변수에서는 **구간**을 나누어 최빈값을 구하는 경우가 많음


```python
df["model"].value_counts()
```




    Avante    205
    K3         69
    Name: model, dtype: int64



#### 분위수 (quantile)
- 데이터에서 값의 순위 0 ~ 1를 표현 (100을 곱해 percentile로 표현하기도 함)
  - 최소값 = 0.0 = 0%
  - 중간값 = 0.5 = 50%
  - 최대값 = 1.0 = 100%


```python
# 최대/최소
df["price"].min(), df["price"].max()
```




    (190, 1820)




```python
# 분위수
df["price"].quantile(.25)
```




    620.0



##### 사분위수 (quartile)
- 데이터를 4등분하는 위치
  - 제 1사분위수 = $\frac{1}{4}$ 지점 = 25%
  - 제 2사분위수 = $\frac{2}{4}$ 지점 = 50% = 중간값
  - 제 3사분위수 = $\frac{3}{4}$ 지점 = 75%
  - 제 4사분위수 = $\frac{4}{4}$ 지점 = 100%
  - ...

#### 변산성 변수
##### 범위 (range)
- 최대 - 최소값
- 극단값이 있으면 커짐

##### 사분위간 범위 (IQR, InterQuartile Rane)
- 제 3사분위수 - 제 1사분위수 or 75% - 25%
- 극단값은 최소값 or 최대값 근처에 있으므로 극단값의 영향이 적음

##### 상자 그림 (box plot)
- 제 1사분위수 ~ 제 3사분위수를 상자로 표현
- 중간값은 상자의 가운데 굵은 선으로 표시
- 최소/최대값은 수염(whisker)으로 표시
  - 수염의 길이는 IQR의 1.5배까지, 넘어가는 경우 점으로 표시


```python
_ = sns.boxplot(data=df, y="price")
```


    
![png](static_files/static_18_0.png)
    



```python
_ = sns.boxplot(data=df, x="model", y="price")
```


    
![png](static_files/static_19_0.png)
    


##### 편차 (deviation)
- 값 - 평균  
$X_i - \bar X$

##### 분산 (variance)  
$Var(X) = \frac{1}{N}\sum_{i=1}^{N}(X_i -\bar X)^2$
- 편차 제곱의 평균
- 직관적이지는 않지만 수학적으로 중요한 성질이 있음
- 표준편차(standard deviation), $SD(X) = \sqrt{Var(X)}$를 많이 사용
- 평균 절대 편자(MAD, Mead Abs Divation),  
  - $MAD = \frac{1}{N}\sum abs(X_i-\bar X)$


```python
# 분산
df["price"].var()
```




    110631.49243335734




```python
# 표준 편차
df["price"].std()
```




    332.6131272715455



### 추론 통계
- 모집단(population): 연구의 관심 대상이 되는 집단 전체
- 표본(sample): 특정한 연구에서 선택된 모집단의 부분 집합
- 표집(sampling): 모집단에서 표본을 추출하는 절차
- 표본 통계량을 일반화하여 모집단에 대해 추론하는 것

#### 모수와 통계량
- 모수(popluation parameter): 모집단의 특성을 기술하는 양
- 통계량(sample statistic): 표본에서 얻어진 수로 계산한 값 (=통계치)
- 통계량으로부터 모수를 추정

#### 추정 (estimation)
- 통계량으로부터 모수를 추측하는 절차
  - 점추정 (point): 가장 가능성이 높은 모수 하나를 추정
  - 구간추정 (interval): 모수가 있을 가능성이 높은 범위를 추정

#### 표집 오차 (sampling err)
- 모집단과 표본의 차이 (확률적인 현상임)
  - 동일한 모집단에서 동일한 절차를 거쳐 추출한 표본끼리도 차이가 존재
- 표본의 크기가 클수록 표집 오차는 작아짐

#### 신뢰구간 (confidence interval)
- 신뢰구간 = 표본 통계량 $\pm$ 오차 범위 
- 대표적인 구간 추정 방법
- 모수의 근처 $\pm$ 오차 범위 내에 있을 확률이 높음 -> 표본 통계량에서 $\pm$ 오차 범위 내에 모수가 있을 확률이 높음
- 영향을 주는 요소
  - 신뢰수준 $\downarrow$
  - 표본의 변산성 $\downarrow$
  - 표본의 크기 $\uparrow$

##### 신뢰구간을 구하는 방법
- 일정한 수학적 가정으로부터 표집 분포를 유도
  - 평균의 경우, 중심극한정리를 이용
- 부트스트래핑(bootstrapping): 현재 가진 표본에서 재표집을 반복해, 표집 오차를 시뮬레이션

##### 중심극한정리 (Central Limit Theorem)
- 표본이 클 수록, 표본 평균의 분포가 정규 분포에 수렴
  - 평균=$\mu$, 표준편차=$\sigma$인 분포에서, 서로 독립이며 동일한 분포를 따르는 무작위 표본을 뽑았을 때 표본의 크기가 n일 경우,
  - 표본 평균의 표집 분포는 평균=$\mu$이고 표준편차가 $\frac{\sigma}{\sqrt{n}}$인 정규분포를 따름
- 모집단의 분포에 무관
- 표본의 분포가 정규 분포에 수렴하는 것이 아님


```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, norm
x = np.linspace(-.5, 1.5, 1000)
p = uniform.pdf(x)
_ = plt.plot(x, p)
```


    
![png](static_files/static_30_0.png)
    


##### 표본 분포
대체로 모집단의 분포(=균등분포)와 비슷


```python
# 균등 분포에서 데이터 100개를 무작위 추출
x = uniform.rvs(size=100)
_ = plt.hist(x)
```


    
![png](static_files/static_32_0.png)
    


##### 표집 분포 (sampling distribution)


```python
# 64개의 데이터를 뽑아 평균 내기를 1000회 반복 -> 1000개의 표본과 표본 평균
m = [uniform.rvs(size=64).mean() for _ in range(1000)]
_ = plt.hist(m)
_ = plt.xlim(0, 1)
```


    
![png](static_files/static_34_0.png)
    


##### 부트스트래핑


```python
from scipy.stats import bootstrap
```


```python
bootstrap([df["price"]], np.mean, confidence_level=.95)
```




    BootstrapResult(confidence_interval=ConfidenceInterval(low=813.9817518248175, high=893.4647838446508), standard_error=20.19969976807167)



#### 신뢰수준 (confidence level)
- 신뢰구간을 무한히 넓게 예측하면, 100% 모수를 포함 (신뢰수준 100%)
- 예측을하는 이유는 행동을 하거나 결정을하기 위해서 -> 넓은 구간의 예측은 쓸모가 없음
- 신뢰구간을 좁게 예측하면, 모수를 포함하지 못하는 경우가 생김
- 신뢰구간을 극단적으로 좁히면, 신뢰수준이 지나치게 낮아 쓸모가 없음
- 95% ,99% 등 일정한 신뢰수준으로 타협해 사용함

#### 유의수준 (significance level)
- 유의수준 = 100% - 신뢰수준
- 추정한 구간이 모수를 포함하지 못할 확률

#### 혼동 주의
- 일상적 표현에서는 신뢰도가 높으면 (측정)오차가 적음 -> 측정의 관점
- 통계에서 신뢰수준이 높으면 오차범위가 넓음 -> 추론의 관점

#### Student의 t 분포 (중심 극한 정리의 연장선)
- 통계학자 윌리엄 고셋이 발견한 확률 분포
- 모분산을 알면, 평균의 신뢰구간을 구할 때 중심 극한 정리에 따라 정규분포를 사용
- 모분산을 모르면, t 분포를 사용
- 표본의 크기가 충분히 크면, t 분포는 정규분포에 근사

##### 충분히 크면 (sufficiently large)
- 어떤 변수 n에 따라 참 거짓이 변하는 명제 P(n)가 있을 때,
  - n > N에서 P(n)이 항상 참인 어떤 N이 존재하는 경우
  - n이 충분히 크면 참인 명제: $\frac{1}{n^2}<0.0001$
  - n이 충분히 크면 거짓인 명제: $\sin{n}<0.5$
  - n의 크기보다 점점 특정한 결론으로 수렴하는 형태에 관한 표현


```python
# Python 통계 분석 라이브러리
import pingouin as pg
```

##### 단일표본 t 검정


```python
pg.ttest(df["price"], 0)
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
      <th>T</th>
      <th>dof</th>
      <th>alternative</th>
      <th>p-val</th>
      <th>CI95%</th>
      <th>cohen-d</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T-test</th>
      <td>42.483582</td>
      <td>273</td>
      <td>two-sided</td>
      <td>2.486212e-122</td>
      <td>[814.1, 893.22]</td>
      <td>2.566527</td>
      <td>2.773e+118</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pg.ttest(df["price"], 0, confidence=.99)
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
      <th>T</th>
      <th>dof</th>
      <th>alternative</th>
      <th>p-val</th>
      <th>CI99%</th>
      <th>cohen-d</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>T-test</th>
      <td>42.483582</td>
      <td>273</td>
      <td>two-sided</td>
      <td>2.486212e-122</td>
      <td>[801.537871688669, 905.7832961945428]</td>
      <td>2.566527</td>
      <td>2.773e+118</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### 통계적 가설 검정 (statistical hypothesis testing) -> 반증주의
1. 유의 수준 결정
2. 귀무가설 설정 (기각할(nullif) 가설)
3. p값: 귀무가설의 모수를 포함할 때까지 신뢰구간을 넓혔을 떄의 유의 수준

p값 < 유의수준: 귀무가설을 기각 -> 통계적으로 유의하다  
- 상관이 없는데, 우연히 이렇게 나올 가능성은 너무 작다. 그러니깐 상관이 있는걸로 인정하겠다 

##### 실증주의와 반증주의
- 과학의 가설 검정은 실증주의이지만, 통계적 가설 검정은 반증주의임  
- 반증 가능성
  - 틀릴 가능성
  - 버티고 있으면 좋은 이론임 

##### 통계적 유의함의 의미
- 유의수준 내에서 귀무가설을 기각할 수 있을만큼의 증거가 있음
- 표본의 크기가 충분하다는 의미로 이해
- 현실적으로 유의미한것은 아님

##### 통계적 가설 검정에서 오류의 종류
- 1종 오류 (False Alarm): 귀무가설이 참일 때 기각
  - 유의수준만큼 발생
- 2종 오류 (Miss): 귀무가설이 거짓이나 기각 못함

동일 조건에서 유의 수준을 낮추면 1종 오류$\downarrow$, 2종 오류$\uparrow$

### 상관 분석

#### 상관계수 (correlation coefficient)
- 두 변수의 연관성을 -1 ~ +1 범위의 수치로 나타낸 것
- 두 변수의 연관성을 파악하기 위해 사용 (두 변수의 관계의 강함, 절대값이 클 수록 관계가 강함)
- 본격적인 분석 전에 탐색적 분석을 위해 많이 사용
- 데이터 진단에도 사용(설문 등의 경우 관련 문항 간에는 높은 상관 관계가 있어야 함)

##### 상관계수의 해석
- 부호
  - +: 두 변수가 같은 방향으로 변화
  - -: 두 변수가 반대 방향으로 변화
- 크기
  - 0: 두 변수가 독립, 한 변수의 변화로 다른 변수의 변화를 예측하지 못함
  - 1: 한 변수의 변화와 다른 변수의 변화가 정확히 일치
    - 낮음: ~0.1
    - 중간: 0.1~0.5
    - 높음: 0.5~

엄밀한 근거에 바탕을 둔 것은 아니며, 실제 의사결저에서는 상대적으로 비교하는 것이 바람직

##### 피어슨 적률 상관계수  
$p(X,Y) = {cov(X,Y) \over \sigma X \sigma Y}$
- 가장 대표적인 상관계수
- 선형적인 상관계수를 측정
- 공분산을 두 변수의 표준편차로 나눔

#### 상관계수와 비단조적 관계
- 상관계수는 우상향 또는 우하향하는 단조적 관계를 표현
- 복잡한 비단조적 관계는 잘 나타내지 못함
- 상관계수가 낮다고 해서 관계가 없는 것은 아님


```python
pg.corr(df["price"], df["mileage"])
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
      <th>n</th>
      <th>r</th>
      <th>CI95%</th>
      <th>p-val</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pearson</th>
      <td>274</td>
      <td>-0.67616</td>
      <td>[-0.74, -0.61]</td>
      <td>5.809388e-38</td>
      <td>5.069e+34</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.scatterplot(data=df, x="mileage", y="price")
```


    
![png](static_files/static_52_0.png)
    



```python
pg.corr(df["price"], df["year"])
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
      <th>n</th>
      <th>r</th>
      <th>CI95%</th>
      <th>p-val</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pearson</th>
      <td>274</td>
      <td>0.828908</td>
      <td>[0.79, 0.86]</td>
      <td>1.388002e-70</td>
      <td>1.004e+67</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.scatterplot(data=df, x="year", y="price")
```


    
![png](static_files/static_54_0.png)
    


피어슨 상관계수
- r: 상관계수
- CI95%: 95% 신뢰 구간
- p-value: 가설 검정을 위한 p값

#### 상관계수의 신뢰구간
- `+~+`: 모집단에서 두 변수의 관계가 `+`
- `-~+`: 모집단에서 두 변수의 관계는 `-, 0, +` 모두 가능
- `-~-`: 모집단에서 두 변수의 관계가 `-`

#### 상관계수의 가설 검정
1. 모집단에서 두 변수가 상관 관계가 없다 -> (`corr=0`)이라는 귀무가설을 수립
2. 귀무가설이 참일 때, 관찰된 통계량 이상의 결과가 관찰될 가능성 (p값)을 계산
3. p값과 유의수준을 비교
   1. p < 유의수준: 귀무가설을 기각하고 모집단에서 두 변수가 상관관계가 있다고 결론
   2. p >= 유의수준: 결론을 유보 (관련이 있을 수도 없을 수도 있음)

##### p-value
- 증거의 부족함
- 0~1 범위, 일반적으로 0.05를 기준으로 함 (유의수준 5%)
- `p<.05`이면, 기울기가 `+` 또는 `-`라고 결론 내릴 수 있음

#### 연습문제


```python
sp = pd.read_excel("data/sp500_gold.xlsx")
pg.corr(sp["SPX"], sp["GLD"])
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
      <th>n</th>
      <th>r</th>
      <th>CI95%</th>
      <th>p-val</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pearson</th>
      <td>194</td>
      <td>0.466896</td>
      <td>[0.35, 0.57]</td>
      <td>6.778106e-12</td>
      <td>1.261e+09</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



1. SPX와 GLD의 상관 계수  
    `0.466896`
2. 95% 신뢰 구간  
    `[0.35,0.57]` (모집단에서 무한히 관측했을 경우 나오는 범위)  
    모집단에서 어쨌든 `+`가 나온다
3. 유의수준 5%에서 가설 검정  
    p-value가 0.05(`6.778106e-12`) 보다 작다  
    -> 통계적으로 유의한 상관이 있다 (유의수준 5%)


```python
_ = sns.scatterplot(data=sp, x="SPX", y="GLD")
```


    
![png](static_files/static_62_0.png)
    


### 회귀 분석

#### 지도학습 (supervised learning)
- 독립 변수 x를 이용해, 종속 변수 y를 예측하는 것
  - 독립 변수(independent var): 예측의 바탕이 되는 정보, 인과관계에서 원인, 입력값
  - 종속 변수(dependent var): 예측의 대상, 인과관계에서 결과, 출력값

##### 종속 변수의 종류에 따른 구분
- 회귀분석(regression): 종속변수가 **연속** (예측-실제가 작은것이 중요)
- 분류분석(classification): 종속변수가 **범주형** (예측과 실제가 맞는것이 중요)

#### 선형 모델 (linear model)  
$\hat{y} = wx + b$
- $\hat{y}$: y의 예측치
- x: 독립변수
- w: 가중치 또는 기울기
- b: 절변 (x=0일때, y의 예측치)

#### 잔차 (residual)    
$r = y-\hat{y}$
- 실제값 y과 예측값 $\hat{y}$의 차이

##### 잔차분산  
$\frac{1}{N}\sum{(y-\hat{y})^2}$
- 잔차를 제곱하여 평균낸 것
- 잔차분산이 크다 -> 예측이 잘 맞지 않음
- 잔차분산이 작다 -> 예측이 잘 맞음

#### 최소제곱법 (Ordinary Least Squares)
- 잔차분산이 최소가 되게하는 w, b등 계수를 추정
- 가장 널리사용되는 추정방법

`price ~ mileage` = $price = w \times mileage + b$


```python
# 회귀분석
from statsmodels.formula.api import ols

ols("price ~ mileage", data=df).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.457</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.455</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   229.1</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th> <td>5.81e-38</td>
</tr>
<tr>
  <th>Time:</th>                 <td>15:39:50</td>     <th>  Log-Likelihood:    </th> <td> -1895.7</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   274</td>      <th>  AIC:               </th> <td>   3795.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   272</td>      <th>  BIC:               </th> <td>   3803.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> 1258.7668</td> <td>   30.599</td> <td>   41.137</td> <td> 0.000</td> <td> 1198.526</td> <td> 1319.008</td>
</tr>
<tr>
  <th>mileage</th>   <td>   -0.0052</td> <td>    0.000</td> <td>  -15.136</td> <td> 0.000</td> <td>   -0.006</td> <td>   -0.005</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.258</td> <th>  Durbin-Watson:     </th> <td>   1.101</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.879</td> <th>  Jarque-Bera (JB):  </th> <td>   0.108</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.032</td> <th>  Prob(JB):          </th> <td>   0.947</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.074</td> <th>  Cond. No.          </th> <td>1.83e+05</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.83e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
_ = sns.lmplot(data=df, x="mileage", y="price")
```


    
![png](static_files/static_70_0.png)
    


#### 회귀계수 추정 결과
- `Intercept`는 절편 b, 나머지는 각 변수의 계수
- 계수 추정 결과는, 아래 순
  - 추정치
  - 표준오차
  - t값
  - p-value
  - 신뢰구간
- 표준오차, t값은 p-value를 구하기 위한 중간 결과 -> 해석이 필요 없음

##### 회귀계수의 가설검정
- 모집단에서 기울기=0을 귀무가설로 p값 계산
  - p < 유의수준(통상 5%) -> 기울기 != 0으로 결론
  - p >= 유의수준 -> 결론을 유보


```python
new_df = {"mileage":[10000, 20000]}
new_df = pd.DataFrame(new_df)
```


```python
m.predict(new_df)
```




    0    1206.483684
    1    1154.200600
    dtype: float64



#### 결정 계수 또는 R제곱
- R제곱(R-Squared): 회귀분석에서 예측의 정확성을 알기 쉽게 표현한 지표 (0~1)
  - R제곱=0: 분석 결과가 y의 예측에 도움이 안됨
  - R제곱=1: y를 완벽하게 예측할 수 있음
- 단순회귀분석(독립변수가 1개인 회귀분석)의 경우, 회귀분석의 **R제곱=독립변수와 종속변수의 피어슨 상관계수**의 제곱

#### 독립 변수가 범주형인 경우
- 범주형 변수는 기울기를 곱할 수 없음
- 연속 변수로 변환하여 모형에 투임
- 여러 가지 방법이 있으나, 더미 코딩을 가장 많이 사용함

##### 더미 코딩 (dummy coding)
- 범주형 변수에 범주가 k개 있을 경우, k-1개의 더미 변수를 대신 투입
- 범주 중에 하나를 기준(reference)로 지정
- 기본적으로 ABC 순으로 먼저 나오는 것이 기준 (변경 가능)
- 기준을 제외한 범주들은 범주별로 더미 변수를 하나씩 가짐
- 더미변수는 해당 범주일 경우에만 고려
- 더미변수의 기울기는 기준과의 차이를 의미

##### 연습문제
독립변수: marriage  
종속변수: rating  


```python
hr = pd.read_excel("data/hr.xlsx")
hr.head(2)
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
      <th>department</th>
      <th>job_level</th>
      <th>marriage</th>
      <th>rating</th>
      <th>overtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sales</td>
      <td>Salaried</td>
      <td>single</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Engineering</td>
      <td>Hourly</td>
      <td>single</td>
      <td>4</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
ols("rating ~ marriage", data=hr).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>rating</td>      <th>  R-squared:         </th> <td>   0.007</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.006</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   9.848</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th>  <td>0.00173</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:20:06</td>     <th>  Log-Likelihood:    </th> <td> -2182.3</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1470</td>      <th>  AIC:               </th> <td>   4369.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  1468</td>      <th>  BIC:               </th> <td>   4379.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>          <td>    2.9208</td> <td>    0.040</td> <td>   72.678</td> <td> 0.000</td> <td>    2.842</td> <td>    3.000</td>
</tr>
<tr>
  <th>marriage[T.single]</th> <td>   -0.1751</td> <td>    0.056</td> <td>   -3.138</td> <td> 0.002</td> <td>   -0.284</td> <td>   -0.066</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>50.923</td> <th>  Durbin-Watson:     </th> <td>   1.980</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  26.234</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.127</td> <th>  Prob(JB):          </th> <td>2.01e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.397</td> <th>  Cond. No.          </th> <td>    2.67</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



1. single의 rating은 얼마로 예측?  
`2.9208 - 0.1751` = `2.7457`
2. married의 rating은 얼마로 예측?  
`2.9208`
3. 모집단에서 둘의 차이가 있다고 할 수 있나?  
미혼이 기혼보다 점수가 낮다

p-value < 0.05이므로 차이가 있다 -> 차이가 큰지 적은지를 의미하는것은 아님

#### 유의미하다
- 통계적: 현재 수준의 데이터로 식별할 만큼의 차이가 있다
- 주관적: 차이가 큰지 적은지와 같은 사람마다 기준이 다른것

혼돈의 여지가 있으므로 통계적으로 유의미하다는 아예 사용하지 않는것이 좋음

##### 범주가 2개인 경우


```python
df["model"].unique()
```




    array(['K3', 'Avante'], dtype=object)




```python
ols("price ~ model", data=df).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.011</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.007</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.039</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th>  <td>0.0824</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:41:33</td>     <th>  Log-Likelihood:    </th> <td> -1977.9</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   274</td>      <th>  AIC:               </th> <td>   3960.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   272</td>      <th>  BIC:               </th> <td>   3967.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>  833.4146</td> <td>   23.144</td> <td>   36.009</td> <td> 0.000</td> <td>  787.850</td> <td>  878.980</td>
</tr>
<tr>
  <th>model[T.K3]</th> <td>   80.3970</td> <td>   46.121</td> <td>    1.743</td> <td> 0.082</td> <td>  -10.402</td> <td>  171.196</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>13.893</td> <th>  Durbin-Watson:     </th> <td>   0.528</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  15.007</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.573</td> <th>  Prob(JB):          </th> <td>0.000551</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.002</td> <th>  Cond. No.          </th> <td>    2.48</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



- Avante가 기준, K3의 더미변수(`model[T.K3]`)가 추가  
- 예상가격  
  - Avante: 833  
  - K3: 833+80 = 913

##### 기준 범주 바꾸기
`ols("price ~ C(model, Treatment("K3"))", data=df).fit()`  
`C`함수로 변수를 범주형으로 지정하고, `Treatment`로 기준 범주를 지정

##### 범주가 3개인 경우


```python
dep = pd.read_excel("data/depression.xlsx")
dep.head(2)
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
      <th>y</th>
      <th>age</th>
      <th>x2</th>
      <th>x3</th>
      <th>TRT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
</div>



y: 치료 효과  
TRT: 치료 방법


```python
ols("y ~ TRT", data=dep).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.172</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.122</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.424</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th>  <td>0.0445</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>15:45:49</td>     <th>  Log-Likelihood:    </th> <td> -137.86</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    36</td>      <th>  AIC:               </th> <td>   281.7</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    33</td>      <th>  BIC:               </th> <td>   286.5</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   62.3333</td> <td>    3.359</td> <td>   18.557</td> <td> 0.000</td> <td>   55.500</td> <td>   69.167</td>
</tr>
<tr>
  <th>TRT[T.B]</th>  <td>  -10.4167</td> <td>    4.750</td> <td>   -2.193</td> <td> 0.035</td> <td>  -20.081</td> <td>   -0.752</td>
</tr>
<tr>
  <th>TRT[T.C]</th>  <td>  -11.0833</td> <td>    4.750</td> <td>   -2.333</td> <td> 0.026</td> <td>  -20.748</td> <td>   -1.419</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.553</td> <th>  Durbin-Watson:     </th> <td>   1.488</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.758</td> <th>  Jarque-Bera (JB):  </th> <td>   0.544</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.267</td> <th>  Prob(JB):          </th> <td>   0.762</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.721</td> <th>  Cond. No.          </th> <td>    3.73</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



##### 연습문제
독립변수(x): Income  
종속변수(y): Consumption


```python
us = pd.read_excel("data/uschange.xlsx")
us.head()
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
      <th>Date</th>
      <th>Consumption</th>
      <th>Income</th>
      <th>Production</th>
      <th>Savings</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01</td>
      <td>0.615986</td>
      <td>0.972261</td>
      <td>-2.452700</td>
      <td>4.810312</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-04-01</td>
      <td>0.460376</td>
      <td>1.169085</td>
      <td>-0.551525</td>
      <td>7.287992</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-07-01</td>
      <td>0.876791</td>
      <td>1.553271</td>
      <td>-0.358708</td>
      <td>7.289013</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-10-01</td>
      <td>-0.274245</td>
      <td>-0.255272</td>
      <td>-2.185455</td>
      <td>0.985230</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1971-01-01</td>
      <td>1.897371</td>
      <td>1.987154</td>
      <td>1.909734</td>
      <td>3.657771</td>
      <td>-0.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
pg.corr(us["Income"], us["Consumption"])
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
      <th>n</th>
      <th>r</th>
      <th>CI95%</th>
      <th>p-val</th>
      <th>BF10</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pearson</th>
      <td>187</td>
      <td>0.398779</td>
      <td>[0.27, 0.51]</td>
      <td>1.577409e-08</td>
      <td>7.007e+05</td>
      <td>0.999922</td>
    </tr>
  </tbody>
</table>
</div>



1. Income과 Consumption의 상관 계수  
    `0.398779`
2. 95% 신뢰 구간  
    `[0.27,0.51]` (모집단에서 무한히 관측했을 경우 나오는 범위)  
    모집단에서 어쨌든 `+`가 나온다
3. 유의수준 5%에서 가설 검정  
    p-value가 0.05(`1.577409e-08`) 보다 작다  
    -> 통계적으로 유의한 상관이 있다 (유의수준 5%) -> 모집단에서 상관=0인데, 표본에서 이만큼 나올 확률은 낮음


```python
ols("Consumption ~ Income", data=us).fit().summary() # 종속 ~ 독립
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Consumption</td>   <th>  R-squared:         </th> <td>   0.159</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.154</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   34.98</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th> <td>1.58e-08</td>
</tr>
<tr>
  <th>Time:</th>                 <td>16:21:51</td>     <th>  Log-Likelihood:    </th> <td> -169.62</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   187</td>      <th>  AIC:               </th> <td>   343.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   185</td>      <th>  BIC:               </th> <td>   349.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>    0.5451</td> <td>    0.056</td> <td>    9.789</td> <td> 0.000</td> <td>    0.435</td> <td>    0.655</td>
</tr>
<tr>
  <th>Income</th>    <td>    0.2806</td> <td>    0.047</td> <td>    5.915</td> <td> 0.000</td> <td>    0.187</td> <td>    0.374</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.528</td> <th>  Durbin-Watson:     </th> <td>   1.696</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  29.145</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.454</td> <th>  Prob(JB):          </th> <td>4.69e-07</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.707</td> <th>  Cond. No.          </th> <td>    2.08</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



1. 절편  
`0.5451`
2. 기울기  
`0.2806` -> 수입이 1 증가하면, 소비는 0.28 증가하는 관계
3. 신뢰구간  
`Incomde: [0.187, 0.374]`
4. 가설검정 해석  
p<0.05 -> 기울기가 0이 아니다 -> 기울기가 있다 or 유의미하다

#### 다중회귀분석
- 독립변수가 2개 이상인 회귀분석
- `+`로 변수를 구분
  - `price ~ mileage + model`

#### 통계적 통제
- 독립 변수 x와 상관관계가 높은 요소 z가 존재할 경우, z가 종속변수 y에 미치는 영향이 x의 기울기에 간접 반영될 수 있음
- 실험적 통제: 데이터에서 z를 일정하게 유지하여 z의 영향을 제거
- 통계적 통제: z를 모형에 독립변수로 함께 포함하여 x의 기울기에 z의 영향이 간접 반영되지 않도록 함

#### 표준화 (standardization)  
${X - \mu} \over \sigma$
- 변수별로 퍼진 정도(분산)을 비슷하게 맞춰주는 절차
  - 표준화를 진행하면, 평균=0, 표준편차=1이 됨
  - 단위가 다른 변수의 기울기를 비교할 때 사용 -> 단위를 없애는 효과
- 관계식에서는 `y ~ scale(x1) + scale(x2)` 형식으로 사용
  - 범주형 변수는 표준화하지 않음

#### 과대적합 (overfitting)
- 최소제곱법(OLS)은 잔차분산이 가장 작은 계수를 추정
- 주어진 표본에 가장 맞는 계수를 찾게 됨
- 표집 오차가 존재하기 때문에, 주어진 표본에 지나치게 맞는데 계수를 추정하면 모집단의 계수와 다를 수 있음

#### 수정 R제곱과 AIC, BIC
- 독립변수의 개수가 다른 모형을 비교할 경우, R제곱으로는 비교가 어려움
  - R제곱은 독립변수가 많을수록 높아지는 경향이 있음
- 독립변수의 개수를 이론적으로 보정한 수정 R제곱, AIC, BIC 등의 지수 사용
  - 수정 R제곱(Adjusted R-Squared): R제곱을 보정 -> 클 수록 좋음
  - AIC와 BIC: 잔차분산을 보정 -> 작을 수록 좋음

##### 수정 R제곱 (Adjusted R-Squared)  
$Adjusted \space R^2 = 1 - {RSS/(n-k-1)  \over TSS/(n-1)}$  
n: 표본의 크기, k: 독립변수의 개수  

- 독립 변수를 추가하면 $R^2$이 작아지도록 보정
- 수정 R제곱이 클 수록 좋은 모형
- 모형 간 비교의 용도
  - 한 모형이 종속변수의 분산을 설명하는 비율을 볼 때는 R제곱을 봐야함

##### AIC (Akaike Information Criterion)  
$AIC = n \log ({RSS \over n}) + 2k$  
- 작을수록 좋은 모형

##### BIC (Bayesian Information Criterion)  
$BIC = n \log ({RSS \over n}) + k\log{n}$  
- 작을수록 좋은 모형

#### 교차 검증
- 수정 R제곱, AIC, BIC 등은 이론적 보정이므로 과적합을 정확히 반영 못함
  - 데이터가 충분히 많다면, 데이터를 여러 개의 셋으로 나누어 교차 검증
  - 한 데이터 셋의 분석 결과를 다른 데이터셋에 적용하여 예측 오차를 확인 (예측 오차가 적은 모형이 좋은 모형)
- 이론적 가정에 의존하지 않으므로, 데이터가 충분히 많을 때는 교차 검증을 권장

##### 교차 검증의 종류
- LpO CV (Leave-p-out)
  - p개를 제외한 모든 사례로 추정에 사용
  - p개는 가능한 모든 방법으로 조합
  - 조합이 지나치게 많아 비현실적
- LOOCV (Leave-one-out)
  - p=1인 경우, 데이터가 N개이면 N번 검증
- K-Fold: 데이터를 크게 k개의 셋으로 나누고, 한 셋 씩 테스트 셋으로 사용 (k번 교차검증)
- holdout: 데이터를 훈련 셋과 테스트 셋으로 한 번만 나누어 1회 교차 검증

##### 교차 검증의 결과
- 훈련 오차와 테스트 오차가 모두 높은 경우
  - 과소적합
  - 모형을 더 복잡하게 수정
- 훈련 오차와 테스트 오차가 모두 낮음 -> 이상적인 모형
- 훈련 오차는 낮고, 테스트 오차는 높은 경우
  - 과대적합
  - 모형을 더 단순하게 수정

#### 데이터 분할


```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # 원자료, 테스트 데이터의 비율
```

#### 변수의 변형
- 선형 모형은 독립 변수와 종속 변수의 선형적 관계를 가정한다는 한계
  - 독립 변수를 비선형 변환하면 일부 극복 가능 (R과 python은 관계식에 수학 함수를 사용하면 자동으로 변수 변환)

#### 로그 함수
- 오른쪽 위로 갈 수록 완만해지는 형태
- 데이터에 적용하면, 오른쪽을 왼쪽으로 끌어당기는 효과
- 독립 변수에 오른쪽으로 크게 떨여지 있는 값이 있는 경우, 로그 함수를 적용해 간격을 일정하게 만들 수 있음


```python
ols("price ~ np.log(mileage)", df).fit()
```




    <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x1f45d755340>



#### 함수
- 관계식에 덧셈, 곱셈, 거듭제곱 등을 할 경우 적용이 불가  
- `I`함수를 이용해 아래와 같은 계산이 가능
  - `y ~ I(x+z)`


$y = ax^2+bx+c$ 같은 모형을 관계식으로 만들 경우 -> `y ~ I(x**2)+x`

#### 절편이 없는 모형
$y = wx + 0$을 표시하기 위해, 관계식에 `0+`를 추가  
`y ~ 0 + x`

#### 절편의 이동
- 절편은 `x=0`일 때의 예측치
- 절편을 `x=100`일 때의 예측치로 바꾸려면, 일괄적으로 `x`에서 100을 빼면 됨  
- 분석 자체에는 영향이 없으나, 절편의 해석이 더 쉬워질 수 있음

`y ~ I(x - 100)`

#### 상호작용 (interaction)
- 상호작용 항: 두 독립변수의 곱으로 이뤄진 항 -> $y = x + m + xm$
- 관계식으로 쓸 때는 `:`을 사용 -> `y ~ x + m + x:m` -> `y ~ x*m` 같이 축약할 수 있음 (*의 일반적인 의미와 용법이 다름)

##### 상호작용의 해석
- x는 연속형, m은 이분 범주형이라고 할 때,
  - `y ~ x + m`: m에 따라 x의 절편이 바뀌는 것으로 해석
  - `y ~ x + x:m`: m에 따라 x의 기울기가 바뀌는 것으로 해석
  - `y ~ x + m + x:m`: m에 따라 x의 기울기와 절편이 바뀌는 것으로 해석

#### 증거의 사다리
인과관계의 증거 수준  
1. 실험적 통제
2. 무작위 대조군
3. 준실험
4. 반사실

##### 실험적 통제 (experimental control)
- 처치를 제외한 다른 모든 조건을 동일하게 유지
- 인과관계를 확인할 수 있는 최선의 조건
- 매우 한정된 조건에서만 가능
  
##### 무작위 대조군 (randomized controlled trials)
- 모든 조건을 완벽하게 통제할 수 없을 경우
- 실험군과 대조군에 무작위 할당
- 표집 오차가 있을 수 있음

##### 준실험 (quasi-experiment)
- 대조군이 없거나 무작위 할당을 하지 않았지만 실험과 비슷한 상황
- 자연적으로 무작위 할당과 비슷한 결과가 생긴 경우

##### 반사실 (counterfactual)
- 순수한 관찰 결과만을 가지고 인과관계를 추측
- 어떤 일이 벌어지지 않았을 때 일어날 일을 예측하는 모형이 필요
- 모형의 예측과 실제의 결과를 비교하여 인과관계를 추론

#### 횡단 비교와 종단 비교
- 횡단 비교(cross-sectional): 동일 시점에서 다른 대상이나 집단을 비교
- 종단 비교(longitudinal): 동일 대상을 다른 시점 간 비교

#### 이중차분법 (Difference-in-Differences)
- 실험이 불가능한 상황에서 사용하는 준실험적 방법
- 실험군 B에 어떤 처치를 했으나 대조군이 없을 때
- 실험군과 비슷한 집단 A를 이용해 비교
- $d = (B_2 - B_1) - (A_2 - A_1)$

##### 결과 해석
- d=0: 실험군 B에서 변화는 대조군 A에서 변화와 비슷 (처치 효과 없음)
- d!=0: 실험군 B에서 대조군 A와 다른 변화를 관찰 (처치 효과 있음)

#### 평행 추세의 가정 (parallel trend assumption)
- 처치 효과가 없다면 실험군과 대조군이 비슷하게 변할 것이라는 가정
  - 가정이 성립하지 않으면, 이중차분법의 결과는 무의미
- 가능한 비슷한 A와 B를 비교하는것이 중요

#### 회귀분석을 통한 이중차분법
- 상호작용을 이용
- $y = a \cdot GROUP + b \cdot POINT + d \cdot (GROUP \times POINT) + e$


```python
nj = pd.read_excel("data/njmin3.xlsx")
nj.head(2)
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
      <th>CO_OWNED</th>
      <th>SOUTHJ</th>
      <th>CENTRALJ</th>
      <th>PA1</th>
      <th>PA2</th>
      <th>DEMP</th>
      <th>nj</th>
      <th>bk</th>
      <th>kfc</th>
      <th>roys</th>
      <th>wendys</th>
      <th>d</th>
      <th>d_nj</th>
      <th>fte</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>12.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6.5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>



`fte`=전일제 환산 고용률, `nj`=뉴저지(1)/펜실베니아(0), `d`=최저임금 인상 전(0)/후(1)


```python
ols("fte ~ nj + d + nj:d", data=nj).fit().summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>fte</td>       <th>  R-squared:         </th> <td>   0.007</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.004</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1.964</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 03 Jun 2022</td> <th>  Prob (F-statistic):</th>  <td> 0.118</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>17:03:35</td>     <th>  Log-Likelihood:    </th> <td> -2904.2</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   794</td>      <th>  AIC:               </th> <td>   5816.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   790</td>      <th>  BIC:               </th> <td>   5835.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   23.3312</td> <td>    1.072</td> <td>   21.767</td> <td> 0.000</td> <td>   21.227</td> <td>   25.435</td>
</tr>
<tr>
  <th>nj</th>        <td>   -2.8918</td> <td>    1.194</td> <td>   -2.423</td> <td> 0.016</td> <td>   -5.235</td> <td>   -0.549</td>
</tr>
<tr>
  <th>d</th>         <td>   -2.1656</td> <td>    1.516</td> <td>   -1.429</td> <td> 0.154</td> <td>   -5.141</td> <td>    0.810</td>
</tr>
<tr>
  <th>nj:d</th>      <td>    2.7536</td> <td>    1.688</td> <td>    1.631</td> <td> 0.103</td> <td>   -0.561</td> <td>    6.068</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>218.742</td> <th>  Durbin-Watson:     </th> <td>   1.842</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 804.488</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 1.268</td>  <th>  Prob(JB):          </th> <td>2.03e-175</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.229</td>  <th>  Cond. No.          </th> <td>    11.3</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


