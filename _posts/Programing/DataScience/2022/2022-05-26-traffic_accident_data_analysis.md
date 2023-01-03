---
title: "교통사고 데이터 시각화"
date: 2022-05-26T04:40:47.697Z

categories:
  - Programming
  - DataScience
tags:
  - Plotly
  - Pandas
---

# Traffic Accident Data Analysis
데이터 선택을 잘못한거 + 지식 부족으로 원하는 결과를 도출해내지 못했음 -> 망한 프로젝트
## 1. 소개
[공공 데이터 포털](https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=FILE&keyword=&detailKeyword=&publicDataPk=&recmSe=&detailText=&relatedKeyword=&commaNotInData=&commaAndData=&commaOrData=&must_not=&tabId=&dataSetCoreTf=&coreDataNm=&sort=inqireCo&relRadio=&orgFullName=&orgFilter=&org=&orgSearch=&currentPage=1&perPage=10&brm=&instt=&svcType=&kwrdArray=&extsn=&coreDataNmArray=&pblonsipScopeCode=)에서 제공하는 교통 사고 관련 데이터를 분석화하고 시각화 해본 프로젝트

## 2. 이용 데이터
- [도로교통공단_사고유형별 교통사고 통계](https://www.data.go.kr/data/15070282/fileData.do)
- [도로교통공단_가해운전자 연령층별 월별 교통사고 통계](https://www.data.go.kr/data/15070199/fileData.do)
- [도로교통공단_월별 교통사고 통계](https://www.data.go.kr/data/15070315/fileData.do)
- [도로교통공단_사상자 연령층별 성별 교통사고 통계](https://www.data.go.kr/data/15070293/fileData.do)

파일 이름을보면 2020년에 발생한 교통사고에 대한 자료인거 같다.

다양한 종류의 교통 사고 관련 데이터가 있지만, 위의 4개의 데이터를 이용할 예정  
공공포털에서 API를 제공하고 있지만, CSV 파일이 좀 더 이용하기 편해서 파일을 다운 받아 이용  
-> 이후에 API로 변경 될 수 있음

## 3. 목표
1. 각각의 데이터 셋에 대한 분석 및 시각화
2. 공통 항목을 가진 하나의 데이터 셋 생성
3. 종합된 데이터 셋에 대한 분석 및 시각화

## 4. 구현

### 1. 각 데이터 셋 분석 및 시각화


```python
import pandas as pd
import numpy as np
import plotly.express as px
import chart_studio as cs
```

#### 1. 사고 유형별 교통사고 통계


```python
df_accident_type = pd.read_csv("data/도로교통공단_사고유형별 교통사고 통계_20201231.csv", encoding="cp949")
```


```python
df_accident_type
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
      <th>사고유형대분류</th>
      <th>사고유형중분류</th>
      <th>사고유형</th>
      <th>사고건수</th>
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
      <th>부상신고자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>횡단중</td>
      <td>13147</td>
      <td>520</td>
      <td>6417</td>
      <td>6617</td>
      <td>517</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>차도통행중</td>
      <td>차도통행중</td>
      <td>3702</td>
      <td>195</td>
      <td>1423</td>
      <td>2022</td>
      <td>209</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>길가장자리구역통행중</td>
      <td>길가장자리구역통행중</td>
      <td>2079</td>
      <td>40</td>
      <td>591</td>
      <td>1371</td>
      <td>160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>보도통행중</td>
      <td>보도통행중</td>
      <td>2015</td>
      <td>26</td>
      <td>707</td>
      <td>1260</td>
      <td>129</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>기타</td>
      <td>기타</td>
      <td>15278</td>
      <td>288</td>
      <td>5195</td>
      <td>9290</td>
      <td>1272</td>
    </tr>
    <tr>
      <th>5</th>
      <td>차대차</td>
      <td>정면충돌</td>
      <td>정면충돌</td>
      <td>7494</td>
      <td>181</td>
      <td>3657</td>
      <td>8468</td>
      <td>677</td>
    </tr>
    <tr>
      <th>6</th>
      <td>차대차</td>
      <td>측면충돌</td>
      <td>측면충돌</td>
      <td>74742</td>
      <td>441</td>
      <td>19893</td>
      <td>88838</td>
      <td>6945</td>
    </tr>
    <tr>
      <th>7</th>
      <td>차대차</td>
      <td>후진중충돌</td>
      <td>후진중충돌</td>
      <td>3253</td>
      <td>4</td>
      <td>297</td>
      <td>4003</td>
      <td>154</td>
    </tr>
    <tr>
      <th>8</th>
      <td>차대차</td>
      <td>추돌</td>
      <td>추돌</td>
      <td>32993</td>
      <td>370</td>
      <td>7924</td>
      <td>48345</td>
      <td>3172</td>
    </tr>
    <tr>
      <th>9</th>
      <td>차대차</td>
      <td>기타</td>
      <td>기타</td>
      <td>46070</td>
      <td>303</td>
      <td>11000</td>
      <td>50973</td>
      <td>4764</td>
    </tr>
    <tr>
      <th>10</th>
      <td>차량단독</td>
      <td>전도</td>
      <td>전도</td>
      <td>1108</td>
      <td>90</td>
      <td>412</td>
      <td>466</td>
      <td>281</td>
    </tr>
    <tr>
      <th>11</th>
      <td>차량단독</td>
      <td>전복</td>
      <td>전복</td>
      <td>178</td>
      <td>26</td>
      <td>81</td>
      <td>98</td>
      <td>32</td>
    </tr>
    <tr>
      <th>12</th>
      <td>차량단독</td>
      <td>공작물충돌</td>
      <td>공작물충돌</td>
      <td>3233</td>
      <td>340</td>
      <td>1376</td>
      <td>1745</td>
      <td>668</td>
    </tr>
    <tr>
      <th>13</th>
      <td>차량단독</td>
      <td>주/정차차량 충돌</td>
      <td>주/정차차량 충돌</td>
      <td>19</td>
      <td>4</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>차량단독</td>
      <td>도로이탈</td>
      <td>도로이탈 추락</td>
      <td>413</td>
      <td>92</td>
      <td>185</td>
      <td>203</td>
      <td>46</td>
    </tr>
    <tr>
      <th>15</th>
      <td>차량단독</td>
      <td>도로이탈</td>
      <td>도로이탈 기타</td>
      <td>199</td>
      <td>30</td>
      <td>113</td>
      <td>113</td>
      <td>24</td>
    </tr>
    <tr>
      <th>16</th>
      <td>차량단독</td>
      <td>기타</td>
      <td>기타</td>
      <td>3728</td>
      <td>130</td>
      <td>1285</td>
      <td>2213</td>
      <td>543</td>
    </tr>
    <tr>
      <th>17</th>
      <td>철길건널목</td>
      <td>철길건널목</td>
      <td>철길건널목</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 사고 유형별 부상자 수
fig = px.bar(df_accident_type, x="사고유형", y=["사망자수", "중상자수", "경상자수", "부상신고자수"], title="사고유형별 부상자 수")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="사고 유형별 부상자 수", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/26.embed" height="525" width="100%"></iframe>



#### 2. 가해 운전자 연령층별 월별 교통사고 통계


```python
df_age_agg_driver_month = pd.read_csv("data/도로교통공단_가해운전자 연령층별 월별 교통사고 통계_12_31_2020.csv", encoding="cp949")
```


```python
df_age_agg_driver_month.sample(5)
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
      <th>가해자연령층</th>
      <th>발생월</th>
      <th>사고건수</th>
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
      <th>부상신고자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>65세이상</td>
      <td>7</td>
      <td>2648</td>
      <td>56</td>
      <td>808</td>
      <td>2783</td>
      <td>205</td>
    </tr>
    <tr>
      <th>87</th>
      <td>불명</td>
      <td>4</td>
      <td>240</td>
      <td>0</td>
      <td>38</td>
      <td>164</td>
      <td>66</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20세이하</td>
      <td>7</td>
      <td>709</td>
      <td>5</td>
      <td>200</td>
      <td>677</td>
      <td>129</td>
    </tr>
    <tr>
      <th>94</th>
      <td>불명</td>
      <td>11</td>
      <td>322</td>
      <td>0</td>
      <td>42</td>
      <td>216</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20세이하</td>
      <td>3</td>
      <td>521</td>
      <td>8</td>
      <td>160</td>
      <td>547</td>
      <td>90</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 가해자 연령별 사고 건수
fig = px.bar(df_age_agg_driver_month, x="가해자연령층", y="사고건수", title="가해자 연령별 사고 건수")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="가해자 연령별 사고 건수", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/28.embed" height="525" width="100%"></iframe>




```python
# 월별 사고 건수
fig = px.bar(df_age_agg_driver_month, x="발생월", y="사고건수", color="가해자연령층", title="월별 사고 건수")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="월별 사고 건수", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/30.embed" height="525" width="100%"></iframe>




```python
# 월별 사고건수
fig = px.line(df_age_agg_driver_month, x="발생월", y="사고건수", facet_col="가해자연령층", color="가해자연령층", title="가해 연령층 별 월별 사고건수")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="월별 사고건수", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/32.embed" height="525" width="100%"></iframe>



#### 3. 월별 교통사고 통계


```python
df_monthly = pd.read_csv("data/도로교통공단_월별 교통사고 통계_20201231.csv", encoding="cp949")
```


```python
df_monthly
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
      <th>발생월</th>
      <th>사고건수</th>
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
      <th>부상신고자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>16968</td>
      <td>277</td>
      <td>5063</td>
      <td>18696</td>
      <td>1683</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>16285</td>
      <td>222</td>
      <td>4463</td>
      <td>17530</td>
      <td>1565</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>14949</td>
      <td>239</td>
      <td>4356</td>
      <td>16155</td>
      <td>1312</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>16186</td>
      <td>213</td>
      <td>4844</td>
      <td>17459</td>
      <td>1478</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18184</td>
      <td>254</td>
      <td>5470</td>
      <td>19563</td>
      <td>1758</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>19017</td>
      <td>254</td>
      <td>5614</td>
      <td>20436</td>
      <td>1830</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>18302</td>
      <td>240</td>
      <td>5207</td>
      <td>19933</td>
      <td>1829</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>17616</td>
      <td>295</td>
      <td>5123</td>
      <td>19410</td>
      <td>1731</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>18124</td>
      <td>301</td>
      <td>5194</td>
      <td>19001</td>
      <td>1644</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>19255</td>
      <td>309</td>
      <td>5554</td>
      <td>20900</td>
      <td>1736</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>18438</td>
      <td>261</td>
      <td>5295</td>
      <td>19832</td>
      <td>1571</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>16330</td>
      <td>216</td>
      <td>4381</td>
      <td>17121</td>
      <td>1457</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 월별 사고 건수
fig = px.bar(df_monthly, x="발생월", y="사고건수", title="월별 사고 건수")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="월별 사고 건수", auto_open=False))
```






 <iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/30.embed" height="525" width="100%"></iframe>



#### 4. 사상자 연령층별 성별 교통사고 통계


```python
df_age_gender = pd.read_csv("data/도로교통공단_사상자 연령층별 성별 교통사고 통계_20201231.csv", encoding="cp949")
```


```python
df_age_gender
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
      <th>사상자연령층</th>
      <th>사상자성별</th>
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
      <th>부상신고자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12세이하</td>
      <td>남</td>
      <td>12</td>
      <td>735</td>
      <td>4597</td>
      <td>637</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12세이하</td>
      <td>여</td>
      <td>12</td>
      <td>393</td>
      <td>3681</td>
      <td>457</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13-20세</td>
      <td>남</td>
      <td>92</td>
      <td>2053</td>
      <td>8108</td>
      <td>1476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13-20세</td>
      <td>여</td>
      <td>11</td>
      <td>784</td>
      <td>4132</td>
      <td>370</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21-30세</td>
      <td>남</td>
      <td>216</td>
      <td>5718</td>
      <td>29486</td>
      <td>3068</td>
    </tr>
    <tr>
      <th>5</th>
      <td>21-30세</td>
      <td>여</td>
      <td>45</td>
      <td>2266</td>
      <td>15676</td>
      <td>1042</td>
    </tr>
    <tr>
      <th>6</th>
      <td>31-40세</td>
      <td>남</td>
      <td>196</td>
      <td>6214</td>
      <td>28270</td>
      <td>2258</td>
    </tr>
    <tr>
      <th>7</th>
      <td>31-40세</td>
      <td>여</td>
      <td>25</td>
      <td>2100</td>
      <td>14765</td>
      <td>884</td>
    </tr>
    <tr>
      <th>8</th>
      <td>41-50세</td>
      <td>남</td>
      <td>264</td>
      <td>6597</td>
      <td>25066</td>
      <td>2059</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41-50세</td>
      <td>여</td>
      <td>52</td>
      <td>3045</td>
      <td>15677</td>
      <td>902</td>
    </tr>
    <tr>
      <th>10</th>
      <td>51-60세</td>
      <td>남</td>
      <td>426</td>
      <td>7213</td>
      <td>24731</td>
      <td>2164</td>
    </tr>
    <tr>
      <th>11</th>
      <td>51-60세</td>
      <td>여</td>
      <td>124</td>
      <td>4842</td>
      <td>16626</td>
      <td>944</td>
    </tr>
    <tr>
      <th>12</th>
      <td>61-64세</td>
      <td>남</td>
      <td>200</td>
      <td>2674</td>
      <td>8441</td>
      <td>698</td>
    </tr>
    <tr>
      <th>13</th>
      <td>61-64세</td>
      <td>여</td>
      <td>63</td>
      <td>1956</td>
      <td>4851</td>
      <td>317</td>
    </tr>
    <tr>
      <th>14</th>
      <td>65세이상</td>
      <td>기타/불명</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>65세이상</td>
      <td>남</td>
      <td>852</td>
      <td>7380</td>
      <td>14103</td>
      <td>1659</td>
    </tr>
    <tr>
      <th>16</th>
      <td>65세이상</td>
      <td>여</td>
      <td>490</td>
      <td>6588</td>
      <td>7794</td>
      <td>623</td>
    </tr>
    <tr>
      <th>17</th>
      <td>불명</td>
      <td>기타/불명</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>불명</td>
      <td>남</td>
      <td>0</td>
      <td>4</td>
      <td>14</td>
      <td>17</td>
    </tr>
    <tr>
      <th>19</th>
      <td>불명</td>
      <td>여</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 사상자 연령층
fig = px.bar(df_age_gender, x="사상자연령층", y=["사망자수", "중상자수", "경상자수", "부상신고자수"], color="사상자성별", title="사상자 연령층")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="사상자 연령층", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/35.embed" height="525" width="100%"></iframe>



### 2. 공통 항목을 가진 하나의 데이터 셋 생성
월별 교통사고 통계를 기본 데이터로해서, 다른 데이터들을 연관 데이터로 사용하면 좋을꺼 같다는 생각으로 4가지 데이터를 이용했다.  


```python
df_monthly[df_monthly["발생월"]==1]["사고건수"] == df_age_agg_driver_month[df_age_agg_driver_month["발생월"]==1]["사고건수"].sum()
```




    0    True
    Name: 사고건수, dtype: bool




```python
df_accident = pd.merge(df_monthly["발생월"], df_age_agg_driver_month, how="inner", on="발생월")
```


```python
df_accident
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
      <th>발생월</th>
      <th>가해자연령층</th>
      <th>사고건수</th>
      <th>사망자수</th>
      <th>중상자수</th>
      <th>경상자수</th>
      <th>부상신고자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20세이하</td>
      <td>452</td>
      <td>5</td>
      <td>137</td>
      <td>456</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>21-30세</td>
      <td>2260</td>
      <td>26</td>
      <td>661</td>
      <td>2616</td>
      <td>297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>31-40세</td>
      <td>2620</td>
      <td>46</td>
      <td>709</td>
      <td>2982</td>
      <td>278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>41-50세</td>
      <td>3057</td>
      <td>41</td>
      <td>945</td>
      <td>3351</td>
      <td>290</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>51-60세</td>
      <td>4218</td>
      <td>58</td>
      <td>1281</td>
      <td>4739</td>
      <td>332</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>12</td>
      <td>41-50세</td>
      <td>3020</td>
      <td>27</td>
      <td>875</td>
      <td>3145</td>
      <td>257</td>
    </tr>
    <tr>
      <th>92</th>
      <td>12</td>
      <td>51-60세</td>
      <td>3990</td>
      <td>54</td>
      <td>1078</td>
      <td>4229</td>
      <td>291</td>
    </tr>
    <tr>
      <th>93</th>
      <td>12</td>
      <td>61-64세</td>
      <td>1385</td>
      <td>22</td>
      <td>380</td>
      <td>1435</td>
      <td>91</td>
    </tr>
    <tr>
      <th>94</th>
      <td>12</td>
      <td>65세이상</td>
      <td>2266</td>
      <td>48</td>
      <td>648</td>
      <td>2361</td>
      <td>171</td>
    </tr>
    <tr>
      <th>95</th>
      <td>12</td>
      <td>불명</td>
      <td>298</td>
      <td>0</td>
      <td>33</td>
      <td>209</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
<p>96 rows × 7 columns</p>
</div>



월별 통계에서 발생월과 가해자 통계를 합쳤다.  
기존 df_monthly에도 사고 건수와 사망자수, 중상자 수 등의 데이터가 있지만 세분화 되어있지는 않기에 세분화 시킨 데이터를 가져갈 생각이다.

`df_monthly`와 `df_age_agg_dirver_month`는 월별로 통계가 나와있어 데이터를 합치기에 유리했지만, `df_age_gender`와 `df_accident_type`은 월별 통계가 아니라 합치지 못했다.  
이들을 함께 이용하기에는 각 데이터간의 공통점을 당장에는 찾을 수가 없어, **월별 교통사고 가해자 연령층** 밖에 생성할 수 없었다.

### 3. 종합된 데이터 셋에 대한 분석 및 시각화


```python
# 월별 사고 건수
fig = px.bar(df_accident, x="발생월", y="사고건수", title="2020년 교통사고 발생 추이")
fig.show()

cs.tools.get_embed(cs.plotly.plot(fig, filename="월별 사고 건수", auto_open=False))
```






<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~nuyhc/30.embed" height="525" width="100%"></iframe>


