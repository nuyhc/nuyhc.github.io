---
title: "아파트 분양 가격 동향 EDA 및 시각화"
date: 2022-06-02T14:26:23.147Z
img_path: /nuyhc.github.io/assets/img
categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Seaborn
  - matplot
---

# 전국 신규 민간 아파트 분양 가격 동향
## 소개
2013년부터 최근까지 부동산 가격 변동 추세가 아파트 분양가에 반양이되는지 확인해본 프로젝트
## 데이터셋
[공공데이터 포털](https://www.data.go.kr/data/15061057/fileData.do)
- 전국 평균 분양가격 (2013년 9월 ~ 2015년 8월)
- 주택도시보증공사 전국 평균 분양가격 (2015년 ~ 2019년 12월)
## 구현
### 1. 라이브러리


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
def get_font_family():
    import platform
    system_name = platform.system()

    if system_name == "Darwin" :
        font_family = "AppleGothic"
    elif system_name == "Windows":
        font_family = "Malgun Gothic"
    else:
        !apt-get install fonts-nanum -qq  > /dev/null
        !fc-cache -fv

        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManager.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont
        
        font_family = "NanumBarunGothic"
    return font_family

plt.rc("font", family=get_font_family())
plt.rc("axes", unicode_minus=False)
```

### 2. Data Load


```python
import glob
glob.glob("data/*")
```




    ['data\\전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv',
     'data\\주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20210930.csv']




```python
df_first = pd.read_csv("data/전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_last = pd.read_csv("data/주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20210930.csv", encoding="cp949")

print(f"2013~2018 데이터: {df_first.shape}\n2015~2021 데이터: {df_last.shape}")
```

    2013~2018 데이터: (17, 22)
    2015~2021 데이터: (6120, 5)
    


```python
df_first.head()
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
      <th>지역</th>
      <th>2013년12월</th>
      <th>2014년1월</th>
      <th>2014년2월</th>
      <th>2014년3월</th>
      <th>2014년4월</th>
      <th>2014년5월</th>
      <th>2014년6월</th>
      <th>2014년7월</th>
      <th>2014년8월</th>
      <th>...</th>
      <th>2014년11월</th>
      <th>2014년12월</th>
      <th>2015년1월</th>
      <th>2015년2월</th>
      <th>2015년3월</th>
      <th>2015년4월</th>
      <th>2015년5월</th>
      <th>2015년6월</th>
      <th>2015년7월</th>
      <th>2015년8월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>18189</td>
      <td>17925</td>
      <td>17925</td>
      <td>18016</td>
      <td>18098</td>
      <td>19446</td>
      <td>18867</td>
      <td>18742</td>
      <td>19274</td>
      <td>...</td>
      <td>20242</td>
      <td>20269</td>
      <td>20670</td>
      <td>20670</td>
      <td>19415</td>
      <td>18842</td>
      <td>18367</td>
      <td>18374</td>
      <td>18152</td>
      <td>18443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>8111</td>
      <td>8111</td>
      <td>9078</td>
      <td>8965</td>
      <td>9402</td>
      <td>9501</td>
      <td>9453</td>
      <td>9457</td>
      <td>9411</td>
      <td>...</td>
      <td>9208</td>
      <td>9208</td>
      <td>9204</td>
      <td>9235</td>
      <td>9279</td>
      <td>9327</td>
      <td>9345</td>
      <td>9515</td>
      <td>9559</td>
      <td>9581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>8080</td>
      <td>8080</td>
      <td>8077</td>
      <td>8101</td>
      <td>8267</td>
      <td>8274</td>
      <td>8360</td>
      <td>8360</td>
      <td>8370</td>
      <td>...</td>
      <td>8439</td>
      <td>8253</td>
      <td>8327</td>
      <td>8416</td>
      <td>8441</td>
      <td>8446</td>
      <td>8568</td>
      <td>8542</td>
      <td>8542</td>
      <td>8795</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>10204</td>
      <td>10204</td>
      <td>10408</td>
      <td>10408</td>
      <td>10000</td>
      <td>9844</td>
      <td>10058</td>
      <td>9974</td>
      <td>9973</td>
      <td>...</td>
      <td>10020</td>
      <td>10020</td>
      <td>10017</td>
      <td>9876</td>
      <td>9876</td>
      <td>9938</td>
      <td>10551</td>
      <td>10443</td>
      <td>10443</td>
      <td>10449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>6098</td>
      <td>7326</td>
      <td>7611</td>
      <td>7346</td>
      <td>7346</td>
      <td>7523</td>
      <td>7659</td>
      <td>7612</td>
      <td>7622</td>
      <td>...</td>
      <td>7752</td>
      <td>7748</td>
      <td>7752</td>
      <td>7756</td>
      <td>7861</td>
      <td>7914</td>
      <td>7877</td>
      <td>7881</td>
      <td>8089</td>
      <td>8231</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df_last.head()
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>모든면적</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60제곱미터이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60제곱미터초과 85제곱미터이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85제곱미터초과 102제곱미터이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102제곱미터초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
  </tbody>
</table>
</div>



`df_first`와 `df_last`의 데이터 형태가 다릅니다.  
**Tidy Data** 형식으로 변경해줘야합니다.

#### Tidy Data
Tidy Data란, 관측치가 행이고 변수가 열인 데이터로 데이터 분석에 용이한 데이터를 의미합니다.  
`df_first`의 경우 `wide form` 형태이고 `df_last`의 경우 `long form` 형태입니다.  
`lonog form`의 경우, 관측치가 행이고 변수가 열이기 때문에, tidy data라 할 수 있으며, `df_first`를 `long form` 형태로 변경해줘야합니다.

### 3. 데이터 요약하기


```python
df_first.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17 entries, 0 to 16
    Data columns (total 22 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   지역        17 non-null     object
     1   2013년12월  17 non-null     int64 
     2   2014년1월   17 non-null     int64 
     3   2014년2월   17 non-null     int64 
     4   2014년3월   17 non-null     int64 
     5   2014년4월   17 non-null     int64 
     6   2014년5월   17 non-null     int64 
     7   2014년6월   17 non-null     int64 
     8   2014년7월   17 non-null     int64 
     9   2014년8월   17 non-null     int64 
     10  2014년9월   17 non-null     int64 
     11  2014년10월  17 non-null     int64 
     12  2014년11월  17 non-null     int64 
     13  2014년12월  17 non-null     int64 
     14  2015년1월   17 non-null     int64 
     15  2015년2월   17 non-null     int64 
     16  2015년3월   17 non-null     int64 
     17  2015년4월   17 non-null     int64 
     18  2015년5월   17 non-null     int64 
     19  2015년6월   17 non-null     int64 
     20  2015년7월   17 non-null     int64 
     21  2015년8월   17 non-null     int64 
    dtypes: int64(21), object(1)
    memory usage: 3.0+ KB
    


```python
df_last.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6120 entries, 0 to 6119
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   지역명     6120 non-null   object
     1   규모구분    6120 non-null   object
     2   연도      6120 non-null   int64 
     3   월       6120 non-null   int64 
     4   분양가격    5641 non-null   object
    dtypes: int64(2), object(3)
    memory usage: 239.2+ KB
    

### 4. 결측치 확인하기


```python
df_first.isnull().sum()
```




    지역          0
    2013년12월    0
    2014년1월     0
    2014년2월     0
    2014년3월     0
    2014년4월     0
    2014년5월     0
    2014년6월     0
    2014년7월     0
    2014년8월     0
    2014년9월     0
    2014년10월    0
    2014년11월    0
    2014년12월    0
    2015년1월     0
    2015년2월     0
    2015년3월     0
    2015년4월     0
    2015년5월     0
    2015년6월     0
    2015년7월     0
    2015년8월     0
    dtype: int64




```python
df_last.isnull().sum()
```




    지역명       0
    규모구분      0
    연도        0
    월         0
    분양가격    479
    dtype: int64




```python
# 결측치 비율
df_last.isnull().mean()*100
```




    지역명     0.000000
    규모구분    0.000000
    연도      0.000000
    월       0.000000
    분양가격    7.826797
    dtype: float64




```python
# 결측치를 시각화
_ = sns.heatmap(df_last.isnull())
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_16_0.png?raw=true)
    


### 5. 데이터 타입 변경


```python
df_last["분양가격"] = pd.to_numeric(df_last["분양가격"], errors="coerce")
df_last.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6120 entries, 0 to 6119
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   지역명     6120 non-null   object 
     1   규모구분    6120 non-null   object 
     2   연도      6120 non-null   int64  
     3   월       6120 non-null   int64  
     4   분양가격    5625 non-null   float64
    dtypes: float64(1), int64(2), object(2)
    memory usage: 239.2+ KB
    

#### pd.melt로 형태 맞추기
`df_first`의 경우 `wide form` 형태이므로 `long form` 형태로 변경해줍니다.


```python
df_first_melt = pd.melt(df_first, id_vars=["지역"])
```


```python
df_first.head(2)
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
      <th>지역</th>
      <th>2013년12월</th>
      <th>2014년1월</th>
      <th>2014년2월</th>
      <th>2014년3월</th>
      <th>2014년4월</th>
      <th>2014년5월</th>
      <th>2014년6월</th>
      <th>2014년7월</th>
      <th>2014년8월</th>
      <th>...</th>
      <th>2014년11월</th>
      <th>2014년12월</th>
      <th>2015년1월</th>
      <th>2015년2월</th>
      <th>2015년3월</th>
      <th>2015년4월</th>
      <th>2015년5월</th>
      <th>2015년6월</th>
      <th>2015년7월</th>
      <th>2015년8월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>18189</td>
      <td>17925</td>
      <td>17925</td>
      <td>18016</td>
      <td>18098</td>
      <td>19446</td>
      <td>18867</td>
      <td>18742</td>
      <td>19274</td>
      <td>...</td>
      <td>20242</td>
      <td>20269</td>
      <td>20670</td>
      <td>20670</td>
      <td>19415</td>
      <td>18842</td>
      <td>18367</td>
      <td>18374</td>
      <td>18152</td>
      <td>18443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>8111</td>
      <td>8111</td>
      <td>9078</td>
      <td>8965</td>
      <td>9402</td>
      <td>9501</td>
      <td>9453</td>
      <td>9457</td>
      <td>9411</td>
      <td>...</td>
      <td>9208</td>
      <td>9208</td>
      <td>9204</td>
      <td>9235</td>
      <td>9279</td>
      <td>9327</td>
      <td>9345</td>
      <td>9515</td>
      <td>9559</td>
      <td>9581</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 22 columns</p>
</div>




```python
df_first_melt.columns = ["지역명","기간", "평당분양가격"]
df_first_melt.head(2)
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
      <th>지역명</th>
      <th>기간</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2013년12월</td>
      <td>18189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>2013년12월</td>
      <td>8111</td>
    </tr>
  </tbody>
</table>
</div>



`df_frist_melt`와 `df_last`의 컬럼을 맞춰줍니다.
##### 연도와 월 분리


```python
def parse_year(date):
    return int(date.split("년")[0])
def parse_month(date):
    return int(date.split("년")[-1].replace("월", ""))
```


```python
df_first_melt["연도"] = df_first_melt["기간"].apply(parse_year)
df_first_melt["월"] = df_first_melt["기간"].apply(parse_month)
```


```python
df_first_melt.sample(5)
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
      <th>지역명</th>
      <th>기간</th>
      <th>평당분양가격</th>
      <th>연도</th>
      <th>월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>대구</td>
      <td>2014년10월</td>
      <td>8403</td>
      <td>2014</td>
      <td>10</td>
    </tr>
    <tr>
      <th>59</th>
      <td>세종</td>
      <td>2014년3월</td>
      <td>7814</td>
      <td>2014</td>
      <td>3</td>
    </tr>
    <tr>
      <th>282</th>
      <td>충북</td>
      <td>2015년4월</td>
      <td>6790</td>
      <td>2015</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>경기</td>
      <td>2013년12월</td>
      <td>10855</td>
      <td>2013</td>
      <td>12</td>
    </tr>
    <tr>
      <th>328</th>
      <td>대전</td>
      <td>2015년7월</td>
      <td>8079</td>
      <td>2015</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



### 6. 분석하기

#### 평당 분양가격 구하기
분양가격을 평당 기준으로 보기 위해 새로운 컬럼을 만듭니다.


```python
df_last["평당분양가격"] = df_last["분양가격"]*3.3
```


```python
df_last.sample(5)
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3634</th>
      <td>전북</td>
      <td>전용면적 102제곱미터초과</td>
      <td>2019</td>
      <td>4</td>
      <td>2634.0</td>
      <td>8692.2</td>
    </tr>
    <tr>
      <th>3214</th>
      <td>전남</td>
      <td>전용면적 102제곱미터초과</td>
      <td>2018</td>
      <td>11</td>
      <td>2502.0</td>
      <td>8256.6</td>
    </tr>
    <tr>
      <th>2779</th>
      <td>충남</td>
      <td>전용면적 102제곱미터초과</td>
      <td>2018</td>
      <td>6</td>
      <td>2580.0</td>
      <td>8514.0</td>
    </tr>
    <tr>
      <th>5836</th>
      <td>충남</td>
      <td>전용면적 60제곱미터이하</td>
      <td>2021</td>
      <td>6</td>
      <td>2806.0</td>
      <td>9259.8</td>
    </tr>
    <tr>
      <th>5471</th>
      <td>대전</td>
      <td>전용면적 60제곱미터이하</td>
      <td>2021</td>
      <td>2</td>
      <td>3061.0</td>
      <td>10101.3</td>
    </tr>
  </tbody>
</table>
</div>



#### 분양가격 요약


```python
df_last.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6120 entries, 0 to 6119
    Data columns (total 6 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   지역명     6120 non-null   object 
     1   규모구분    6120 non-null   object 
     2   연도      6120 non-null   int64  
     3   월       6120 non-null   int64  
     4   분양가격    5625 non-null   float64
     5   평당분양가격  5625 non-null   float64
    dtypes: float64(2), int64(2), object(2)
    memory usage: 287.0+ KB
    


```python
df_last["분양가격"].describe()
```




    count     5625.000000
    mean      3459.317867
    std       1411.776054
    min       1868.000000
    25%       2574.000000
    50%       3067.000000
    75%       3922.000000
    max      13835.000000
    Name: 분양가격, dtype: float64



#### 컬럼 정리하기
컬럼의 내용을 좀 더 직관적이고 간결하게 변환해줍니다.


```python
df_last["규모구분"].unique()
```




    array(['모든면적', '전용면적 60제곱미터이하', '전용면적 60제곱미터초과 85제곱미터이하',
           '전용면적 85제곱미터초과 102제곱미터이하', '전용면적 102제곱미터초과'], dtype=object)




```python
df_last["전용면적"] = df_last["규모구분"].str.replace("전용면적|제곱미터|이하| ", "", regex=True)
df_last["전용면적"] = df_last["전용면적"].str.replace("초과", "~")
```


```python
df_last.sample(5)
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격</th>
      <th>평당분양가격</th>
      <th>전용면적</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3411</th>
      <td>경기</td>
      <td>전용면적 60제곱미터이하</td>
      <td>2019</td>
      <td>2</td>
      <td>4444.0</td>
      <td>14665.2</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4070</th>
      <td>경남</td>
      <td>모든면적</td>
      <td>2019</td>
      <td>9</td>
      <td>2995.0</td>
      <td>9883.5</td>
      <td>모든면적</td>
    </tr>
    <tr>
      <th>175</th>
      <td>인천</td>
      <td>모든면적</td>
      <td>2015</td>
      <td>12</td>
      <td>3184.0</td>
      <td>10507.2</td>
      <td>모든면적</td>
    </tr>
    <tr>
      <th>959</th>
      <td>대구</td>
      <td>전용면적 102제곱미터초과</td>
      <td>2016</td>
      <td>9</td>
      <td>3045.0</td>
      <td>10048.5</td>
      <td>102~</td>
    </tr>
    <tr>
      <th>5487</th>
      <td>강원</td>
      <td>전용면적 60제곱미터초과 85제곱미터이하</td>
      <td>2021</td>
      <td>2</td>
      <td>3083.0</td>
      <td>10173.9</td>
      <td>60~85</td>
    </tr>
  </tbody>
</table>
</div>



#### 필요없는 컬럼 제거하기


```python
df_last = df_last.drop(["규모구분", "분양가격"], axis=1)
```


```python
df_last.sample(5)
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
      <th>지역명</th>
      <th>연도</th>
      <th>월</th>
      <th>평당분양가격</th>
      <th>전용면적</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>611</th>
      <td>부산</td>
      <td>2016</td>
      <td>5</td>
      <td>9824.1</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1021</th>
      <td>서울</td>
      <td>2016</td>
      <td>10</td>
      <td>21981.3</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4561</th>
      <td>충남</td>
      <td>2020</td>
      <td>3</td>
      <td>7887.0</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1884</th>
      <td>경기</td>
      <td>2017</td>
      <td>8</td>
      <td>13322.1</td>
      <td>102~</td>
    </tr>
    <tr>
      <th>4449</th>
      <td>광주</td>
      <td>2020</td>
      <td>2</td>
      <td>14223.0</td>
      <td>102~</td>
    </tr>
  </tbody>
</table>
</div>



#### 컬럼 맞추기


```python
df_first_melt.columns.to_list()
```




    ['지역명', '기간', '평당분양가격', '연도', '월']




```python
df_last.columns.to_list()
```




    ['지역명', '연도', '월', '평당분양가격', '전용면적']




```python
cols = ["지역명", "연도", "월", "평당분양가격"]
```


```python
df_first_prepare = df_first_melt.loc[:, cols].copy()
df_last_prepare = df_last.loc[df_last["전용면적"]=="모든면적", cols].copy()
```

#### 데이터 합치기


```python
df = pd.concat([df_first_prepare, df_last_prepare], axis=0)
```


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
      <th>지역명</th>
      <th>연도</th>
      <th>월</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2013</td>
      <td>12</td>
      <td>18189.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>2013</td>
      <td>12</td>
      <td>8111.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>2013</td>
      <td>12</td>
      <td>8080.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>2013</td>
      <td>12</td>
      <td>10204.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>2013</td>
      <td>12</td>
      <td>6098.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6095</th>
      <td>전북</td>
      <td>2021</td>
      <td>9</td>
      <td>8715.3</td>
    </tr>
    <tr>
      <th>6100</th>
      <td>전남</td>
      <td>2021</td>
      <td>9</td>
      <td>10487.4</td>
    </tr>
    <tr>
      <th>6105</th>
      <td>경북</td>
      <td>2021</td>
      <td>9</td>
      <td>10345.5</td>
    </tr>
    <tr>
      <th>6110</th>
      <td>경남</td>
      <td>2021</td>
      <td>9</td>
      <td>10873.5</td>
    </tr>
    <tr>
      <th>6115</th>
      <td>제주</td>
      <td>2021</td>
      <td>9</td>
      <td>27574.8</td>
    </tr>
  </tbody>
</table>
<p>1581 rows × 4 columns</p>
</div>



### 7. 시각화

#### 수치데이터 히스토그램


```python
_ = df_last.hist(figsize=(10, 6), bins=100)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_51_0.png?raw=true)
    


`bins` 옵션을 높게 설정하면, 해당 데이터의 수치형 데이터와 범주형 데이터를 찾는데 도움이 됩니다.

#### Pairplot


```python
_ = sns.pairplot(data=df_last, hue="지역명")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_54_0.png?raw=true)
    


특정 데이터만 보고 싶은 경우, `isin` 메서드를 사용할 수 있습니다.


```python
_ = sns.pairplot(data=df_last[df_last["지역명"].isin(["서울", "경기", "인천"])], hue="지역명")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_56_0.png?raw=true)
    


#### 연도별 평당분양가격


```python
plt.figure(figsize=(12, 4))
_ = sns.barplot(data=df, x="연도", y="평당분양가격", ci=None)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_58_0.png?raw=true)
    



```python
plt.figure(figsize=(12, 4))
_ = sns.pointplot(data=df, x="연도", y="평당분양가격", ci=None)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_59_0.png?raw=true)
    



```python
plt.figure(figsize=(12, 4))
_ = sns.boxplot(data=df, x="연도", y="평당분양가격")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_60_0.png?raw=true)
    



```python
plt.figure(figsize=(12, 4))
_ = sns.violinplot(data=df, x="연도", y="평당분양가격")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_61_0.png?raw=true)
    



```python
plt.figure(figsize=(12, 4))
_ = sns.swarmplot(data=df, x="연도", y="평당분양가격", size=2)

```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_62_0.png?raw=true)
    


#### 지역별 평당분양 가격


```python
_ = sns.barplot(data=df, x="지역명", y="평당분양가격", ci=None)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_64_0.png?raw=true)
    



```python
_ = sns.boxplot(data=df, x="지역명", y="평당분양가격")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_65_0.png?raw=true)
    



```python
plt.figure(figsize=(12, 4))
_ = sns.violinplot(data=df, x="지역명", y="평당분양가격")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_66_0.png?raw=true)
    



```python
plt.figure(figsize=(15, 8))
_ = sns.swarmplot(data=df, x="지역명", y="평당분양가격", size=2)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_67_0.png?raw=true)
    


### 8. 데이터 집계
#### 지역별 분양가격 평균


```python
_ = df.groupby("지역명")["평당분양가격"].mean().plot(kind="bar", rot=30)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_69_0.png?raw=true)
    


#### 연도별 지역별 평당 분양가격 평균


```python
_ = df.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack().plot.bar(rot=30)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_71_0.png?raw=true)
    



```python
_ = df.groupby(["연도", "지역명"])["평당분양가격"].mean().unstack()[["서울", "경기"]].plot(kind="bar", rot=30)
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/apt-price-analysis_files/apt-price-analysis_72_0.png?raw=true)
    

