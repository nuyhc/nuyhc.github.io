---
title: "KOSIS 암 검진 데이터 시각화"
date: 2022-06-06T12:10:05.401Z
img_path: /nuyhc.github.io/assets/img
categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Seaborn
---


# KOSIS 암 검진 데이터 분석
## 1. 소계
KOSIS에서 제공해주는 연령, 성별 암검진 대상 및 수검현환에 대한 데이터의 분석과 시각화를 진행해본 프로젝트입니다.

## 2. 목표
EDA와 시각화

## 3. 사용 데이터 셋
[KOSIS 건강검진 통계 -> 암검진 -> 연령별 성별 암검진 대상 및 수검인원 현황](https://kosis.kr/statHtml/statHtml.do?orgId=350&tblId=DT_35007_N010&conn_path=I2)

## 4. 구현

### 라이브러리


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### 한글폰트 설정


```python
def get_font_family():
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """
    import os
    import platform
    system_name = platform.system()

    if system_name == "Darwin" :
        font_family = "AppleGothic"
    elif system_name == "Windows":
        font_family = "Malgun Gothic"
    else:
        # Linux(colab)
        os.system("apt-get install fonts-nanum -qq  > /dev/null")
        os.system("fc-cache -fv")

        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManager.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont

        font_family = "NanumBarunGothic"
    return font_family
```


```python
# 시각화를 위한 폰트설정
# 위에서 만든 함수를 통해 시스템 폰트를 불러와서 font_family라는 변수에 할당합니다.
plt.style.use("ggplot")

font_family = get_font_family()
# 폰트설정
plt.rc("font", family=font_family)
# 마이너스 폰트 설정
plt.rc("axes", unicode_minus=False)
# 그래프에 retina display 적용
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
```

### Data Load


```python
df = pd.read_csv("data/kosis-cancer-raw.csv", encoding="cp949")
df.sample(5)
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
      <th>연령별(1)</th>
      <th>시점</th>
      <th>암검진별(1)</th>
      <th>성별(1)</th>
      <th>대상인원 (명)</th>
      <th>수검인원 (명)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1331</th>
      <td>55 ~ 59세</td>
      <td>2018</td>
      <td>대장암</td>
      <td>남자</td>
      <td>1677732</td>
      <td>578815</td>
    </tr>
    <tr>
      <th>971</th>
      <td>45 ~ 49세</td>
      <td>2018</td>
      <td>위암</td>
      <td>합계</td>
      <td>1570334</td>
      <td>942193</td>
    </tr>
    <tr>
      <th>931</th>
      <td>45 ~ 49세</td>
      <td>2015</td>
      <td>자궁경부암</td>
      <td>여자</td>
      <td>787632</td>
      <td>461445</td>
    </tr>
    <tr>
      <th>701</th>
      <td>40 ~ 44세</td>
      <td>2012</td>
      <td>간암</td>
      <td>여자</td>
      <td>65330</td>
      <td>31125</td>
    </tr>
    <tr>
      <th>340</th>
      <td>30 ~ 34세</td>
      <td>2011</td>
      <td>계</td>
      <td>합계</td>
      <td>615279</td>
      <td>254831</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (2428, 6)



### 컬럼 이름 바꿔주기


```python
df = df.rename(columns={"연령별(1)":"연령별", "암검진별(1)":"암검진별", "성별(1)":"성별"}).copy()
df.sample(5)
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
      <th>연령별</th>
      <th>시점</th>
      <th>암검진별</th>
      <th>성별</th>
      <th>대상인원 (명)</th>
      <th>수검인원 (명)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>665</th>
      <td>40 ~ 44세</td>
      <td>2010</td>
      <td>간암</td>
      <td>여자</td>
      <td>29728</td>
      <td>15744</td>
    </tr>
    <tr>
      <th>1365</th>
      <td>60 ~ 64세</td>
      <td>2010</td>
      <td>위암</td>
      <td>여자</td>
      <td>564894</td>
      <td>343419</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>60 ~ 64세</td>
      <td>2015</td>
      <td>대장암</td>
      <td>합계</td>
      <td>2308221</td>
      <td>968199</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>80 ~ 84세</td>
      <td>2013</td>
      <td>간암</td>
      <td>남자</td>
      <td>5669</td>
      <td>1867</td>
    </tr>
    <tr>
      <th>151</th>
      <td>계</td>
      <td>2018</td>
      <td>간암</td>
      <td>합계</td>
      <td>746363</td>
      <td>533580</td>
    </tr>
  </tbody>
</table>
</div>



### 파생변수 생성


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2428 entries, 0 to 2427
    Data columns (total 6 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   연령별       2428 non-null   object
     1   시점        2428 non-null   int64 
     2   암검진별      2428 non-null   object
     3   성별        2428 non-null   object
     4   대상인원 (명)  2428 non-null   object
     5   수검인원 (명)  2428 non-null   object
    dtypes: int64(1), object(5)
    memory usage: 113.9+ KB
    


```python
df["대상인원 (명)"].value_counts()[:3]
```




    -         560
    0          56
    622102      4
    Name: 대상인원 (명), dtype: int64




```python
df["대상인원"] = df["대상인원 (명)"].replace("-", 0).astype(int)
df["수검인원"] = df["수검인원 (명)"].replace("-", 0).astype(int)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2428 entries, 0 to 2427
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   연령별       2428 non-null   object
     1   시점        2428 non-null   int64 
     2   암검진별      2428 non-null   object
     3   성별        2428 non-null   object
     4   대상인원 (명)  2428 non-null   object
     5   수검인원 (명)  2428 non-null   object
     6   대상인원      2428 non-null   int32 
     7   수검인원      2428 non-null   int32 
    dtypes: int32(2), int64(1), object(5)
    memory usage: 132.9+ KB
    

### 사용하지 않는 데이터 제거


```python
df = df.drop(df[(df["연령별"]=="계") | (df["암검진별"]=="계") | (df["성별"]=="합계")].index).reset_index(drop=True).copy()
df.sample(5)
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
      <th>연령별</th>
      <th>시점</th>
      <th>암검진별</th>
      <th>성별</th>
      <th>대상인원 (명)</th>
      <th>수검인원 (명)</th>
      <th>대상인원</th>
      <th>수검인원</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>30 ~ 34세</td>
      <td>2016</td>
      <td>유방암</td>
      <td>여자</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>572</th>
      <td>55 ~ 59세</td>
      <td>2012</td>
      <td>위암</td>
      <td>남자</td>
      <td>627522</td>
      <td>329496</td>
      <td>627522</td>
      <td>329496</td>
    </tr>
    <tr>
      <th>965</th>
      <td>75 ~ 79세</td>
      <td>2012</td>
      <td>위암</td>
      <td>여자</td>
      <td>236755</td>
      <td>112939</td>
      <td>236755</td>
      <td>112939</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>80 ~ 84세</td>
      <td>2013</td>
      <td>위암</td>
      <td>남자</td>
      <td>110984</td>
      <td>45147</td>
      <td>110984</td>
      <td>45147</td>
    </tr>
    <tr>
      <th>174</th>
      <td>35 ~ 39세</td>
      <td>2010</td>
      <td>대장암</td>
      <td>여자</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 고유값 확인


```python
df.nunique()
```




    연령별          14
    시점           10
    암검진별          5
    성별            2
    대상인원 (명)    789
    수검인원 (명)    789
    대상인원        788
    수검인원        788
    dtype: int64



### 파생변수 만들기


```python
df["연령별"].unique()
```




    array(['20 ~ 24세', '25 ~ 29세', '30 ~ 34세', '35 ~ 39세', '40 ~ 44세',
           '45 ~ 49세', '50 ~ 54세', '55 ~ 59세', '60 ~ 64세', '65 ~ 69세',
           '70 ~ 74세', '75 ~ 79세', '80 ~ 84세', '85세 이상'], dtype=object)




```python
df["연령대"] = df["연령별"].str[0]+"0대"
df.sample(5)
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
      <th>연령별</th>
      <th>시점</th>
      <th>암검진별</th>
      <th>성별</th>
      <th>대상인원 (명)</th>
      <th>수검인원 (명)</th>
      <th>대상인원</th>
      <th>수검인원</th>
      <th>연령대</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>333</th>
      <td>40 ~ 44세</td>
      <td>2017</td>
      <td>간암</td>
      <td>여자</td>
      <td>41531</td>
      <td>30471</td>
      <td>41531</td>
      <td>30471</td>
      <td>40대</td>
    </tr>
    <tr>
      <th>528</th>
      <td>50 ~ 54세</td>
      <td>2017</td>
      <td>유방암</td>
      <td>남자</td>
      <td>-</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>50대</td>
    </tr>
    <tr>
      <th>613</th>
      <td>55 ~ 59세</td>
      <td>2016</td>
      <td>대장암</td>
      <td>여자</td>
      <td>1676782</td>
      <td>603242</td>
      <td>1676782</td>
      <td>603242</td>
      <td>50대</td>
    </tr>
    <tr>
      <th>992</th>
      <td>75 ~ 79세</td>
      <td>2014</td>
      <td>자궁경부암</td>
      <td>남자</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70대</td>
    </tr>
    <tr>
      <th>807</th>
      <td>65 ~ 69세</td>
      <td>2016</td>
      <td>위암</td>
      <td>여자</td>
      <td>429477</td>
      <td>318689</td>
      <td>429477</td>
      <td>318689</td>
      <td>60대</td>
    </tr>
  </tbody>
</table>
</div>



### 사용하지 않는 컬럼 제거


```python
df = df.drop(columns=["대상인원 (명)", "수검인원 (명)"])
df.sample(5)
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
      <th>연령별</th>
      <th>시점</th>
      <th>암검진별</th>
      <th>성별</th>
      <th>대상인원</th>
      <th>수검인원</th>
      <th>연령대</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>30 ~ 34세</td>
      <td>2012</td>
      <td>위암</td>
      <td>남자</td>
      <td>0</td>
      <td>0</td>
      <td>30대</td>
    </tr>
    <tr>
      <th>6</th>
      <td>20 ~ 24세</td>
      <td>2016</td>
      <td>유방암</td>
      <td>남자</td>
      <td>0</td>
      <td>0</td>
      <td>20대</td>
    </tr>
    <tr>
      <th>585</th>
      <td>55 ~ 59세</td>
      <td>2013</td>
      <td>대장암</td>
      <td>여자</td>
      <td>1412672</td>
      <td>459468</td>
      <td>50대</td>
    </tr>
    <tr>
      <th>176</th>
      <td>35 ~ 39세</td>
      <td>2010</td>
      <td>간암</td>
      <td>여자</td>
      <td>0</td>
      <td>0</td>
      <td>30대</td>
    </tr>
    <tr>
      <th>901</th>
      <td>70 ~ 74세</td>
      <td>2015</td>
      <td>간암</td>
      <td>여자</td>
      <td>28056</td>
      <td>15686</td>
      <td>70대</td>
    </tr>
  </tbody>
</table>
</div>



### 시각화


```python
_ = sns.countplot(data=df, x="암검진별").set_title("암검진별 빈도수")
```


    
![png](kosis-cancer_files/kosis-cancer_26_0.png)
    



```python
_ = sns.countplot(data=df, x="암검진별", hue="성별").set_title("암검진별 빈도수 성별 분류")
```


    
![png](kosis-cancer_files/kosis-cancer_27_0.png)
    



```python
_ = df.loc[df["대상인원"]>0, "암검진별"].value_counts().plot.barh().set_title("0을 제외한 빈도수")
```


    
![png](kosis-cancer_files/kosis-cancer_28_0.png)
    



```python
# 수치형 데이터
_ = df.hist(bins=100, figsize=(12, 6))
```


    
![png](kosis-cancer_files/kosis-cancer_29_0.png)
    



```python
plt.figure(figsize=(15, 8))
_ = sns.barplot(data=df, x="암검진별", y="수검인원", hue="성별", ci=None, estimator=np.sum).set_title("암검진별 수검인원 합계")
```


    
![png](kosis-cancer_files/kosis-cancer_30_0.png)
    



```python
_ = sns.catplot(data=df, x="연령대", y="수검인원", hue="암검진별", col="성별", kind="bar", estimator=np.sum, ci=None).fig.suptitle("연령대별 암검진 수검인원 합계")
```


    
![png](kosis-cancer_files/kosis-cancer_31_0.png)
    



```python
_ = sns.pointplot(data=df, x="시점", y="수검인원", ci=None).set_title("연도별 암검진 평균")
```


    
![png](kosis-cancer_files/kosis-cancer_32_0.png)
    



```python
_ = sns.pointplot(data=df, x="시점", y="수검인원", ci=None, estimator=np.sum).set_title("연도별 암검진 합계")
```


    
![png](kosis-cancer_files/kosis-cancer_33_0.png)
    



```python
_ = sns.catplot(data=df, x="시점", y="수검인원").fig.suptitle("연도별 암검진 수검인원")
```


    
![png](kosis-cancer_files/kosis-cancer_34_0.png)
    



```python
_ = sns.catplot(data=df, x="시점", y="수검인원", estimator=np.sum, kind="bar", ci=None, col="성별", hue="연령대").fig.suptitle("연도별 암검진 수검인원 합계")
```


    
![png](kosis-cancer_files/kosis-cancer_35_0.png)
    

