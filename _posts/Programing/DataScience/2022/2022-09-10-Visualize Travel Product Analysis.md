---
title: DACON 여행 상품 분석 시각화 경진대회
date: 2022-09-10T12:12:08.891Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Numpy
  - matplot
  - Seaborn
---

# DACON Basic 여행 상품 분석 시각화 경진대회
단순히 EDA와 시각화를 진행하는건데 가설 설정이 너무 어려운거 같다..  
많은 시간을 두고 진행했지만 딱히 떠오르는게 없어서 스터디에서 다른 사람들이 해오는걸 보고 내용을 추가해봐야할꺼 같다

### 라이브러리


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
```

### Data Load


```python
df = pd.read_csv("./train.csv")

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
      <th>id</th>
      <th>Age</th>
      <th>TypeofContact</th>
      <th>CityTier</th>
      <th>DurationOfPitch</th>
      <th>Occupation</th>
      <th>Gender</th>
      <th>NumberOfPersonVisiting</th>
      <th>NumberOfFollowups</th>
      <th>ProductPitched</th>
      <th>PreferredPropertyStar</th>
      <th>MaritalStatus</th>
      <th>NumberOfTrips</th>
      <th>Passport</th>
      <th>PitchSatisfactionScore</th>
      <th>OwnCar</th>
      <th>NumberOfChildrenVisiting</th>
      <th>Designation</th>
      <th>MonthlyIncome</th>
      <th>ProdTaken</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>28.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>10.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>4.0</td>
      <td>Basic</td>
      <td>3.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>20384.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>34.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>NaN</td>
      <td>Small Business</td>
      <td>Female</td>
      <td>2</td>
      <td>4.0</td>
      <td>Deluxe</td>
      <td>4.0</td>
      <td>Single</td>
      <td>1.0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>19599.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>45.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>NaN</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Deluxe</td>
      <td>4.0</td>
      <td>Married</td>
      <td>2.0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>29.0</td>
      <td>Company Invited</td>
      <td>1</td>
      <td>7.0</td>
      <td>Small Business</td>
      <td>Male</td>
      <td>3</td>
      <td>5.0</td>
      <td>Basic</td>
      <td>4.0</td>
      <td>Married</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1.0</td>
      <td>Executive</td>
      <td>21274.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>42.0</td>
      <td>Self Enquiry</td>
      <td>3</td>
      <td>6.0</td>
      <td>Salaried</td>
      <td>Male</td>
      <td>2</td>
      <td>3.0</td>
      <td>Deluxe</td>
      <td>3.0</td>
      <td>Divorced</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0.0</td>
      <td>Manager</td>
      <td>19907.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



- id : 샘플 아이디
- Age : 나이
- TypeofContact : 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색)
- CityTier : 주거 중인 도시의 등급. (인구, 시설, 생활 수준 기준) (1등급 > 2등급 > 3등급)
- DurationOfPitch : 영업 사원이 고객에게 제공하는 프레젠테이션 기간
- Occupation : 직업
- Gender : 성별
- NumberOfPersonVisiting : 고객과 함께 여행을 계획 중인 총 인원
- NumberOfFollowups : 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
- ProductPitched : 영업 사원이 제시한 상품
- PreferredPropertyStar : 선호 호텔 숙박업소 등급
- MaritalStatus : 결혼여부
- NumberOfTrips : 평균 연간 여행 횟수
- Passport : 여권 보유 여부 (0: 없음, 1: 있음)
- PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도
- OwnCar : 자동차 보유 여부 (0: 없음, 1: 있음)
- NumberOfChildrenVisiting : 함께 여행을 계획 중인 5세 미만의 어린이 수
- Designation : (직업의) 직급
- MonthlyIncome : 월 급여
- ProdTaken : 여행 패키지 신청 여부 (0: 신청 안 함, 1: 신청함)

`id`는 버림


```python
df.drop(columns=["id"], inplace=True)
```

### 결측치 확인


```python
plt.figure(figsize=(12, 7))
sns.heatmap(df.isna())
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_7_0.png)
    


시각화가 목적이므로 결측치는 모두 버림


```python
df.dropna(inplace=True)
```

### Age: 나이


```python
plt.figure(figsize=(15, 5))
sns.countplot(data=df, x="Age")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_11_0.png)
    


#### Band Age
예측이나 분류에서는 주어진 데이터의 분포에 맞춰 정제하는 것이 좋은 성능의 모델을 만드는데 도움이 된다고 하지만, 시각화가 목표이므로 분포는 무시하고 일반적인 분포로 나눔


```python
df["Band Age"] = df["Age"].map(lambda x: str(x)[0]+"0대")
```


```python
plt.figure(figsize=(10, 5))
plt.subplot(221)
sns.countplot(data=df, y="Band Age", order=df["Band Age"].value_counts().index)
plt.subplot(222)
plt.pie(df["Band Age"].value_counts().values, labels=df["Band Age"].value_counts().index, autopct="%.2f%%")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_14_0.png)
    


### TypeofContact: 인지 방법


```python
plt.figure(figsize=(12, 7))
plt.subplot(221)
sns.countplot(data=df, x="TypeofContact")
plt.subplot(222)
plt.pie(df["TypeofContact"].value_counts().values, labels=df["TypeofContact"].value_counts().index, autopct="%.2f%%")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_16_0.png)
    


### CityTier: 주거 중인 도시 등급


```python
sns.countplot(data=df, x="CityTier");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_18_0.png)
    


### DurationOfPitch: 고객에게 제공하는 프레젠테이션 기간


```python
plt.figure(figsize=(15, 4))
sns.countplot(data=df, x="DurationOfPitch");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_20_0.png)
    


### Gender: 성별
`Fe Male`과 `Female`은 같은 것 아닌가..?


```python
df["Gender"] = df["Gender"].str.replace("Fe Male", "Female")
```


```python
sns.countplot(data=df, x="Gender");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_23_0.png)
    


### NumberOfPersonVisiting: 고객과 함께 여행을 계획 중인 총 인원


```python
sns.countplot(data=df, x="NumberOfPersonVisiting");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_25_0.png)
    


### NumberOfFollowups: 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수
정확히 의미하는게 뭔지 모르겠음


```python
sns.countplot(data=df, x="NumberOfFollowups");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_27_0.png)
    


### ProductPitched: 영업 사원이 제시한 상품


```python
plt.figure(figsize=(12, 7), facecolor="white")
plt.subplot(221)
plt.pie(df["ProductPitched"].value_counts().values, labels=df["ProductPitched"].value_counts().index, autopct="%.2f%%")
plt.subplot(222)
sns.countplot(data=df, x="ProductPitched")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_29_0.png)
    


### PreferredPropertyStar: 선호 호텔 숙박업소 등급


```python
plt.figure(figsize=(12, 7), facecolor="white")
plt.subplot(221)
plt.pie(df["PreferredPropertyStar"].value_counts().values, labels=df["PreferredPropertyStar"].value_counts().index, autopct="%.2f%%")
plt.subplot(222)
sns.countplot(data=df, x="PreferredPropertyStar")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_31_0.png)
    


### MaritalStatus: 결혼여부


```python
plt.figure(figsize=(12, 7))
plt.subplot(221)
plt.pie(df["MaritalStatus"].value_counts().values, labels=df["MaritalStatus"].value_counts().index, autopct="%.2f%%")
plt.subplot(222)
sns.countplot(data=df, y="MaritalStatus")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_33_0.png)
    


### NumberOfTrips: 평균 연간 여행 횟수


```python
sns.countplot(data=df, x="NumberOfTrips");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_35_0.png)
    


### Passport: 여권 보유 여부 (0: 없음, 1: 있음)


```python
sns.countplot(data=df, x="Passport");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_37_0.png)
    


### PitchSatisfactionScore : 영업 사원의 프레젠테이션 만족도


```python
plt.figure(figsize=(12, 7))
plt.subplot(221)
plt.pie(df["PitchSatisfactionScore"].value_counts().values, labels=df["PitchSatisfactionScore"].value_counts().index, autopct="%.2f%%")
plt.subplot(222)
sns.countplot(data=df, y="PitchSatisfactionScore")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_39_0.png)
    


### OwnCar: 자동차 보유 여부 (0: 없음, 1: 있음)


```python
sns.countplot(data=df, x="OwnCar");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_41_0.png)
    


### NumberOfChildrenVisiting: 함께 여행을 계획 중인 5세 미만의 어린이 수


```python
sns.countplot(data=df, y="NumberOfChildrenVisiting");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_43_0.png)
    


### Designation: (직업의) 직급


```python
plt.figure(figsize=(12, 7))
plt.subplot(221)
plt.pie(df["Designation"].value_counts().values, labels=df["Designation"].value_counts().index, autopct="%.2f%%")
plt.subplot(222)
sns.countplot(data=df, x="Designation")
plt.show()
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_45_0.png)
    


### MonthlyIncome: 월 급여


```python
sns.scatterplot(data=df, x="Designation", y="MonthlyIncome");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_47_0.png)
    


### 상관 관계로 본 시각화


```python
df.drop(columns=["id"]).corr(method="pearson").style.background_gradient()
```




<style type="text/css">
#T_65a08_row0_col0, #T_65a08_row1_col1, #T_65a08_row2_col2, #T_65a08_row3_col3, #T_65a08_row4_col4, #T_65a08_row5_col5, #T_65a08_row6_col6, #T_65a08_row7_col7, #T_65a08_row8_col8, #T_65a08_row9_col9, #T_65a08_row10_col10, #T_65a08_row11_col11, #T_65a08_row12_col12 {
  background-color: #023858;
  color: #f1f1f1;
}
#T_65a08_row0_col1, #T_65a08_row0_col10 {
  background-color: #faf3f9;
  color: #000000;
}
#T_65a08_row0_col2, #T_65a08_row2_col5, #T_65a08_row2_col6, #T_65a08_row2_col10, #T_65a08_row7_col1, #T_65a08_row7_col4, #T_65a08_row8_col4, #T_65a08_row9_col1 {
  background-color: #f9f2f8;
  color: #000000;
}
#T_65a08_row0_col3, #T_65a08_row7_col6, #T_65a08_row7_col10, #T_65a08_row8_col2, #T_65a08_row9_col6, #T_65a08_row11_col5 {
  background-color: #fbf4f9;
  color: #000000;
}
#T_65a08_row0_col4, #T_65a08_row1_col7, #T_65a08_row1_col9, #T_65a08_row4_col2, #T_65a08_row5_col6, #T_65a08_row5_col7, #T_65a08_row7_col2, #T_65a08_row8_col6, #T_65a08_row10_col1 {
  background-color: #f7f0f7;
  color: #000000;
}
#T_65a08_row0_col5, #T_65a08_row4_col8, #T_65a08_row5_col10, #T_65a08_row11_col8 {
  background-color: #fcf4fa;
  color: #000000;
}
#T_65a08_row0_col6 {
  background-color: #dddbec;
  color: #000000;
}
#T_65a08_row0_col7, #T_65a08_row5_col9, #T_65a08_row10_col5, #T_65a08_row10_col7 {
  background-color: #f4eef6;
  color: #000000;
}
#T_65a08_row0_col8, #T_65a08_row3_col9, #T_65a08_row6_col8, #T_65a08_row7_col5, #T_65a08_row8_col7, #T_65a08_row10_col2, #T_65a08_row11_col7 {
  background-color: #f6eff7;
  color: #000000;
}
#T_65a08_row0_col9, #T_65a08_row5_col11, #T_65a08_row9_col8 {
  background-color: #f0eaf4;
  color: #000000;
}
#T_65a08_row0_col11 {
  background-color: #6fa7ce;
  color: #f1f1f1;
}
#T_65a08_row0_col12, #T_65a08_row6_col1, #T_65a08_row12_col9 {
  background-color: #fef6fb;
  color: #000000;
}
#T_65a08_row1_col0, #T_65a08_row4_col0, #T_65a08_row10_col12 {
  background-color: #ece7f2;
  color: #000000;
}
#T_65a08_row1_col2, #T_65a08_row1_col4 {
  background-color: #f5eef6;
  color: #000000;
}
#T_65a08_row1_col3, #T_65a08_row2_col9, #T_65a08_row5_col3, #T_65a08_row8_col5, #T_65a08_row9_col3, #T_65a08_row9_col10, #T_65a08_row11_col2 {
  background-color: #fbf3f9;
  color: #000000;
}
#T_65a08_row1_col5, #T_65a08_row2_col8, #T_65a08_row4_col7, #T_65a08_row6_col2, #T_65a08_row7_col3 {
  background-color: #faf2f8;
  color: #000000;
}
#T_65a08_row1_col6, #T_65a08_row1_col8, #T_65a08_row4_col5, #T_65a08_row5_col4, #T_65a08_row7_col9, #T_65a08_row8_col1, #T_65a08_row8_col3, #T_65a08_row9_col2, #T_65a08_row9_col7, #T_65a08_row11_col12, #T_65a08_row12_col0, #T_65a08_row12_col10, #T_65a08_row12_col11 {
  background-color: #fff7fb;
  color: #000000;
}
#T_65a08_row1_col10, #T_65a08_row3_col8, #T_65a08_row5_col1, #T_65a08_row8_col10, #T_65a08_row12_col3 {
  background-color: #fdf5fa;
  color: #000000;
}
#T_65a08_row1_col11, #T_65a08_row6_col4, #T_65a08_row9_col0, #T_65a08_row11_col10 {
  background-color: #e1dfed;
  color: #000000;
}
#T_65a08_row1_col12 {
  background-color: #dcdaeb;
  color: #000000;
}
#T_65a08_row2_col0 {
  background-color: #e8e4f0;
  color: #000000;
}
#T_65a08_row2_col1, #T_65a08_row6_col5 {
  background-color: #f3edf5;
  color: #000000;
}
#T_65a08_row2_col3 {
  background-color: #efe9f3;
  color: #000000;
}
#T_65a08_row2_col4, #T_65a08_row2_col7, #T_65a08_row9_col12, #T_65a08_row11_col1, #T_65a08_row12_col2 {
  background-color: #f2ecf5;
  color: #000000;
}
#T_65a08_row2_col11, #T_65a08_row7_col11 {
  background-color: #e9e5f1;
  color: #000000;
}
#T_65a08_row2_col12, #T_65a08_row6_col10 {
  background-color: #dfddec;
  color: #000000;
}
#T_65a08_row3_col0, #T_65a08_row3_col12 {
  background-color: #ebe6f2;
  color: #000000;
}
#T_65a08_row3_col1, #T_65a08_row4_col1, #T_65a08_row6_col7, #T_65a08_row6_col9, #T_65a08_row7_col8, #T_65a08_row10_col8 {
  background-color: #f8f1f8;
  color: #000000;
}
#T_65a08_row3_col2, #T_65a08_row8_col9, #T_65a08_row12_col1 {
  background-color: #eee9f3;
  color: #000000;
}
#T_65a08_row3_col4 {
  background-color: #a9bfdc;
  color: #000000;
}
#T_65a08_row3_col5, #T_65a08_row3_col7, #T_65a08_row12_col6 {
  background-color: #f5eff6;
  color: #000000;
}
#T_65a08_row3_col6, #T_65a08_row11_col4 {
  background-color: #d4d4e8;
  color: #000000;
}
#T_65a08_row3_col10 {
  background-color: #3f93c2;
  color: #f1f1f1;
}
#T_65a08_row3_col11 {
  background-color: #c9cee4;
  color: #000000;
}
#T_65a08_row4_col3 {
  background-color: #b1c2de;
  color: #000000;
}
#T_65a08_row4_col6, #T_65a08_row10_col0, #T_65a08_row11_col6 {
  background-color: #e6e2ef;
  color: #000000;
}
#T_65a08_row4_col9, #T_65a08_row12_col8 {
  background-color: #f1ebf5;
  color: #000000;
}
#T_65a08_row4_col10 {
  background-color: #c2cbe2;
  color: #000000;
}
#T_65a08_row4_col11 {
  background-color: #c1cae2;
  color: #000000;
}
#T_65a08_row4_col12 {
  background-color: #d8d7e9;
  color: #000000;
}
#T_65a08_row5_col0, #T_65a08_row9_col4 {
  background-color: #f1ebf4;
  color: #000000;
}
#T_65a08_row5_col2, #T_65a08_row5_col8 {
  background-color: #fef6fa;
  color: #000000;
}
#T_65a08_row5_col12, #T_65a08_row6_col3 {
  background-color: #d6d6e9;
  color: #000000;
}
#T_65a08_row6_col0 {
  background-color: #c8cde4;
  color: #000000;
}
#T_65a08_row6_col11 {
  background-color: #d2d2e7;
  color: #000000;
}
#T_65a08_row6_col12 {
  background-color: #e4e1ef;
  color: #000000;
}
#T_65a08_row7_col0, #T_65a08_row8_col0, #T_65a08_row11_col9, #T_65a08_row12_col4 {
  background-color: #e7e3f0;
  color: #000000;
}
#T_65a08_row7_col12 {
  background-color: #a4bcda;
  color: #000000;
}
#T_65a08_row8_col11 {
  background-color: #ede8f3;
  color: #000000;
}
#T_65a08_row8_col12 {
  background-color: #e0dded;
  color: #000000;
}
#T_65a08_row9_col5, #T_65a08_row10_col9 {
  background-color: #f4edf6;
  color: #000000;
}
#T_65a08_row9_col11 {
  background-color: #d7d6e9;
  color: #000000;
}
#T_65a08_row10_col3 {
  background-color: #3b92c1;
  color: #f1f1f1;
}
#T_65a08_row10_col4, #T_65a08_row12_col7 {
  background-color: #b7c5df;
  color: #000000;
}
#T_65a08_row10_col6 {
  background-color: #dad9ea;
  color: #000000;
}
#T_65a08_row10_col11 {
  background-color: #c6cce3;
  color: #000000;
}
#T_65a08_row11_col0 {
  background-color: #71a8ce;
  color: #f1f1f1;
}
#T_65a08_row11_col3 {
  background-color: #e0deed;
  color: #000000;
}
#T_65a08_row12_col5 {
  background-color: #e5e1ef;
  color: #000000;
}
</style>
<table id="T_65a08">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_65a08_level0_col0" class="col_heading level0 col0" >Age</th>
      <th id="T_65a08_level0_col1" class="col_heading level0 col1" >CityTier</th>
      <th id="T_65a08_level0_col2" class="col_heading level0 col2" >DurationOfPitch</th>
      <th id="T_65a08_level0_col3" class="col_heading level0 col3" >NumberOfPersonVisiting</th>
      <th id="T_65a08_level0_col4" class="col_heading level0 col4" >NumberOfFollowups</th>
      <th id="T_65a08_level0_col5" class="col_heading level0 col5" >PreferredPropertyStar</th>
      <th id="T_65a08_level0_col6" class="col_heading level0 col6" >NumberOfTrips</th>
      <th id="T_65a08_level0_col7" class="col_heading level0 col7" >Passport</th>
      <th id="T_65a08_level0_col8" class="col_heading level0 col8" >PitchSatisfactionScore</th>
      <th id="T_65a08_level0_col9" class="col_heading level0 col9" >OwnCar</th>
      <th id="T_65a08_level0_col10" class="col_heading level0 col10" >NumberOfChildrenVisiting</th>
      <th id="T_65a08_level0_col11" class="col_heading level0 col11" >MonthlyIncome</th>
      <th id="T_65a08_level0_col12" class="col_heading level0 col12" >ProdTaken</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_65a08_level0_row0" class="row_heading level0 row0" >Age</th>
      <td id="T_65a08_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_65a08_row0_col1" class="data row0 col1" >0.007875</td>
      <td id="T_65a08_row0_col2" class="data row0 col2" >0.025779</td>
      <td id="T_65a08_row0_col3" class="data row0 col3" >0.010795</td>
      <td id="T_65a08_row0_col4" class="data row0 col4" >0.009834</td>
      <td id="T_65a08_row0_col5" class="data row0 col5" >-0.026789</td>
      <td id="T_65a08_row0_col6" class="data row0 col6" >0.178143</td>
      <td id="T_65a08_row0_col7" class="data row0 col7" >0.030162</td>
      <td id="T_65a08_row0_col8" class="data row0 col8" >0.032860</td>
      <td id="T_65a08_row0_col9" class="data row0 col9" >0.060298</td>
      <td id="T_65a08_row0_col10" class="data row0 col10" >0.039495</td>
      <td id="T_65a08_row0_col11" class="data row0 col11" >0.440733</td>
      <td id="T_65a08_row0_col12" class="data row0 col12" >-0.135832</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row1" class="row_heading level0 row1" >CityTier</th>
      <td id="T_65a08_row1_col0" class="data row1 col0" >0.007875</td>
      <td id="T_65a08_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_65a08_row1_col2" class="data row1 col2" >0.056010</td>
      <td id="T_65a08_row1_col3" class="data row1 col3" >0.018071</td>
      <td id="T_65a08_row1_col4" class="data row1 col4" >0.023532</td>
      <td id="T_65a08_row1_col5" class="data row1 col5" >-0.011882</td>
      <td id="T_65a08_row1_col6" class="data row1 col6" >-0.020887</td>
      <td id="T_65a08_row1_col7" class="data row1 col7" >0.013665</td>
      <td id="T_65a08_row1_col8" class="data row1 col8" >-0.028168</td>
      <td id="T_65a08_row1_col9" class="data row1 col9" >0.014177</td>
      <td id="T_65a08_row1_col10" class="data row1 col10" >0.025359</td>
      <td id="T_65a08_row1_col11" class="data row1 col11" >0.057705</td>
      <td id="T_65a08_row1_col12" class="data row1 col12" >0.085583</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row2" class="row_heading level0 row2" >DurationOfPitch</th>
      <td id="T_65a08_row2_col0" class="data row2 col0" >0.025779</td>
      <td id="T_65a08_row2_col1" class="data row2 col1" >0.056010</td>
      <td id="T_65a08_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_65a08_row2_col3" class="data row2 col3" >0.096268</td>
      <td id="T_65a08_row2_col4" class="data row2 col4" >0.039485</td>
      <td id="T_65a08_row2_col5" class="data row2 col5" >-0.004448</td>
      <td id="T_65a08_row2_col6" class="data row2 col6" >0.022236</td>
      <td id="T_65a08_row2_col7" class="data row2 col7" >0.043478</td>
      <td id="T_65a08_row2_col8" class="data row2 col8" >0.011926</td>
      <td id="T_65a08_row2_col9" class="data row2 col9" >-0.015087</td>
      <td id="T_65a08_row2_col10" class="data row2 col10" >0.047770</td>
      <td id="T_65a08_row2_col11" class="data row2 col11" >0.016011</td>
      <td id="T_65a08_row2_col12" class="data row2 col12" >0.072899</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row3" class="row_heading level0 row3" >NumberOfPersonVisiting</th>
      <td id="T_65a08_row3_col0" class="data row3 col0" >0.010795</td>
      <td id="T_65a08_row3_col1" class="data row3 col1" >0.018071</td>
      <td id="T_65a08_row3_col2" class="data row3 col2" >0.096268</td>
      <td id="T_65a08_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_65a08_row3_col4" class="data row3 col4" >0.333738</td>
      <td id="T_65a08_row3_col5" class="data row3 col5" >0.017057</td>
      <td id="T_65a08_row3_col6" class="data row3 col6" >0.214895</td>
      <td id="T_65a08_row3_col7" class="data row3 col7" >0.023638</td>
      <td id="T_65a08_row3_col8" class="data row3 col8" >-0.012981</td>
      <td id="T_65a08_row3_col9" class="data row3 col9" >0.018545</td>
      <td id="T_65a08_row3_col10" class="data row3 col10" >0.610193</td>
      <td id="T_65a08_row3_col11" class="data row3 col11" >0.168701</td>
      <td id="T_65a08_row3_col12" class="data row3 col12" >0.006483</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row4" class="row_heading level0 row4" >NumberOfFollowups</th>
      <td id="T_65a08_row4_col0" class="data row4 col0" >0.009834</td>
      <td id="T_65a08_row4_col1" class="data row4 col1" >0.023532</td>
      <td id="T_65a08_row4_col2" class="data row4 col2" >0.039485</td>
      <td id="T_65a08_row4_col3" class="data row4 col3" >0.333738</td>
      <td id="T_65a08_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_65a08_row4_col5" class="data row4 col5" >-0.049151</td>
      <td id="T_65a08_row4_col6" class="data row4 col6" >0.135183</td>
      <td id="T_65a08_row4_col7" class="data row4 col7" >-0.005332</td>
      <td id="T_65a08_row4_col8" class="data row4 col8" >-0.007195</td>
      <td id="T_65a08_row4_col9" class="data row4 col9" >0.051920</td>
      <td id="T_65a08_row4_col10" class="data row4 col10" >0.293942</td>
      <td id="T_65a08_row4_col11" class="data row4 col11" >0.194668</td>
      <td id="T_65a08_row4_col12" class="data row4 col12" >0.105038</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row5" class="row_heading level0 row5" >PreferredPropertyStar</th>
      <td id="T_65a08_row5_col0" class="data row5 col0" >-0.026789</td>
      <td id="T_65a08_row5_col1" class="data row5 col1" >-0.011882</td>
      <td id="T_65a08_row5_col2" class="data row5 col2" >-0.004448</td>
      <td id="T_65a08_row5_col3" class="data row5 col3" >0.017057</td>
      <td id="T_65a08_row5_col4" class="data row5 col4" >-0.049151</td>
      <td id="T_65a08_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_65a08_row5_col6" class="data row5 col6" >0.035064</td>
      <td id="T_65a08_row5_col7" class="data row5 col7" >0.014701</td>
      <td id="T_65a08_row5_col8" class="data row5 col8" >-0.019620</td>
      <td id="T_65a08_row5_col9" class="data row5 col9" >0.031355</td>
      <td id="T_65a08_row5_col10" class="data row5 col10" >0.027038</td>
      <td id="T_65a08_row5_col11" class="data row5 col11" >-0.024338</td>
      <td id="T_65a08_row5_col12" class="data row5 col12" >0.114923</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row6" class="row_heading level0 row6" >NumberOfTrips</th>
      <td id="T_65a08_row6_col0" class="data row6 col0" >0.178143</td>
      <td id="T_65a08_row6_col1" class="data row6 col1" >-0.020887</td>
      <td id="T_65a08_row6_col2" class="data row6 col2" >0.022236</td>
      <td id="T_65a08_row6_col3" class="data row6 col3" >0.214895</td>
      <td id="T_65a08_row6_col4" class="data row6 col4" >0.135183</td>
      <td id="T_65a08_row6_col5" class="data row6 col5" >0.035064</td>
      <td id="T_65a08_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_65a08_row6_col7" class="data row6 col7" >0.004418</td>
      <td id="T_65a08_row6_col8" class="data row6 col8" >0.034816</td>
      <td id="T_65a08_row6_col9" class="data row6 col9" >0.005982</td>
      <td id="T_65a08_row6_col10" class="data row6 col10" >0.189517</td>
      <td id="T_65a08_row6_col11" class="data row6 col11" >0.137093</td>
      <td id="T_65a08_row6_col12" class="data row6 col12" >0.044922</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row7" class="row_heading level0 row7" >Passport</th>
      <td id="T_65a08_row7_col0" class="data row7 col0" >0.030162</td>
      <td id="T_65a08_row7_col1" class="data row7 col1" >0.013665</td>
      <td id="T_65a08_row7_col2" class="data row7 col2" >0.043478</td>
      <td id="T_65a08_row7_col3" class="data row7 col3" >0.023638</td>
      <td id="T_65a08_row7_col4" class="data row7 col4" >-0.005332</td>
      <td id="T_65a08_row7_col5" class="data row7 col5" >0.014701</td>
      <td id="T_65a08_row7_col6" class="data row7 col6" >0.004418</td>
      <td id="T_65a08_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_65a08_row7_col8" class="data row7 col8" >0.018526</td>
      <td id="T_65a08_row7_col9" class="data row7 col9" >-0.045133</td>
      <td id="T_65a08_row7_col10" class="data row7 col10" >0.030512</td>
      <td id="T_65a08_row7_col11" class="data row7 col11" >0.017044</td>
      <td id="T_65a08_row7_col12" class="data row7 col12" >0.293726</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row8" class="row_heading level0 row8" >PitchSatisfactionScore</th>
      <td id="T_65a08_row8_col0" class="data row8 col0" >0.032860</td>
      <td id="T_65a08_row8_col1" class="data row8 col1" >-0.028168</td>
      <td id="T_65a08_row8_col2" class="data row8 col2" >0.011926</td>
      <td id="T_65a08_row8_col3" class="data row8 col3" >-0.012981</td>
      <td id="T_65a08_row8_col4" class="data row8 col4" >-0.007195</td>
      <td id="T_65a08_row8_col5" class="data row8 col5" >-0.019620</td>
      <td id="T_65a08_row8_col6" class="data row8 col6" >0.034816</td>
      <td id="T_65a08_row8_col7" class="data row8 col7" >0.018526</td>
      <td id="T_65a08_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_65a08_row8_col9" class="data row8 col9" >0.073097</td>
      <td id="T_65a08_row8_col10" class="data row8 col10" >0.023842</td>
      <td id="T_65a08_row8_col11" class="data row8 col11" >-0.005497</td>
      <td id="T_65a08_row8_col12" class="data row8 col12" >0.067736</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row9" class="row_heading level0 row9" >OwnCar</th>
      <td id="T_65a08_row9_col0" class="data row9 col0" >0.060298</td>
      <td id="T_65a08_row9_col1" class="data row9 col1" >0.014177</td>
      <td id="T_65a08_row9_col2" class="data row9 col2" >-0.015087</td>
      <td id="T_65a08_row9_col3" class="data row9 col3" >0.018545</td>
      <td id="T_65a08_row9_col4" class="data row9 col4" >0.051920</td>
      <td id="T_65a08_row9_col5" class="data row9 col5" >0.031355</td>
      <td id="T_65a08_row9_col6" class="data row9 col6" >0.005982</td>
      <td id="T_65a08_row9_col7" class="data row9 col7" >-0.045133</td>
      <td id="T_65a08_row9_col8" class="data row9 col8" >0.073097</td>
      <td id="T_65a08_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_65a08_row9_col10" class="data row9 col10" >0.036416</td>
      <td id="T_65a08_row9_col11" class="data row9 col11" >0.109662</td>
      <td id="T_65a08_row9_col12" class="data row9 col12" >-0.040465</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row10" class="row_heading level0 row10" >NumberOfChildrenVisiting</th>
      <td id="T_65a08_row10_col0" class="data row10 col0" >0.039495</td>
      <td id="T_65a08_row10_col1" class="data row10 col1" >0.025359</td>
      <td id="T_65a08_row10_col2" class="data row10 col2" >0.047770</td>
      <td id="T_65a08_row10_col3" class="data row10 col3" >0.610193</td>
      <td id="T_65a08_row10_col4" class="data row10 col4" >0.293942</td>
      <td id="T_65a08_row10_col5" class="data row10 col5" >0.027038</td>
      <td id="T_65a08_row10_col6" class="data row10 col6" >0.189517</td>
      <td id="T_65a08_row10_col7" class="data row10 col7" >0.030512</td>
      <td id="T_65a08_row10_col8" class="data row10 col8" >0.023842</td>
      <td id="T_65a08_row10_col9" class="data row10 col9" >0.036416</td>
      <td id="T_65a08_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_65a08_row10_col11" class="data row10 col11" >0.179255</td>
      <td id="T_65a08_row10_col12" class="data row10 col12" >0.006060</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row11" class="row_heading level0 row11" >MonthlyIncome</th>
      <td id="T_65a08_row11_col0" class="data row11 col0" >0.440733</td>
      <td id="T_65a08_row11_col1" class="data row11 col1" >0.057705</td>
      <td id="T_65a08_row11_col2" class="data row11 col2" >0.016011</td>
      <td id="T_65a08_row11_col3" class="data row11 col3" >0.168701</td>
      <td id="T_65a08_row11_col4" class="data row11 col4" >0.194668</td>
      <td id="T_65a08_row11_col5" class="data row11 col5" >-0.024338</td>
      <td id="T_65a08_row11_col6" class="data row11 col6" >0.137093</td>
      <td id="T_65a08_row11_col7" class="data row11 col7" >0.017044</td>
      <td id="T_65a08_row11_col8" class="data row11 col8" >-0.005497</td>
      <td id="T_65a08_row11_col9" class="data row11 col9" >0.109662</td>
      <td id="T_65a08_row11_col10" class="data row11 col10" >0.179255</td>
      <td id="T_65a08_row11_col11" class="data row11 col11" >1.000000</td>
      <td id="T_65a08_row11_col12" class="data row11 col12" >-0.140617</td>
    </tr>
    <tr>
      <th id="T_65a08_level0_row12" class="row_heading level0 row12" >ProdTaken</th>
      <td id="T_65a08_row12_col0" class="data row12 col0" >-0.135832</td>
      <td id="T_65a08_row12_col1" class="data row12 col1" >0.085583</td>
      <td id="T_65a08_row12_col2" class="data row12 col2" >0.072899</td>
      <td id="T_65a08_row12_col3" class="data row12 col3" >0.006483</td>
      <td id="T_65a08_row12_col4" class="data row12 col4" >0.105038</td>
      <td id="T_65a08_row12_col5" class="data row12 col5" >0.114923</td>
      <td id="T_65a08_row12_col6" class="data row12 col6" >0.044922</td>
      <td id="T_65a08_row12_col7" class="data row12 col7" >0.293726</td>
      <td id="T_65a08_row12_col8" class="data row12 col8" >0.067736</td>
      <td id="T_65a08_row12_col9" class="data row12 col9" >-0.040465</td>
      <td id="T_65a08_row12_col10" class="data row12 col10" >0.006060</td>
      <td id="T_65a08_row12_col11" class="data row12 col11" >-0.140617</td>
      <td id="T_65a08_row12_col12" class="data row12 col12" >1.000000</td>
    </tr>
  </tbody>
</table>




상관 계수의 경우,
- 0.8 <= r : 강한 상관
- 0.6 <= r < 0.8 : 상관
- 0.4 <= r < 0.6 : 약한 상관


```python
corr = pd.DataFrame(df.drop(columns=["id", "ProdTaken"]).corr(method="pearson"))

plt.figure(figsize=(12, 5))

mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,
            cmap="RdYlBu_r",
            annot=True,
            mask=mask,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            vmin=-1,
            vmax=1
);
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_51_0.png)
    


#### 가설 설정
1. 중장년층의 급여가 높을것이다 (`Age-MonthlyIncome`)
2. 총 여행 인원이 많은 경우는 아이가 많은 경우일 것이다 (`NumberOfPersonVisiting-NumberOfChildrenVisiting`)

##### [가설] 중장년층의 급여가 높을 것이다


```python
sns.scatterplot(data=df, x="Age", y="MonthlyIncome");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_53_0.png)
    


30대 후반부터 임금이 증가해 비슷한 추이를 갖는 모습을 볼 수 있음

##### [가설] 총 여행 인원이 많은 경우는 아이가 많은 경우일 것이다


```python
sns.scatterplot(data=df, x="NumberOfChildrenVisiting", y="NumberOfPersonVisiting");
```


    
![png](Visualize%20Travel%20Product%20Analysis_files/Visualize%20Travel%20Product%20Analysis_55_0.png)
    

