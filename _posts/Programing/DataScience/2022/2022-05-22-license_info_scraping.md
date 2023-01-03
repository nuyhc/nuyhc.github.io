---
title: "라이센스 정보 가져오기"
date: 2022-05-22T06:24:34.659Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
---

# Proj - License Information Scraping
## 1. 소개
[오픈소스SW 라이선스 종합정보시스템](https://olis.or.kr/)에 명시된 [오픈 소스 라이센스 비교표](https://olis.or.kr/license/compareGuide.do)의 정보를 가져와 라이센스에 대한 정보를 제공하는 스크래핑 툴 제작

## 2. 동기 
멋쟁이 사자처럼 AI School 6기 수업 데이터 수집 파트를 공부하게 되었다.  

관련 실습으로 네이버 증권 데이터를 스크래핑해오는 실습이 있어,  
이를 이용해 특정 종목에 대한 시세 정보와 종목 토론방, 관련 뉴스 정보 등을 제공하는 미니 프로젝트를 진행하려고 했다.  

초기 계획했던 미니 프로젝트를 추후에 발전 시켜, 챗봇의 형태로 수집한 정보를 전달하거나 간단한 퀀트를 적용하기 위한 초석으로 사용하려는 생각을 했다.  
깃허브에 관련 레포지토리를 생성하는 과정에서, "어떤 라이센스를 선택하고 사용해야하는가"라는 의문을 가졌고 관련 라이센스를 찾아보다 영감을 얻어 첫 프로젝트 주제로 선정하게 되었다.  

데이터 수집 파트 첫 목차가 **저작권**, 크롤링과 스크래핑의 차이 이해였는데, 관련 정보들의 개념을 공부하면서 해보기 좋은 미니 프로젝트인거 같다.  
라이센스의 이름뿐만 아니라, 세부 사항 검색하는 부분도 시도해봤지만 깔끔하게 구현이 되지 않아서 일단 이름으로 검색하는 부분만 구현했다.

## 3. 개념
### 저작권과 라이센스
해당 개념은 따로 [정리](https://nuyhc.github.io/til/github-license/)했다
### [크롤링과 스크래핑 차이](https://www.promptcloud.com/blog/data-scraping-vs-data-crawling/)
- 웹 크롤러
  - 서로 연결된 URL을 수집하고, 인덱싱(키워드를 통해 URL을 검색할 수 있게 해주는 작업)하기 위해 사용
  - 기존의 복사본을 생성하는 개념
- 웹 스크래퍼
  - 특정 데이터를 추출하는 과정  
  - 크롤링과는 달리 특정 웹 사이트 또는 페이지에서 특정 정보를 가져오는 방식
  - 분석을 위한 특정 데이터를 추출하거나 새로 만드는 개념

## 4. 목표 구현 기능
1. 모든 라이센스 정보를 가져와 표시
2. 입력한 이름과 동일한 특정 라이센스의 정보만을 가져와 표시
3. GPL 라이센스와의 호환성 표시

## 5. 구현

### 1. 모든 라이센스 정보 가져오기
read_html 사용시 큰 문제 없이 원하는 정보를 모두 가져올 수 있었다.


```python
import pandas as pd

url = "https://olis.or.kr/license/compareGuide.do"
```


```python
table = pd.read_html(url, encoding='utf-8')
```


```python
len(table)
```




    2




```python
type(table[0]), type(table[1])
```




    (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)




```python
# 라이선스 비교표 - 라이선스 주요내용
table[0].head()
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
      <th>라이선스 이름</th>
      <th>복제, 배포, 수정의 권한허용</th>
      <th>배포시라이선스사본첨부</th>
      <th>저작권고지사항또는Attribution고지사항 유지</th>
      <th>배포시소스코드제공의무와범위</th>
      <th>조합저작물작성 및타 라이선스배포허용</th>
      <th>수정내용 고지</th>
      <th>명시적특허라이선스의허용</th>
      <th>라이선시가특허소송 제기시라이선스종료</th>
      <th>이름,상표,상호에 대한사용제한</th>
      <th>보증의 부인</th>
      <th>책임의 제한</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Academic Free License</td>
      <td>NaN</td>
      <td>O</td>
      <td>O</td>
      <td>NaN</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adaptive Public License</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>모듈 단위</td>
      <td>O</td>
      <td>O</td>
      <td>선택</td>
      <td>선택</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Affero GNU General Public License 3.0</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>네트워크서비스 포함 전체 코드</td>
      <td>NaN</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apache License 1.1</td>
      <td>O</td>
      <td>NaN</td>
      <td>O</td>
      <td>NaN</td>
      <td>조건부</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apache License 2.0</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>NaN</td>
      <td>O</td>
      <td>NaN</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 라이선스 비교표 - 주요 오픈 라이선스의 GPL 호환성
table[1].head()
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
      <th>오픈 소스 소프트웨어 라이선스</th>
      <th>GPL 2.0 호환</th>
      <th>GPL 3.0 호환</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Academic Free License</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Affero GNU General Public License version 3.0</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Apache License version 1.0</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Apache License version 1.1</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apache License version 2.0</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# GPL 라이센스와의 호환성을 합쳐 하나의 데이터 프레임으로 생성 (목표 구현 3)
tables = pd.concat([table[0], table[1][['GPL 2.0 호환', 'GPL 3.0 호환']]], axis=1)
```

table 변수에는 read_html로 가져온 테이블을 담았는데, 1번 테이블에는 라이선스의 주요내용에 대한 데이터가, 2번 테이블에는 주요 오픈 라이선스의 GPL 호환성에 대한 데이터가 들어있었다.  

간단한 과정이지만, 전체적인 프로그램을 작성할 때 함수화가 되어있으면 편리해서 함수화 시켰다.


```python
def get_license_info_read_html()->pd.DataFrame:
    """read_html을 이용해 테이블 정보를 가져오는 함수
    NaN 값은 보기 좋게 -로 대체
    Returns:
        table (pd.DataFrame): url에서 가져온 데이터
    """
    url = "https://olis.or.kr/license/compareGuide.do"
    table = pd.read_html(url, encoding="utf-8")
    return pd.concat([table[0], table[1][['GPL 2.0 호환', 'GPL 3.0 호환']]], axis=1)
```

### 2. 입력한 이름과 동일한 특정 라이센스의 정보만을 가져와 표시
검색에 용이하게, 라이선스의 이름을 소문자로 바꾼 컬럼을 추가해 이용 (`search_key`)  
`str.contains` 매서드를 이용해 정확한 이름이 아니라도 유사한 이름의 라이선스 정보를 표시하게함


```python
def search_by_name(df:pd.DataFrame, name:str)->pd.DataFrame:
    """입력한 이름의 라이센스 정보를 검색해 출력해주는 함수
    대소문자 구분 없이, str.contains 매서드를 이용해 모두 검색 가능하게 구현
    라이선스 이름을 소문자로 바꾼 search_key 컬럼을 생성하고 이용하지만, 표시되는 결과에는 출력되지 않는다

    Args:
        df (pd.DataFrame): 모든 라이센스들의 정보
        name (str): 검색하고자하는 라이센스의 이름
    Returns:
        pd.DataFrame: 검색 결과
    """
    df["search_key"] = df["라이선스 이름"].str.lower()
    
    return df[df["search_key"].str.contains(name)].iloc[:, [_ for _ in range(0,14)]].fillna('-')
```


```python
table = get_license_info_read_html()

name = input().lower()

search_by_name(table, name)
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
      <th>라이선스 이름</th>
      <th>복제, 배포, 수정의 권한허용</th>
      <th>배포시라이선스사본첨부</th>
      <th>저작권고지사항또는Attribution고지사항 유지</th>
      <th>배포시소스코드제공의무와범위</th>
      <th>조합저작물작성 및타 라이선스배포허용</th>
      <th>수정내용 고지</th>
      <th>명시적특허라이선스의허용</th>
      <th>라이선시가특허소송 제기시라이선스종료</th>
      <th>이름,상표,상호에 대한사용제한</th>
      <th>보증의 부인</th>
      <th>책임의 제한</th>
      <th>GPL 2.0 호환</th>
      <th>GPL 3.0 호환</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Apache License 1.1</td>
      <td>O</td>
      <td>-</td>
      <td>O</td>
      <td>-</td>
      <td>조건부</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Apache License 2.0</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>-</td>
      <td>O</td>
      <td>-</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>O</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



## 6. 요약


```python
import pandas as pd

def get_license_info_read_html()->pd.DataFrame:
    """read_html을 이용해 테이블 정보를 가져오는 함수
    NaN 값은 보기 좋게 -로 대체
    Returns:
        table (pd.DataFrame): url에서 가져온 데이터
    """
    url = "https://olis.or.kr/license/compareGuide.do"
    table = pd.read_html(url, encoding="utf-8")
    return pd.concat([table[0], table[1][['GPL 2.0 호환', 'GPL 3.0 호환']]], axis=1)

def search_by_name(df:pd.DataFrame, name:str)->pd.DataFrame:
    """입력한 이름의 라이센스 정보를 검색해 출력해주는 함수
    대소문자 구분 없이, str.contains 매서드를 이용해 모두 검색 가능하게 구현
    라이선스 이름을 소문자로 바꾼 search_key 컬럼을 생성하고 이용하지만, 표시되는 결과에는 출력되지 않는다

    Args:
        df (pd.DataFrame): 모든 라이센스들의 정보
        name (str): 검색하고자하는 라이센스의 이름
    Returns:
        pd.DataFrame: 검색 결과
    """
    df["search_key"] = df["라이선스 이름"].str.lower()
    
    return df[df["search_key"].str.contains(name)].iloc[:, [_ for _ in range(0,14)]].fillna('-')
```

url 부분도 변동 없이 처음에 지정한 url을 계속해서 사용해 원하는 데이터를 불러 올 수 있어서 데이터를 가져오는 부분은 어렵지 않았다.  
이름으로 검색하는 부분도 `str` 메서드를 사용해 쉽게 구현 가능했지만, 특정 조건을 입력해 만족하는 라이센스 정보들만 가져오는 부분은 고민하다 구현하지 못했다.  
마지막까지 가장 가능성이 높다고 생각한 구현 방식은, 데이터 프레임의 컬럼명을 딕셔너리로 가져와서 이용하는 방식이었는데 이후의 방식이 문제였다.  
각 컬럼 별로 조건을 만족하는 데이터프레임을 가져오고 `concat`으로 붙이는 방법이었는데 이것보다는 더 좋은 방법이 있을꺼 같아서 좀 더 생각을 해봐야할꺼 같다.
