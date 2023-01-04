---
title: "스크래핑을 이용해 네이버 금융 정보(뉴스, 시세) 수집하기"
date: 2022-05-18T16:19:40.935Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - requests
  - beautifulsoup
---
# 네이버 금융 정보 수집 (스크래핑)

## Pandas만으로 데이터 수집하기 (뉴스 데이터)


```python
import pandas as pd
```

### url 설정하기
일반적인 url을 이용하면되는 줄 알았는데, 웹 브라우저 개발자 도구를 통해 url을 찾아야한다.  
카카오페이 뉴스 헤드라인이 있는 url 주소가 `https://finance.naver.com/item/news_news.naver?code=377300&page=1&sm=title_entity_id.basic&clusterId=`식으로 구성되어 있다.  
해당 링크로 접속해보면, `code` 부분은 카카오페이 등록 코드로 고정되어 있고, `page`의 값만 뉴스의 게시판 수에 따라 변화한다.


```python
# 쿼리 스트링에 사용할 변수는 일단 선언하고 추후에 변경 가능
item_code="377300"
item_name="카카오페이"
page_no = 1
# 뉴스 기사를 스크래핑 해 올 베이스 쿼리 스트링 생성
url = f"https://finance.naver.com/item/news_news.naver?code={item_code}&page={page_no}&sm=title_entity_id.basic&clusterId="
```

### read_html
`read_html`은 pandas에서 html의 table 태그를 가져오는 함수이다.  
내부는 requests 모듈로 이뤄져있다는데, 모든 상황에서 사용하는것은 불가능한거 같다.  
예외 사항 없이 스크래핑을하기 위해서는 조금 불편해도 처음부터 requests 모듈과 bs 모듈을 이용하는게 편한거 같다.  

`read_html`의 경우, 인코딩값을 함께 넘겨줘야하는데
- cp949: 한글 윈도우에서 많이 사용하는 인코딩
- utf-8: 범용적으로 많이 사용하는 인코딩  

정도로만 이해하고 있다.



```python
table = pd.read_html(url, encoding='cp949')
```


```python
len(table)
```




    6



일단 카카오페이의 뉴스 1페이지에 있는 기사 데이터를 스크래핑 해왔다.  
결과는 6개의 테이블이 존재하는데,


```python
table[0]
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
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>12</th>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>18</th>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
  </tbody>
</table>
</div>




```python
table[1]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
  </tbody>
</table>
</div>




```python
table[2]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
  </tbody>
</table>
</div>




```python
table[3]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
  </tbody>
</table>
</div>




```python
table[4]
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
  </tbody>
</table>
</div>




```python
table[5]
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
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>다음</td>
      <td>맨뒤</td>
    </tr>
  </tbody>
</table>
</div>



이 과정을 직접 안해봐서 개념에 조금 혼돈이 왔었다.


```python
df = table[0]
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
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>12</th>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>18</th>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
  </tbody>
</table>
</div>



### 모든 페이지의 데이터 가져오기
모든 테이블의 정보를 하나의 데이터프레임으로 합치고, 컬럼의 이름도 통일 시켜준다.  
tabel[0]을 기준으로 보면, `{"제목":0, "정보제공":1, "날짜":2}` 형식으로 볼 수 있다.


```python
cols = table[0].columns # Index(['제목', '정보제공', '날짜'], dtype='object')
temp_list = []
for _ in table[:-1]:
    _.columns = cols
    temp_list.append(_)
    
temp_list
```




    [                                                   제목  \
     0                      진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인   
     1                 '카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는   
     2                            실적 부진에 주가도…카카오페이 반등 전략은?   
     3                카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'   
     4                          “출구가 없다”…카카오페이·카카오뱅크 또 신저가   
     5   연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...   
     6                  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'   
     7             [특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세   
     8   연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...   
     9                             [특징주] 카카오페이, 장초반 신저가 경신   
     10                          카카오페이, 장초반 공모가 아래로…신저가 경신   
     11                          4거래일 연속 하락… 카카오페이 공모가도 위태   
     12  연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...   
     13              카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까   
     14                       카카오페이, 4거래일 연속 하락 끝 공모가마저 위태   
     15          공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...   
     16              美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로   
     17                      카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화   
     18  연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...   
     19                      [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대   
     20                              카카오페이 9만원도 붕괴…공모가 밑돌아   
     21                   25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아   
     22         24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...   
     
                                                      정보제공  \
     0                                                조선비즈   
     1                                               머니투데이   
     2                                               머니투데이   
     3                                               머니투데이   
     4                                              이코노미스트   
     5   연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...   
     6                                               머니투데이   
     7                                               아시아경제   
     8   연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...   
     9                                                서울경제   
     10                                               한국경제   
     11                                             파이낸셜뉴스   
     12  연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...   
     13                                              머니투데이   
     14                                             파이낸셜뉴스   
     15                                             매경이코노미   
     16                                               매일경제   
     17                                               한국경제   
     18  연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...   
     19                                               조선비즈   
     20                                               이데일리   
     21                                              머니투데이   
     22                                               매일경제   
     
                                                        날짜  
     0                                    2022.05.16 11:11  
     1                                    2022.05.16 08:00  
     2                                    2022.05.15 13:06  
     3                                    2022.05.12 17:45  
     4                                    2022.05.12 11:26  
     5   연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...  
     6                                    2022.05.12 09:46  
     7                                    2022.05.12 10:13  
     8   연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...  
     9                                    2022.05.12 09:52  
     10                                   2022.05.12 09:37  
     11                                   2022.05.10 18:09  
     12  연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...  
     13                                   2022.05.10 16:59  
     14                                   2022.05.10 16:30  
     15                                   2022.05.10 15:34  
     16                                   2022.05.10 16:15  
     17                                   2022.05.10 09:59  
     18  연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...  
     19                                   2022.05.10 09:27  
     20                                   2022.05.10 09:26  
     21                                   2022.05.10 09:25  
     22                                   2022.05.08 11:32  ,
                                        제목   정보제공                날짜
     0  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머니투데이  2022.05.12 09:46,
                               제목  정보제공                날짜
     0    [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.05.12 09:52
     1  카카오페이, 장초반 공모가 아래로…신저가 경신  한국경제  2022.05.12 09:37,
                                               제목    정보제공                날짜
     0      카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까   머니투데이  2022.05.10 16:59
     1               카카오페이, 4거래일 연속 하락 끝 공모가마저 위태  파이낸셜뉴스  2022.05.10 16:30
     2  공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...  매경이코노미  2022.05.10 15:34,
                                      제목   정보제공                날짜
     0     [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대   조선비즈  2022.05.10 09:27
     1             카카오페이 9만원도 붕괴…공모가 밑돌아   이데일리  2022.05.10 09:26
     2  25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아  머니투데이  2022.05.10 09:25]




```python
df = pd.concat(temp_list)
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
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
      <td>연관기사 목록  공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'  머...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>8</th>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
      <td>연관기사 목록  [특징주] 카카오페이, 장초반 신저가 경신  서울경제  2022.0...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>12</th>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
      <td>연관기사 목록  카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>18</th>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
      <td>연관기사 목록  [특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대  조선비즈  ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
    <tr>
      <th>0</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
  </tbody>
</table>
</div>



**1 페이지에서 가져온 모든 정보**를 하나의 데이터프레임으로 결합 해줬다.  
중간중간 중복되는 값들도 있고 결측치도 있는데, 이부분을 제거해준다.


```python
df = df.dropna(how="all", axis=0) # 결측치 제거
df = df[~df["제목"].str.contains("연관기사")] # 연관 기사 제거

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
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>13</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
    <tr>
      <th>0</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
  </tbody>
</table>
</div>



## read_html을 이용한 과정 요약
다른 종목 코드나, 페이지를 넣어도 수집할 수 있도록 위 과정을 일반화 시켜 하나의 함수로 작성해줬다.


```python
import pandas as pd

def get_one_page_news(item_code, page_no=1):
    url = f"https://finance.naver.com/item/news_news.naver?code={item_code}&page={page_no}&sm=title_entity_id.basic&clusterId="
    # 스크래핑
    table = pd.read_html(url, encoding="cp949")
    
    cols = table[0].columns
    temp_list = []
    for _ in table[:-1]:
        _.columns = cols
        temp_list.append(_)
    # 결측치 제거    
    df = pd.concat(temp_list).dropna(how="all", axis=0)
    # 연관 기사를 반환하고 반환
    return df[~df["제목"].str.contains("연관기사")]
```


```python
get_one_page_news(item_code)
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
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>13</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>14</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>15</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>16</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>17</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>20</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>21</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
    <tr>
      <th>22</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
    <tr>
      <th>0</th>
      <td>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이, 장초반 신저가 경신</td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 장초반 공모가 아래로…신저가 경신</td>
      <td>한국경제</td>
      <td>2022.05.12 09:37</td>
    </tr>
    <tr>
      <th>0</th>
      <td>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 16:30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</td>
      <td>매경이코노미</td>
      <td>2022.05.10 15:34</td>
    </tr>
    <tr>
      <th>0</th>
      <td>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>카카오페이 9만원도 붕괴…공모가 밑돌아</td>
      <td>이데일리</td>
      <td>2022.05.10 09:26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</td>
      <td>머니투데이</td>
      <td>2022.05.10 09:25</td>
    </tr>
  </tbody>
</table>
</div>



pandas 모듈의 read_html을 이용하면 확실히 편하게 데이터를 수집해 올 수 있다.

위 함수와 반복문을 이용하면, 해당 종목과 관련된 모든 데이터를 가져오는것도 가능하다.  
서버에 과부하를 줄수도 있으니 time 모듈을 이용해 반복문에 시간차를 주도록하자.

## requsets 모듈과 bs 모듈로 데이터 수집하기 (뉴스 데이터)
read_html을 통해서 데이터를 수집할 수 없는 경우에 사용한다.  
read_html이 확실히 편하지만, 일반적인 웹사이트에서 일반적인 패턴으로 특정 데이터를 가져오는건 requests와 bs 모듈을 사용하는게 더 편할꺼 같다는 생각이 든다.


```python
import requests
from bs4 import BeautifulSoup as bs
```

### url 설정하기
브라우저 설정에 따라, 요청자의 정보(headers)를 함께 보내야 정보를 가져오는 경우가 있다. 이 부분도 그냥 당연하게 사용하는게 좋을꺼 같다.  
requests의 경우, `post`와 `get`이 있는데 해당 사이트의 웹 브라우저 개발자 도구를 살펴보면 어떤 방식을 사용해야하는지 알 수 있다.  
네이버 금융 탭의 경우 get 방식을 사용해야한다.


```python
headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"}

response = requests.get(url, headers=headers)
```


```python
response.status_code
```




    200



요청한 정보(카카오페이의 뉴스 1페이지의 정보)를 정상적으로 처리했다는 것을 확인할 수 있다.

### 받아온 정보 내용 확인 및 원하는 정보 추출하기


```python
response.text
```




    '<html lang="ko">\n<head>\n<meta http-equiv="Content-Type" content="text/html; charset=euc-kr">\n<title>네이버 금융</title>\n\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/common.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/layout.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/main.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock2.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock3.css">\n<link rel="stylesheet" type="text/css" href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/world.css">\n</head>\n<body>\n<script type="text/javascript" src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/jindo.min.ns.1.5.3.euckr.js"></script>\n<script language="JavaScript">\n//document.domain="naver.com";\nvar nsc="finance.stockend";\n\nfunction mouseOver(obj){\n  obj.style.backgroundColor="#f6f4e5";\n}\nfunction mouseOut(obj){\n  obj.style.backgroundColor="#ffffff";\n}\nfunction showPannel(layerId){\n\tvar layer = jindo.$(layerId);\n\tlayer.style.display=\'block\';\n\t\n\tif (layerId == "summary_lyr") {\n\t\tvar layerHeight = jindo.$Element(layer).height();\n\t\tjindo.$Element("summary_ifr").height(layerHeight);\n\t}\n}\nfunction hidePannel(layerId){\n\tvar layer = jindo.$(layerId);\n\tlayer.style.display=\'none\';\n}\nfunction togglePannel(layerId) {\n\tvar elTargetLayer = jindo.$Element(jindo.$$.getSingle("#" + layerId));\n\t\n\tif (elTargetLayer != null) {\n\t\tif (elTargetLayer.visible()) {\n\t\t\thidePannel(layerId);\n\t\t} else {\n\t\t\tshowPannel(layerId);\n\t\t}\n\t}\n}\nfunction toggleList(classId, clusterId) {\n\tvar up = true;\n\tvar sm = \'title_entity_id.basic\';\n\tif (classId.text == \'접기\') {\n\t\tif (sm == \'title_entity_id.basic\') {\n\t\t\tclickcr(this,\'stn.ntitclustclose\',\'\',\'\',event);\n\t\t} else {\n\t\t\tclickcr(this,\'stn.nbdyclustclose\',\'\',\'\',event);\n\t\t}\n\t\tjindo.$Element(classId).html(\'관련뉴스 <em>\' + classId.getAttribute("data-count") + \'</em>건 더보기<span class="ico_down"></span>\');\n\t\tup = false;\n\t} else {\n\t\tclickcr(this,\'stn.ntitclustclose\',\'\',\'\',event);\n\t\tif (sm == \'title_entity_id.basic\') {\n\t\t\tclickcr(this,\'stn.ntitclustmore\',\'\',\'\',event);\n\t\t} else {\n\t\t\tclickcr(this,\'stn.nbdyclustmore\',\'\',\'\',event);\n\t\t}\n\t\tjindo.$Element(classId).html(\'접기<span class="ico_up"></span>\');\n\t}\n\n\tvar height = 0;\n\tif (document.querySelector("._clusterId"+clusterId)) {\n\t\theight = document.querySelector("._clusterId"+clusterId).offsetHeight - 46;\n\t}\n\n\tjindo.$ElementList(jindo.$$(".hide_news", classId.parentElement.parentElement.parentElement)).toggleClass(\'none\');\n\n\tif (up) {\n\t\tparentHeightResize();\n\t} else {\n\t\tparentHeightResize(document.body.offsetHeight - height);\n\t}\n}\n</script>\n<script type="text/javascript" src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/nclktag.js"></script>\n\t\t<div class="tb_cont">\n\t\t\t<div class="tlline2">\n\t\t\t\t<strong><span class="red03">종목</span>뉴스</strong>\n\t\t\t\t<!-- [D] 선택시 blind 태그 추가 -->\n\t\t\t\t<div class="select_lst">\n\t\t\t\t\t<a href="/item/news.naver?code=377300&sm=title_entity_id.basic" onClick="clickcr(this,\'stn.ntitle\',\'\',\'\',event);" class="on" target=_top><span class="ico_chk"></span>제목<span class="blind">선택됨</span></a>\n\t\t\t\t\t<a href="/item/news.naver?code=377300&sm=entity_id.basic" onClick="clickcr(this,\'stn.nbdy\',\'\',\'\',event);" target=_top><span class="ico_chk"></span>내용<span class="blind">선택됨</span></a>\n\t\t\t\t</div>\n\t\t\t\t<div class="info_txt">종목뉴스 안내<a href="#" onclick="togglePannel(\'layer_section\'); clickcr(this,\'stn.ninfo\',\'\',\'\',event); return false;" class="btn_info_layer"><img src="https://ssl.pstatic.net/static/nfinance/2017/10/13/btn_help2.png" alt="상세 설명"></a>\n\t\t\t\t\t<div class="layer_section" id="layer_section" style="display:none">\n\t\t\t\t\t\t<strong>네이버 증권 종목뉴스 안내</strong>\n\t\t\t\t\t\t<p class="txt">\n\t\t\t\t\t\t\tAI(인공지능 검색기술)을 이용한 종목 관련 뉴스입니다. <br>관련뉴스는 자동으로 묶어 보여줍니다.\n\t\t\t\t\t\t\t<br>\n\t\t\t\t\t\t\t<span class="txt_opt">\n\t\t\t\t\t\t\t\t검색영역 옵션\n\t\t\t\t\t\t\t\t<br>\n\t\t\t\t\t\t\t\t제목 : 제목에서 종목명이 검색된 결과입니다. <br>\n\t\t\t\t\t\t\t\t내용 : 제목과 본문에서 종목명이 검색된 결과입니다.\n\t\t\t\t\t\t\t</span>\n\t\t\t\t\t\t</p>\n\t\t\t\t\t\t<a href="#" onclick="togglePannel(\'layer_section\'); return false;" class="btn_close"><span class="blind">닫기</span></a>\n\t\t\t\t\t</div>\n\t\t\t\t</div>\n\t\t\t</div>\n\t\t\t\n\t\t\t<table summary="종목뉴스의 제목, 정보제공, 날짜" cellspacing="0" class="type5">\n\t\t\t\t<caption>종목뉴스</caption>\n\t\t\t\t<colgroup>\n\t\t\t\t\t<col>\n\t\t\t\t\t<col width="130px">\n\t\t\t\t\t<col width="110px">\n\t\t\t\t</colgroup>\n\t\t\t\t<thead>\n\t\t\t\t\t<tr>\n\t\t\t\t\t\t<th scope="col">제목</th>\n\t\t\t\t\t\t<th scope="col">정보제공</th>\n\t\t\t\t\t\t<th scope="col">날짜</th>\n\t\t\t\t\t</tr>\n\t\t\t\t</thead>\n\n\t\t \t\t<tbody>\n\t\t \t\t\n\t\t \t\t\n\t\t \t\t\n\t\t \t\t<!-- [D] tr class : 첫번째 first, 마지막 last, 연관 뉴스 relation_tit, 연관 뉴스 목록 relation_lst -->\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t class="first"\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0000813881&office_id=366&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">조선비즈</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.16 11:11</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004746144&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>\'카뱅&middot;카카오페이\' 쓰는 사람은 많은데...떨어진 주가 돌파구는</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.16 08:00</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004746009&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>실적 부진에 주가도&hellip;카카오페이 반등 전략은?</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.15 13:06</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004745179&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>카카오페이&middot;카카오뱅크, 공모가 밑으로 \'추락\'&hellip;카카오도 \'와르르\'</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 17:45</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t class="relation_tit"\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0000026345&office_id=243&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>&ldquo;출구가 없다&rdquo;&hellip;카카오페이&middot;카카오뱅크 또 신저가</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">이코노미스트</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 11:26</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t \t\t\t\t\t\t<tr class="relation_lst _clusterId2430000026345">\n\t\t\t\t\t\t\t\t<td colspan="3">\n\t\t\t\t\t\t\t\t\t<table class="type5">\n\t\t\t\t\t\t\t\t\t\t<caption>연관기사 목록</caption>\n\t\t\t\t\t\t\t\t\t\t<colgroup>\n\t\t\t\t\t\t\t\t\t\t\t<col>\n\t\t\t\t\t\t\t\t\t\t\t<col width="130px">\n\t\t\t\t\t\t\t\t\t\t\t<col width="110px">\n\t\t\t\t\t\t\t\t\t\t</colgroup>\n\t\t\t\t\t\t\t\t\t\t<tbody>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  >\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004744775&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>공모가 붕괴? &quot;악!&quot;&hellip;카카오페이&middot;카뱅 개미들 \'멘탈도 붕괴\'</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 09:46</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t</tbody>\n\t\t\t\t\t\t\t\t\t</table>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t class="relation_tit"\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0005087734&office_id=277&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>[특징주]카카오페이 공모가 9만원 아래로 \'뚝\'&hellip;금리인상에 성장주 약세</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">아시아경제</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 10:13</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t \t\t\t\t\t\t<tr class="relation_lst _clusterId2770005087734">\n\t\t\t\t\t\t\t\t<td colspan="3">\n\t\t\t\t\t\t\t\t\t<table class="type5">\n\t\t\t\t\t\t\t\t\t\t<caption>연관기사 목록</caption>\n\t\t\t\t\t\t\t\t\t\t<colgroup>\n\t\t\t\t\t\t\t\t\t\t\t<col>\n\t\t\t\t\t\t\t\t\t\t\t<col width="130px">\n\t\t\t\t\t\t\t\t\t\t\t<col width="110px">\n\t\t\t\t\t\t\t\t\t\t</colgroup>\n\t\t\t\t\t\t\t\t\t\t<tbody>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  >\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004052749&office_id=011&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>[특징주] 카카오페이, 장초반 신저가 경신</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">서울경제</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 09:52</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  class="hide_news none">\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004697724&office_id=015&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>카카오페이, 장초반 공모가 아래로&hellip;신저가 경신</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">한국경제</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.12 09:37</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t</tbody>\n\t\t\t\t\t\t\t\t\t</table>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t<div class="link_area">\n\t\t\t\t\t\t\t\t\t\t<a href="#" class="_moreBtn" onClick="toggleList(this, \'2770005087734\'); return false;" data-count="1">관련뉴스 <em>1</em>건 더보기<span class="ico_down"></span></a>\n\t\t\t\t\t\t\t\t\t</div>\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t class="relation_tit"\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004833931&office_id=014&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>4거래일 연속 하락&hellip; 카카오페이 공모가도 위태</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">파이낸셜뉴스</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 18:09</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t \t\t\t\t\t\t<tr class="relation_lst _clusterId0140004833931">\n\t\t\t\t\t\t\t\t<td colspan="3">\n\t\t\t\t\t\t\t\t\t<table class="type5">\n\t\t\t\t\t\t\t\t\t\t<caption>연관기사 목록</caption>\n\t\t\t\t\t\t\t\t\t\t<colgroup>\n\t\t\t\t\t\t\t\t\t\t\t<col>\n\t\t\t\t\t\t\t\t\t\t\t<col width="130px">\n\t\t\t\t\t\t\t\t\t\t\t<col width="110px">\n\t\t\t\t\t\t\t\t\t\t</colgroup>\n\t\t\t\t\t\t\t\t\t\t<tbody>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  >\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004743945&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>카카오페이, \'공모가 9만원\' 밑으로&hellip;진짜 \'바닥\' 찍은 게 맞을까</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 16:59</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  class="hide_news none">\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004833845&office_id=014&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">파이낸셜뉴스</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 16:30</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  class="hide_news none">\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0000074952&office_id=024&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>공모가마저 깨진 카카오페이&hellip;&lsquo;성장주 약세&middot;대량 매도 우려&rsquo;에 연일 신...</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">매경이코노미</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 15:34</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t</tbody>\n\t\t\t\t\t\t\t\t\t</table>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t<div class="link_area">\n\t\t\t\t\t\t\t\t\t\t<a href="#" class="_moreBtn" onClick="toggleList(this, \'0140004833931\'); return false;" data-count="2">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>\n\t\t\t\t\t\t\t\t\t</div>\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004961784&office_id=009&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>美증시 충격에 코스피 2600선 붕괴&hellip;카카오페이 장중 공모가 밑으로</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">매일경제</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 16:15</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t class="relation_tit"\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004696614&office_id=015&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>카카오페이, 9만원 공모가 붕괴&hellip;오버행&middot;투자심리 악화</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">한국경제</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 09:59</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t \t\t\t\t\t\t<tr class="relation_lst _clusterId0150004696614">\n\t\t\t\t\t\t\t\t<td colspan="3">\n\t\t\t\t\t\t\t\t\t<table class="type5">\n\t\t\t\t\t\t\t\t\t\t<caption>연관기사 목록</caption>\n\t\t\t\t\t\t\t\t\t\t<colgroup>\n\t\t\t\t\t\t\t\t\t\t\t<col>\n\t\t\t\t\t\t\t\t\t\t\t<col width="130px">\n\t\t\t\t\t\t\t\t\t\t\t<col width="110px">\n\t\t\t\t\t\t\t\t\t\t</colgroup>\n\t\t\t\t\t\t\t\t\t\t<tbody>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  >\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0000812275&office_id=366&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>[특징주] 카카오페이 공모가 깨졌다&hellip;장 초반 9만원대</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">조선비즈</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 09:27</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  class="hide_news none">\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0005211440&office_id=018&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>카카오페이 9만원도 붕괴&hellip;공모가 밑돌아</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">이데일리</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 09:26</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t<tr  class="hide_news none">\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004743588&office_id=008&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclustsub\',\'\',\'\',event);" target=_top><span class="ico_reply"></span>25만원 바라보다 9만원 뚫렸다&hellip;카카오페이, 공모가 밑돌아</a>\n\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="info">머니투데이</td>\n\t\t\t\t\t\t\t\t\t\t\t\t<td class="date"> 2022.05.10 09:25</td>\n\t\t\t\t\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\t</tbody>\n\t\t\t\t\t\t\t\t\t</table>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t<div class="link_area">\n\t\t\t\t\t\t\t\t\t\t<a href="#" class="_moreBtn" onClick="toggleList(this, \'0150004696614\'); return false;" data-count="2">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>\n\t\t\t\t\t\t\t\t\t</div>\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t</tr>\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n \t\t\t\t\t\n \t\t\t\t\t\t\n \t\t\t\t\t\t\t\n \t\t\t\t\t\t\t<tr \n \t\t\t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t\t class="last"\n\t\t \t\t\t\t\t\t\t\t\t \t\t\t\t\t\t\t\n\t\t \t\t\t\t\t\t >\t \t\t\t\t\t\t\n\t \t\t\t\t\t\t\t<td class="title">\n\t\t\t\t\t\t\t\t\t<a href="/item/news_read.naver?article_id=0004960722&office_id=009&code=377300&page=1&sm=title_entity_id.basic" class="tit" onClick="clickcr(this,\'stn.ntitclust\',\'\',\'\',event);" target=_top>24만8500원&rarr;9만7100원&hellip;연일 신저가 경신, 카카오페이 주주들 악...</a>\n\t\t\t\t\t\t\t\t\t\n\t\t\t\t\t\t\t\t</td>\n\t\t\t\t\t\t\t\t<td class="info">매일경제</td>\n\t\t\t\t\t\t\t\t<td class="date"> 2022.05.08 11:32</td>\n \t\t\t\t\t\t\t</tr>\t \t\t\t\t\t\n \t\t\t\t\t\t\n\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t\t\t\t\t\n \t\t\t\t\t\n\t\t\t\t\t \t\t\t\t\t\n \t\t\t\t\n\t\t\t\t</tbody>\n\t\t\t</table>\n \t\t\t\t\t\t\n\t\t\t<!--- 종목뉴스 끝--->\n\t\t\t<!--- 페이지 네비게이션 시작--->\n\t\t\t\n\t\t\t<table summary="페이지 네비게이션 리스트" class="Nnavi" align="center">\n\t\t\t\t<caption>페이지 네비게이션</caption>\n\t\t\t\t<tr>\n\t\t\t\t\t\n\t\t\t\t\t\t\n\t\t                \n\t\t                <td class="on">\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=1&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>1</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=2&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>2</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=3&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>3</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=4&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>4</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=5&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>5</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=6&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>6</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=7&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>7</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=8&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>8</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=9&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>9</a>\n\t\t\t\t</td>\n<td>\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=10&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>10</a>\n\t\t\t\t</td>\n\n\t\t                <td class="pgR">\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=11&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>\n\t\t\t\t다음<img src="https://ssl.pstatic.net/static/n/cmn/bu_pgarR.gif" width="3" height="5" alt="" border="0">\n\t\t\t\t</a>\n\t\t\t\t</td>\n\n\t\t                <td class="pgRR">\n\t\t\t\t<a href="/item/news_news.naver?code=377300&amp;page=140&amp;sm=title_entity_id.basic&amp;clusterId="  onclick=clickcr(this,\'stn.npag\',\'\',\'\',event);>맨뒤\n\t\t\t\t<img src="https://ssl.pstatic.net/static/n/cmn/bu_pgarRR.gif" width="8" height="5" alt="" border="0">\n\t\t\t\t</a>\n\t\t\t\t</td>\n\n\t\t            \n\t\t\t\t</tr>\n\t\t\t</table>\n\t\t\t\n\n\n\t<script type="text/javascript" src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/lcslog.js"></script>\n    <script type="text/javascript">\n        ;(function(){\n            var eventType = "onpageshow" in window ? "pageshow" : "load";\n            jindo.$Fn(function(){\n                lcs_do();\n            }).attach(window, eventType);\n        })();\n\t</script>\t\n\n\n<script type="text/javascript">\n\tjindo.$Fn(function(){\n\t\tparentHeightResize();\n\t}).attach(window, \'load\');\n</script>\n<script>\n\tvar targetWindow = window.top;\n\tvar targetOrigin = location.protocol+\'//\'+location.host;\n\n\tfunction parentHeightResize(height) {\n\t\twindow.scrollTo(0, 0);\n\t\ttargetWindow.postMessage({\n\t\t\t"type":"syncHeight",\n\t\t\t"height": (height) ? height : document.body.offsetHeight\n\t\t}, targetOrigin);\n\t}\n\n\tfunction parentScrollTo(to) {\n\t\ttargetWindow.postMessage({\n\t\t\t"type":"syncScrollTo",\n\t\t\t"to": to\n\t\t}, targetOrigin);\n\t}\n\n\t// iframe의 사이즈가 갱신되기 위해서 dom load되기전 300으로 먼저 수정\n\tparentHeightResize(300);\n</script>\n</body>\n'



`response.text`로 가져온 값을 그대로 가져오면 보기가 힘들다.  
보기 편하게 하기 위함과 원하는 정보를 찾기 위한 연산을하기 위해 bs 모듈을 사용한다.  
bs 모듈의 파서는 여러 종류가 있지만, `lxml` 파서가 가장 빠르다고 한다.


```python
html = bs(response.text, "lxml")
html
```




    <html lang="ko">
    <head>
    <meta content="text/html; charset=utf-8" http-equiv="Content-Type"/>
    <title>네이버 금융</title>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/common.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/layout.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/main.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock2.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/newstock3.css" rel="stylesheet" type="text/css"/>
    <link href="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/css/world.css" rel="stylesheet" type="text/css"/>
    </head>
    <body>
    <script src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/jindo.min.ns.1.5.3.euckr.js" type="text/javascript"></script>
    <script language="JavaScript">
    //document.domain="naver.com";
    var nsc="finance.stockend";
    
    function mouseOver(obj){
      obj.style.backgroundColor="#f6f4e5";
    }
    function mouseOut(obj){
      obj.style.backgroundColor="#ffffff";
    }
    function showPannel(layerId){
    	var layer = jindo.$(layerId);
    	layer.style.display='block';
    	
    	if (layerId == "summary_lyr") {
    		var layerHeight = jindo.$Element(layer).height();
    		jindo.$Element("summary_ifr").height(layerHeight);
    	}
    }
    function hidePannel(layerId){
    	var layer = jindo.$(layerId);
    	layer.style.display='none';
    }
    function togglePannel(layerId) {
    	var elTargetLayer = jindo.$Element(jindo.$$.getSingle("#" + layerId));
    	
    	if (elTargetLayer != null) {
    		if (elTargetLayer.visible()) {
    			hidePannel(layerId);
    		} else {
    			showPannel(layerId);
    		}
    	}
    }
    function toggleList(classId, clusterId) {
    	var up = true;
    	var sm = 'title_entity_id.basic';
    	if (classId.text == '접기') {
    		if (sm == 'title_entity_id.basic') {
    			clickcr(this,'stn.ntitclustclose','','',event);
    		} else {
    			clickcr(this,'stn.nbdyclustclose','','',event);
    		}
    		jindo.$Element(classId).html('관련뉴스 <em>' + classId.getAttribute("data-count") + '</em>건 더보기<span class="ico_down"></span>');
    		up = false;
    	} else {
    		clickcr(this,'stn.ntitclustclose','','',event);
    		if (sm == 'title_entity_id.basic') {
    			clickcr(this,'stn.ntitclustmore','','',event);
    		} else {
    			clickcr(this,'stn.nbdyclustmore','','',event);
    		}
    		jindo.$Element(classId).html('접기<span class="ico_up"></span>');
    	}
    
    	var height = 0;
    	if (document.querySelector("._clusterId"+clusterId)) {
    		height = document.querySelector("._clusterId"+clusterId).offsetHeight - 46;
    	}
    
    	jindo.$ElementList(jindo.$$(".hide_news", classId.parentElement.parentElement.parentElement)).toggleClass('none');
    
    	if (up) {
    		parentHeightResize();
    	} else {
    		parentHeightResize(document.body.offsetHeight - height);
    	}
    }
    </script>
    <script src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/nclktag.js" type="text/javascript"></script>
    <div class="tb_cont">
    <div class="tlline2">
    <strong><span class="red03">종목</span>뉴스</strong>
    <!-- [D] 선택시 blind 태그 추가 -->
    <div class="select_lst">
    <a class="on" href="/item/news.naver?code=377300&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitle','','',event);" target="_top"><span class="ico_chk"></span>제목<span class="blind">선택됨</span></a>
    <a href="/item/news.naver?code=377300&amp;sm=entity_id.basic" onclick="clickcr(this,'stn.nbdy','','',event);" target="_top"><span class="ico_chk"></span>내용<span class="blind">선택됨</span></a>
    </div>
    <div class="info_txt">종목뉴스 안내<a class="btn_info_layer" href="#" onclick="togglePannel('layer_section'); clickcr(this,'stn.ninfo','','',event); return false;"><img alt="상세 설명" src="https://ssl.pstatic.net/static/nfinance/2017/10/13/btn_help2.png"/></a>
    <div class="layer_section" id="layer_section" style="display:none">
    <strong>네이버 증권 종목뉴스 안내</strong>
    <p class="txt">
    							AI(인공지능 검색기술)을 이용한 종목 관련 뉴스입니다. <br/>관련뉴스는 자동으로 묶어 보여줍니다.
    							<br/>
    <span class="txt_opt">
    								검색영역 옵션
    								<br/>
    								제목 : 제목에서 종목명이 검색된 결과입니다. <br/>
    								내용 : 제목과 본문에서 종목명이 검색된 결과입니다.
    							</span>
    </p>
    <a class="btn_close" href="#" onclick="togglePannel('layer_section'); return false;"><span class="blind">닫기</span></a>
    </div>
    </div>
    </div>
    <table cellspacing="0" class="type5" summary="종목뉴스의 제목, 정보제공, 날짜">
    <caption>종목뉴스</caption>
    <colgroup>
    <col/>
    <col width="130px"/>
    <col width="110px"/>
    </colgroup>
    <thead>
    <tr>
    <th scope="col">제목</th>
    <th scope="col">정보제공</th>
    <th scope="col">날짜</th>
    </tr>
    </thead>
    <tbody>
    <!-- [D] tr class : 첫번째 first, 마지막 last, 연관 뉴스 relation_tit, 연관 뉴스 목록 relation_lst -->
    <tr class="first">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0000813881&amp;office_id=366&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</a>
    </td>
    <td class="info">조선비즈</td>
    <td class="date"> 2022.05.16 11:11</td>
    </tr>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004746144&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.16 08:00</td>
    </tr>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004746009&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">실적 부진에 주가도…카카오페이 반등 전략은?</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.15 13:06</td>
    </tr>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004745179&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.12 17:45</td>
    </tr>
    <tr class="relation_tit">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0000026345&amp;office_id=243&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">“출구가 없다”…카카오페이·카카오뱅크 또 신저가</a>
    </td>
    <td class="info">이코노미스트</td>
    <td class="date"> 2022.05.12 11:26</td>
    </tr>
    <tr class="relation_lst _clusterId2430000026345">
    <td colspan="3">
    <table class="type5">
    <caption>연관기사 목록</caption>
    <colgroup>
    <col/>
    <col width="130px"/>
    <col width="110px"/>
    </colgroup>
    <tbody>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004744775&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.12 09:46</td>
    </tr>
    </tbody>
    </table>
    </td>
    </tr>
    <tr class="relation_tit">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0005087734&amp;office_id=277&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</a>
    </td>
    <td class="info">아시아경제</td>
    <td class="date"> 2022.05.12 10:13</td>
    </tr>
    <tr class="relation_lst _clusterId2770005087734">
    <td colspan="3">
    <table class="type5">
    <caption>연관기사 목록</caption>
    <colgroup>
    <col/>
    <col width="130px"/>
    <col width="110px"/>
    </colgroup>
    <tbody>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004052749&amp;office_id=011&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>[특징주] 카카오페이, 장초반 신저가 경신</a>
    </td>
    <td class="info">서울경제</td>
    <td class="date"> 2022.05.12 09:52</td>
    </tr>
    <tr class="hide_news none">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004697724&amp;office_id=015&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, 장초반 공모가 아래로…신저가 경신</a>
    </td>
    <td class="info">한국경제</td>
    <td class="date"> 2022.05.12 09:37</td>
    </tr>
    </tbody>
    </table>
    <div class="link_area">
    <a class="_moreBtn" data-count="1" href="#" onclick="toggleList(this, '2770005087734'); return false;">관련뉴스 <em>1</em>건 더보기<span class="ico_down"></span></a>
    </div>
    </td>
    </tr>
    <tr class="relation_tit">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004833931&amp;office_id=014&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">4거래일 연속 하락… 카카오페이 공모가도 위태</a>
    </td>
    <td class="info">파이낸셜뉴스</td>
    <td class="date"> 2022.05.10 18:09</td>
    </tr>
    <tr class="relation_lst _clusterId0140004833931">
    <td colspan="3">
    <table class="type5">
    <caption>연관기사 목록</caption>
    <colgroup>
    <col/>
    <col width="130px"/>
    <col width="110px"/>
    </colgroup>
    <tbody>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004743945&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.10 16:59</td>
    </tr>
    <tr class="hide_news none">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004833845&amp;office_id=014&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</a>
    </td>
    <td class="info">파이낸셜뉴스</td>
    <td class="date"> 2022.05.10 16:30</td>
    </tr>
    <tr class="hide_news none">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0000074952&amp;office_id=024&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</a>
    </td>
    <td class="info">매경이코노미</td>
    <td class="date"> 2022.05.10 15:34</td>
    </tr>
    </tbody>
    </table>
    <div class="link_area">
    <a class="_moreBtn" data-count="2" href="#" onclick="toggleList(this, '0140004833931'); return false;">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>
    </div>
    </td>
    </tr>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004961784&amp;office_id=009&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</a>
    </td>
    <td class="info">매일경제</td>
    <td class="date"> 2022.05.10 16:15</td>
    </tr>
    <tr class="relation_tit">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004696614&amp;office_id=015&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</a>
    </td>
    <td class="info">한국경제</td>
    <td class="date"> 2022.05.10 09:59</td>
    </tr>
    <tr class="relation_lst _clusterId0150004696614">
    <td colspan="3">
    <table class="type5">
    <caption>연관기사 목록</caption>
    <colgroup>
    <col/>
    <col width="130px"/>
    <col width="110px"/>
    </colgroup>
    <tbody>
    <tr>
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0000812275&amp;office_id=366&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</a>
    </td>
    <td class="info">조선비즈</td>
    <td class="date"> 2022.05.10 09:27</td>
    </tr>
    <tr class="hide_news none">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0005211440&amp;office_id=018&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이 9만원도 붕괴…공모가 밑돌아</a>
    </td>
    <td class="info">이데일리</td>
    <td class="date"> 2022.05.10 09:26</td>
    </tr>
    <tr class="hide_news none">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004743588&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</a>
    </td>
    <td class="info">머니투데이</td>
    <td class="date"> 2022.05.10 09:25</td>
    </tr>
    </tbody>
    </table>
    <div class="link_area">
    <a class="_moreBtn" data-count="2" href="#" onclick="toggleList(this, '0150004696614'); return false;">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>
    </div>
    </td>
    </tr>
    <tr class="last">
    <td class="title">
    <a class="tit" href="/item/news_read.naver?article_id=0004960722&amp;office_id=009&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</a>
    </td>
    <td class="info">매일경제</td>
    <td class="date"> 2022.05.08 11:32</td>
    </tr>
    </tbody>
    </table>
    <!--- 종목뉴스 끝--->
    <!--- 페이지 네비게이션 시작--->
    <table align="center" class="Nnavi" summary="페이지 네비게이션 리스트">
    <caption>페이지 네비게이션</caption>
    <tr>
    <td class="on">
    <a href="/item/news_news.naver?code=377300&amp;page=1&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">1</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=2&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">2</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=3&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">3</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=4&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">4</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=5&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">5</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=6&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">6</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=7&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">7</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=8&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">8</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=9&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">9</a>
    </td>
    <td>
    <a href="/item/news_news.naver?code=377300&amp;page=10&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">10</a>
    </td>
    <td class="pgR">
    <a href="/item/news_news.naver?code=377300&amp;page=11&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">
    				다음<img alt="" border="0" height="5" src="https://ssl.pstatic.net/static/n/cmn/bu_pgarR.gif" width="3"/>
    </a>
    </td>
    <td class="pgRR">
    <a href="/item/news_news.naver?code=377300&amp;page=140&amp;sm=title_entity_id.basic&amp;clusterId=" onclick="clickcr(this,'stn.npag','','',event);">맨뒤
    				<img alt="" border="0" height="5" src="https://ssl.pstatic.net/static/n/cmn/bu_pgarRR.gif" width="8"/>
    </a>
    </td>
    </tr>
    </table>
    <script src="https://ssl.pstatic.net/imgstock/static.pc/20220511192542/js/lcslog.js" type="text/javascript"></script>
    <script type="text/javascript">
            ;(function(){
                var eventType = "onpageshow" in window ? "pageshow" : "load";
                jindo.$Fn(function(){
                    lcs_do();
                }).attach(window, eventType);
            })();
    	</script>
    <script type="text/javascript">
    	jindo.$Fn(function(){
    		parentHeightResize();
    	}).attach(window, 'load');
    </script>
    <script>
    	var targetWindow = window.top;
    	var targetOrigin = location.protocol+'//'+location.host;
    
    	function parentHeightResize(height) {
    		window.scrollTo(0, 0);
    		targetWindow.postMessage({
    			"type":"syncHeight",
    			"height": (height) ? height : document.body.offsetHeight
    		}, targetOrigin);
    	}
    
    	function parentScrollTo(to) {
    		targetWindow.postMessage({
    			"type":"syncScrollTo",
    			"to": to
    		}, targetOrigin);
    	}
    
    	// iframe의 사이즈가 갱신되기 위해서 dom load되기전 300으로 먼저 수정
    	parentHeightResize(300);
    </script>
    </div></body>
    </html>



read_html을 사용할 경우 제목, 정보제공, 날짜를 `tr`태그로 가져옴  
개발자도구에서 select path copy 기능을 사용


```python
tables = html.select("body > div > table.type5 > tbody > tr")
tables
```




    [<tr class="first">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0000813881&amp;office_id=366&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</a>
     </td>
     <td class="info">조선비즈</td>
     <td class="date"> 2022.05.16 11:11</td>
     </tr>,
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004746144&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.16 08:00</td>
     </tr>,
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004746009&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">실적 부진에 주가도…카카오페이 반등 전략은?</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.15 13:06</td>
     </tr>,
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004745179&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.12 17:45</td>
     </tr>,
     <tr class="relation_tit">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0000026345&amp;office_id=243&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">“출구가 없다”…카카오페이·카카오뱅크 또 신저가</a>
     </td>
     <td class="info">이코노미스트</td>
     <td class="date"> 2022.05.12 11:26</td>
     </tr>,
     <tr class="relation_lst _clusterId2430000026345">
     <td colspan="3">
     <table class="type5">
     <caption>연관기사 목록</caption>
     <colgroup>
     <col/>
     <col width="130px"/>
     <col width="110px"/>
     </colgroup>
     <tbody>
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004744775&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>공모가 붕괴? "악!"…카카오페이·카뱅 개미들 '멘탈도 붕괴'</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.12 09:46</td>
     </tr>
     </tbody>
     </table>
     </td>
     </tr>,
     <tr class="relation_tit">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0005087734&amp;office_id=277&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</a>
     </td>
     <td class="info">아시아경제</td>
     <td class="date"> 2022.05.12 10:13</td>
     </tr>,
     <tr class="relation_lst _clusterId2770005087734">
     <td colspan="3">
     <table class="type5">
     <caption>연관기사 목록</caption>
     <colgroup>
     <col/>
     <col width="130px"/>
     <col width="110px"/>
     </colgroup>
     <tbody>
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004052749&amp;office_id=011&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>[특징주] 카카오페이, 장초반 신저가 경신</a>
     </td>
     <td class="info">서울경제</td>
     <td class="date"> 2022.05.12 09:52</td>
     </tr>
     <tr class="hide_news none">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004697724&amp;office_id=015&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, 장초반 공모가 아래로…신저가 경신</a>
     </td>
     <td class="info">한국경제</td>
     <td class="date"> 2022.05.12 09:37</td>
     </tr>
     </tbody>
     </table>
     <div class="link_area">
     <a class="_moreBtn" data-count="1" href="#" onclick="toggleList(this, '2770005087734'); return false;">관련뉴스 <em>1</em>건 더보기<span class="ico_down"></span></a>
     </div>
     </td>
     </tr>,
     <tr class="relation_tit">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004833931&amp;office_id=014&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">4거래일 연속 하락… 카카오페이 공모가도 위태</a>
     </td>
     <td class="info">파이낸셜뉴스</td>
     <td class="date"> 2022.05.10 18:09</td>
     </tr>,
     <tr class="relation_lst _clusterId0140004833931">
     <td colspan="3">
     <table class="type5">
     <caption>연관기사 목록</caption>
     <colgroup>
     <col/>
     <col width="130px"/>
     <col width="110px"/>
     </colgroup>
     <tbody>
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004743945&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, '공모가 9만원' 밑으로…진짜 '바닥' 찍은 게 맞을까</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.10 16:59</td>
     </tr>
     <tr class="hide_news none">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004833845&amp;office_id=014&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이, 4거래일 연속 하락 끝 공모가마저 위태</a>
     </td>
     <td class="info">파이낸셜뉴스</td>
     <td class="date"> 2022.05.10 16:30</td>
     </tr>
     <tr class="hide_news none">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0000074952&amp;office_id=024&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>공모가마저 깨진 카카오페이…‘성장주 약세·대량 매도 우려’에 연일 신...</a>
     </td>
     <td class="info">매경이코노미</td>
     <td class="date"> 2022.05.10 15:34</td>
     </tr>
     </tbody>
     </table>
     <div class="link_area">
     <a class="_moreBtn" data-count="2" href="#" onclick="toggleList(this, '0140004833931'); return false;">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>
     </div>
     </td>
     </tr>,
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004961784&amp;office_id=009&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</a>
     </td>
     <td class="info">매일경제</td>
     <td class="date"> 2022.05.10 16:15</td>
     </tr>,
     <tr class="relation_tit">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004696614&amp;office_id=015&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</a>
     </td>
     <td class="info">한국경제</td>
     <td class="date"> 2022.05.10 09:59</td>
     </tr>,
     <tr class="relation_lst _clusterId0150004696614">
     <td colspan="3">
     <table class="type5">
     <caption>연관기사 목록</caption>
     <colgroup>
     <col/>
     <col width="130px"/>
     <col width="110px"/>
     </colgroup>
     <tbody>
     <tr>
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0000812275&amp;office_id=366&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>[특징주] 카카오페이 공모가 깨졌다…장 초반 9만원대</a>
     </td>
     <td class="info">조선비즈</td>
     <td class="date"> 2022.05.10 09:27</td>
     </tr>
     <tr class="hide_news none">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0005211440&amp;office_id=018&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>카카오페이 9만원도 붕괴…공모가 밑돌아</a>
     </td>
     <td class="info">이데일리</td>
     <td class="date"> 2022.05.10 09:26</td>
     </tr>
     <tr class="hide_news none">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004743588&amp;office_id=008&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclustsub','','',event);" target="_top"><span class="ico_reply"></span>25만원 바라보다 9만원 뚫렸다…카카오페이, 공모가 밑돌아</a>
     </td>
     <td class="info">머니투데이</td>
     <td class="date"> 2022.05.10 09:25</td>
     </tr>
     </tbody>
     </table>
     <div class="link_area">
     <a class="_moreBtn" data-count="2" href="#" onclick="toggleList(this, '0150004696614'); return false;">관련뉴스 <em>2</em>건 더보기<span class="ico_down"></span></a>
     </div>
     </td>
     </tr>,
     <tr class="last">
     <td class="title">
     <a class="tit" href="/item/news_read.naver?article_id=0004960722&amp;office_id=009&amp;code=377300&amp;page=1&amp;sm=title_entity_id.basic" onclick="clickcr(this,'stn.ntitclust','','',event);" target="_top">24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</a>
     </td>
     <td class="info">매일경제</td>
     <td class="date"> 2022.05.08 11:32</td>
     </tr>]




```python
# .tit 타이틀 .info 정보제공 .date 날짜
temp_list = []
for _ in range(len(tables)):
    s = (str(tables[_].select(".tit")).split('>')[1].split('<')[0], str(tables[_].select(".info")).split('>')[1].split('<')[0], str(tables[_].select(".date")).split('>')[1].split('<')[0])
    temp_list.append(s)
```


```python
cols = [["제목", "정보제공", "날짜"]]

df = pd.DataFrame(temp_list)
df.columns = cols

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

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
  </tbody>
</table>
</div>



결측치 및 중복값도 제거 해줘야하는데, 해당 과정에서는 어떻게해야할지 모르겠다..

## requests와 bs를 이용한 과정 요약
pandas를 사용하는 이유는 반복문을 적게 사용하기 위함이라고 했는데, requests와 bs를 이용해 같은 과정을 구하는걸 해보니 이해가 됬다.  
일단 html에 대한 이해가 없으니 너무 힘들었다.. 중간중간 쓸모 없는 코드가 있을수도 있지만 일단은 read_html과 같은 결과가 비슷하게 나오게 만들긴했다.


```python
import requests
from bs4 import BeautifulSoup as bs

def get_one_page_news_rb(item_code, page_no=1):
    url = f"https://finance.naver.com/item/news_news.naver?code={item_code}&page={page_no}&sm=title_entity_id.basic&clusterId="
    headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"}

    html = bs(requests.get(url, headers=headers).text, "lxml")
    tables = html.select("body > div > table.type5 > tbody > tr")
    # .tit 타이틀 .info 정보제공 .date 날짜
    temp_list = []
    for _ in range(len(tables)):
        s = (str(tables[_].select(".tit")).split('>')[1].split('<')[0], str(tables[_].select(".info")).split('>')[1].split('<')[0], str(tables[_].select(".date")).split('>')[1].split('<')[0])
        temp_list.append(s)
    cols = [["제목", "정보제공", "날짜"]]

    df = pd.DataFrame(temp_list)
    df.columns = cols
    
    # 결측치와 중복값을 제거해줘야함
        
    return df
    
```


```python
get_one_page_news_rb(item_code)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>제목</th>
      <th>정보제공</th>
      <th>날짜</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>진에어, 카카오페이로 항공권 결제하면 최대 3만원 할인</td>
      <td>조선비즈</td>
      <td>2022.05.16 11:11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>'카뱅·카카오페이' 쓰는 사람은 많은데...떨어진 주가 돌파구는</td>
      <td>머니투데이</td>
      <td>2022.05.16 08:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>실적 부진에 주가도…카카오페이 반등 전략은?</td>
      <td>머니투데이</td>
      <td>2022.05.15 13:06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>카카오페이·카카오뱅크, 공모가 밑으로 '추락'…카카오도 '와르르'</td>
      <td>머니투데이</td>
      <td>2022.05.12 17:45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>“출구가 없다”…카카오페이·카카오뱅크 또 신저가</td>
      <td>이코노미스트</td>
      <td>2022.05.12 11:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>머니투데이</td>
      <td>2022.05.12 09:46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[특징주]카카오페이 공모가 9만원 아래로 '뚝'…금리인상에 성장주 약세</td>
      <td>아시아경제</td>
      <td>2022.05.12 10:13</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>서울경제</td>
      <td>2022.05.12 09:52</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4거래일 연속 하락… 카카오페이 공모가도 위태</td>
      <td>파이낸셜뉴스</td>
      <td>2022.05.10 18:09</td>
    </tr>
    <tr>
      <th>9</th>
      <td></td>
      <td>머니투데이</td>
      <td>2022.05.10 16:59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>美증시 충격에 코스피 2600선 붕괴…카카오페이 장중 공모가 밑으로</td>
      <td>매일경제</td>
      <td>2022.05.10 16:15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>카카오페이, 9만원 공모가 붕괴…오버행·투자심리 악화</td>
      <td>한국경제</td>
      <td>2022.05.10 09:59</td>
    </tr>
    <tr>
      <th>12</th>
      <td></td>
      <td>조선비즈</td>
      <td>2022.05.10 09:27</td>
    </tr>
    <tr>
      <th>13</th>
      <td>24만8500원→9만7100원…연일 신저가 경신, 카카오페이 주주들 악...</td>
      <td>매일경제</td>
      <td>2022.05.08 11:32</td>
    </tr>
  </tbody>
</table>
</div>



## Pandas 코드 한 줄로 시세 정보 가져오기


```python
def get_day_list(item_code, page_no=1):
    url = f"https://finance.naver.com/item/sise_day.naver?code={item_code}&page={page_no}"
    headers = {"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36"}
    response = requests.get(url, headers=headers)
    html = bs(response.text, "lxml")
    tables = html.select("table")
    table = pd.read_html(str(tables), encoding='cp949')
    return table[0].dropna()
```


```python
get_day_list(item_code)
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
      <th>날짜</th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022.05.18</td>
      <td>89200.0</td>
      <td>500.0</td>
      <td>89600.0</td>
      <td>90900.0</td>
      <td>88700.0</td>
      <td>237954.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022.05.17</td>
      <td>88700.0</td>
      <td>2900.0</td>
      <td>85800.0</td>
      <td>88700.0</td>
      <td>85400.0</td>
      <td>214643.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022.05.16</td>
      <td>85800.0</td>
      <td>200.0</td>
      <td>86500.0</td>
      <td>88700.0</td>
      <td>85700.0</td>
      <td>277584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022.05.13</td>
      <td>86000.0</td>
      <td>100.0</td>
      <td>85900.0</td>
      <td>86600.0</td>
      <td>85000.0</td>
      <td>307400.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022.05.12</td>
      <td>85900.0</td>
      <td>5500.0</td>
      <td>90000.0</td>
      <td>90000.0</td>
      <td>85000.0</td>
      <td>512917.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2022.05.11</td>
      <td>91400.0</td>
      <td>500.0</td>
      <td>91000.0</td>
      <td>92800.0</td>
      <td>89700.0</td>
      <td>333197.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2022.05.10</td>
      <td>91900.0</td>
      <td>1900.0</td>
      <td>91200.0</td>
      <td>92000.0</td>
      <td>89700.0</td>
      <td>514324.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2022.05.09</td>
      <td>93800.0</td>
      <td>4000.0</td>
      <td>96000.0</td>
      <td>96700.0</td>
      <td>93400.0</td>
      <td>462857.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2022.05.06</td>
      <td>97800.0</td>
      <td>8700.0</td>
      <td>104500.0</td>
      <td>105000.0</td>
      <td>97100.0</td>
      <td>735556.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2022.05.04</td>
      <td>106500.0</td>
      <td>2000.0</td>
      <td>108000.0</td>
      <td>109500.0</td>
      <td>106000.0</td>
      <td>302514.0</td>
    </tr>
  </tbody>
</table>
</div>



### 모든 페이지의 정보 가져오기


```python
item_list = []

while True:
    item_list.append(get_day_list(item_code, page_no))
    page_no += 1
    if get_day_list(item_code, page_no-1).to_dict() == get_day_list(item_code, page_no).to_dict(): break # 마지막 페이지를 검사하는 부분
```


```python
df = pd.concat(item_list)
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
      <th>날짜</th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022.05.18</td>
      <td>89200.0</td>
      <td>500.0</td>
      <td>89600.0</td>
      <td>90900.0</td>
      <td>88700.0</td>
      <td>237954.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022.05.17</td>
      <td>88700.0</td>
      <td>2900.0</td>
      <td>85800.0</td>
      <td>88700.0</td>
      <td>85400.0</td>
      <td>214643.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022.05.16</td>
      <td>85800.0</td>
      <td>200.0</td>
      <td>86500.0</td>
      <td>88700.0</td>
      <td>85700.0</td>
      <td>277584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022.05.13</td>
      <td>86000.0</td>
      <td>100.0</td>
      <td>85900.0</td>
      <td>86600.0</td>
      <td>85000.0</td>
      <td>307400.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022.05.12</td>
      <td>85900.0</td>
      <td>5500.0</td>
      <td>90000.0</td>
      <td>90000.0</td>
      <td>85000.0</td>
      <td>512917.0</td>
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
      <th>13</th>
      <td>2021.11.09</td>
      <td>147000.0</td>
      <td>6500.0</td>
      <td>153500.0</td>
      <td>157500.0</td>
      <td>146000.0</td>
      <td>892617.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021.11.08</td>
      <td>153500.0</td>
      <td>16500.0</td>
      <td>168500.0</td>
      <td>169500.0</td>
      <td>152000.0</td>
      <td>1394625.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021.11.05</td>
      <td>170000.0</td>
      <td>1000.0</td>
      <td>167500.0</td>
      <td>179000.0</td>
      <td>167500.0</td>
      <td>1597937.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021.11.04</td>
      <td>169000.0</td>
      <td>24000.0</td>
      <td>190000.0</td>
      <td>191000.0</td>
      <td>166000.0</td>
      <td>3487030.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021.11.03</td>
      <td>193000.0</td>
      <td>13000.0</td>
      <td>180000.0</td>
      <td>230000.0</td>
      <td>173000.0</td>
      <td>11799881.0</td>
    </tr>
  </tbody>
</table>
<p>134 rows × 7 columns</p>
</div>




```python
page_no = 1

item_list = []
prev = ""

while True:
    df_one_page = get_day_list(item_code, page_no)
    curr = df_one_page.iloc[-1, 0]
    
    if curr==prev: break
    
    item_list.append(df_one_page)
    page_no += 1
    
    prev = curr
```


```python
df = pd.concat(item_list)
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
      <th>날짜</th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2022.05.18</td>
      <td>89200.0</td>
      <td>500.0</td>
      <td>89600.0</td>
      <td>90900.0</td>
      <td>88700.0</td>
      <td>237954.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022.05.17</td>
      <td>88700.0</td>
      <td>2900.0</td>
      <td>85800.0</td>
      <td>88700.0</td>
      <td>85400.0</td>
      <td>214643.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022.05.16</td>
      <td>85800.0</td>
      <td>200.0</td>
      <td>86500.0</td>
      <td>88700.0</td>
      <td>85700.0</td>
      <td>277584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022.05.13</td>
      <td>86000.0</td>
      <td>100.0</td>
      <td>85900.0</td>
      <td>86600.0</td>
      <td>85000.0</td>
      <td>307400.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2022.05.12</td>
      <td>85900.0</td>
      <td>5500.0</td>
      <td>90000.0</td>
      <td>90000.0</td>
      <td>85000.0</td>
      <td>512917.0</td>
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
      <th>13</th>
      <td>2021.11.09</td>
      <td>147000.0</td>
      <td>6500.0</td>
      <td>153500.0</td>
      <td>157500.0</td>
      <td>146000.0</td>
      <td>892617.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021.11.08</td>
      <td>153500.0</td>
      <td>16500.0</td>
      <td>168500.0</td>
      <td>169500.0</td>
      <td>152000.0</td>
      <td>1394625.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021.11.05</td>
      <td>170000.0</td>
      <td>1000.0</td>
      <td>167500.0</td>
      <td>179000.0</td>
      <td>167500.0</td>
      <td>1597937.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021.11.04</td>
      <td>169000.0</td>
      <td>24000.0</td>
      <td>190000.0</td>
      <td>191000.0</td>
      <td>166000.0</td>
      <td>3487030.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021.11.03</td>
      <td>193000.0</td>
      <td>13000.0</td>
      <td>180000.0</td>
      <td>230000.0</td>
      <td>173000.0</td>
      <td>11799881.0</td>
    </tr>
  </tbody>
</table>
<p>134 rows × 7 columns</p>
</div>



두 번째 방식은, 마지막 페이지를 확인하는 경우가 첫 번째 방식과 다르다.  
첫 번째 방식에서는 추가적인 접근을 하지만, 두 번째 방식에서는 추가적인 접근이 필요가 없다. 따라서 두 번째 방식이 좀 더 좋은 방식인거 같다.

## 요약


```python
import requests
import time

def get_item_list(item_code, item_name):
    page_no = 1
    item_list = []
    
    while True:
        temp = get_day_list(item_code, page_no)
        if page_no>1:
            if item_list[-1]["날짜"].iloc[0]!=temp["날짜"].iloc[0]:
                pass
            else:
                print(f"{page_no}쪽 완료")
                break
        item_list.append(temp)
        print("*", end="")
        page_no += 1
        time.sleep(0.1)
    
    df = pd.concat(item_list)
    df["종목코드"] = item_code
    df["종목명"] = item_name
    
    cols = ['종목코드', '종목명', '날짜', '종가', '전일비', '시가', '고가', '저가', '거래량']
    df = df[cols]
    
    return df
```


```python
get_item_list(item_code, item_name)
```

    **************15쪽 완료
    




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
      <th>종목코드</th>
      <th>종목명</th>
      <th>날짜</th>
      <th>종가</th>
      <th>전일비</th>
      <th>시가</th>
      <th>고가</th>
      <th>저가</th>
      <th>거래량</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2022.05.18</td>
      <td>89200.0</td>
      <td>500.0</td>
      <td>89600.0</td>
      <td>90900.0</td>
      <td>88700.0</td>
      <td>237954.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2022.05.17</td>
      <td>88700.0</td>
      <td>2900.0</td>
      <td>85800.0</td>
      <td>88700.0</td>
      <td>85400.0</td>
      <td>214643.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2022.05.16</td>
      <td>85800.0</td>
      <td>200.0</td>
      <td>86500.0</td>
      <td>88700.0</td>
      <td>85700.0</td>
      <td>277584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2022.05.13</td>
      <td>86000.0</td>
      <td>100.0</td>
      <td>85900.0</td>
      <td>86600.0</td>
      <td>85000.0</td>
      <td>307400.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2022.05.12</td>
      <td>85900.0</td>
      <td>5500.0</td>
      <td>90000.0</td>
      <td>90000.0</td>
      <td>85000.0</td>
      <td>512917.0</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2021.11.09</td>
      <td>147000.0</td>
      <td>6500.0</td>
      <td>153500.0</td>
      <td>157500.0</td>
      <td>146000.0</td>
      <td>892617.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2021.11.08</td>
      <td>153500.0</td>
      <td>16500.0</td>
      <td>168500.0</td>
      <td>169500.0</td>
      <td>152000.0</td>
      <td>1394625.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2021.11.05</td>
      <td>170000.0</td>
      <td>1000.0</td>
      <td>167500.0</td>
      <td>179000.0</td>
      <td>167500.0</td>
      <td>1597937.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2021.11.04</td>
      <td>169000.0</td>
      <td>24000.0</td>
      <td>190000.0</td>
      <td>191000.0</td>
      <td>166000.0</td>
      <td>3487030.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>377300</td>
      <td>카카오페이</td>
      <td>2021.11.03</td>
      <td>193000.0</td>
      <td>13000.0</td>
      <td>180000.0</td>
      <td>230000.0</td>
      <td>173000.0</td>
      <td>11799881.0</td>
    </tr>
  </tbody>
</table>
<p>134 rows × 9 columns</p>
</div>


