---
title: 깃허브 페이지에서 Plotly 그래프 출력하기
date: 2022-05-26T00:55:09.080Z

categories:
  - GitHub Page
tags:
  - Plotly
  - iframe
---

# 깃허브 페이지(블로그)에서 Plotly 그래프 출력하기
파이썬에는 다양한 시각화 툴이 존재하지만, 개인적으로 Plotly가 가장 깔끔하고 세련된 느낌을 받아 앞으로 시각화에는 Plotly를 사용하고 싶다는 생각이 강하게 들었다.  

한가지 문제가 있었다면, plotly에서 출력으로 나오는 그래프는 깃허브에서 랜더링이되지 않아 사용할 수 없다는 거였는데 이것저것 찾아보다 일단 해결법을 찾았다.  
생각보다 관련된 자료들이 별로 없어서 [stackoverflow](https://stackoverflow.com/questions/72378397/about-plotly-how-can-i-post-it-for-my-github-pages?noredirect=1#comment127864773_72378397)에도 질문을 했는데, 원하던 답변을 얻지는 못했다. 그러다 맨땅에 해딩까지는 아니어도 이것저것해보면서 해결법을 찾았다.

# iframe
찾은 해결법은 iframe을 이용한 방식이다. 그래프를 html 파일로 변환해 이용하는 방법도 있었지만 생각보다 잘되지 않았다. 물론 해당 분야에 관련 지식이 없어서 실패한것도 있겠지만..  

plotly에서 제공하는 iframe url을 이용해 마크다운 파일에 적용하는 방식이다. 주피터 노트북으로 생성하고 마크다운 파일로 주피터 노트북을 변환할때 iframe이 변환되는지는 모르겠지만 이건 나중에 테스트해봐야할 부분이다.

# iframe 이용하기
먼저 `plotly`와 `chart_studio` 라이브러리를 함께 이용해야하고, plotly의 api키도 발급 받아야한다.  

plotly 홈페이지에서 무료로 가입을하고 api키를 발급 받으면 사용할 준비는 끝난다.

## import 
```python
import plotly.express as px
import chart_studio
```
위 2개의 라이브러리를 사용한다.

## api 설정
```python
username = "plotly 닉네임"
api_key = "plotly에서 발급 받은 api 키"
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
```

## iframe 링크 발급 받기
그래프를 변수에 담아 저장하고 이용합니다.
```python
chart_studio.plotly.plot(fig, filename="파일 이름", auto_open=True)
```
`filename`에는 원하는 파일 이름을 적고, `auto_open`을 `True`로 옵션을 주면 웹페이지에서 창을 열어준다.  
결과로 나오는 링크에 들어가, embed 창에서 iframe url을 받아 마크다운 파일에 집어 넣으면 사용 가능하다.

## embed
웹페이지에서 직접 가져오는 방식을 사용하고 있지만,  
```python
chart_studio.tools.get_embed(url)
```
을 이용하면 자동으로 `iframe` 형식이 나오는거 같다. 해당 부분을 결과로 받아서 주피터노트북을 마크다운으로 변환하면 자동적으로 될꺼 같다는 생각이 들었다.  
`url`에는 `chart_studio.plotly.plot(fig, filename="파일 이름", auto_open=True)`의 결과로 나오는 주소를 넣으면 된다.

## 최종 iframe
```
<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plotly.com/내 주소.embed"></iframe>
```
위와 같은 형식의 주소가 나오는데, 해당 주소를 그대로 마크다운에서 원하는 부분에 집어 넣으면된다.  
width와 height를 각각 100%, 525로 설정하는게 페이지에서 보기 괜찮은거 같다.

# 정리
일반적으로 사용할 방식
```python
import plotly.express as px
import chart_studio as cs

# 그래프 그리기
fig = px.line(df, x=, y=, title=)
fig.show() # 그래프 출력

cs.tools.get_embed(cs.plotly.plot(fig, filename="파일 이름", auto_open=False))
```
위와 같이 코드를 작성하면, 주피터에서 그래프 결과도 보고 iframe 주소도 출력되어 마크다운으로 변경해 바로 사용 가능하지 않을까.. 라고 희망을 갖는다.

# 추가
해당 방식을 이용하다보니, 무료 계정에서는 웹에 올릴 수 있는 plotly의 파일 개수가 한정되어 있음..  
plotly의 경우 데이터를 상세하게 정제해서 이용해야한다는 점과 마크다운 형식으로 변경했을 때, 바로 사용하지 못한다는 점으로 인해  
개인적으로 사용은하나 일일이 게시불에 임베드 시키지는 않을 예정..  
웹에 있는 파일을 모두 지워서 게시물 중에 plotly 그래프가 404로 찾을 수 없게 됨
 