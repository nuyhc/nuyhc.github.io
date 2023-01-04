---
title: 깃허브 프로필 꾸미기
date: 2022-05-11T07:41:03.396Z

categories:
  - TIL
tags:
  - Git
  - GitHub
---

# 깃허브 프로필 꾸미기
사실 깃허브 프로필에 딱히 관심이 없었는데 갑자기 너무 허전한거 같아서 좀 꾸미고 싶어졌다.  
구글링을하면서 여러 사람들의 깃허브 프로필을 봤는데 이쁜것도 많고 신기한것도 많았다.  
무엇보다 백준 등급을 추가할 수 있는 기능을 봤는데, 한동안? 오랫동안 방치한 백준을 하루에 하나씩이라도 풀어서 올리고 싶다는 생각이 강하게 들었다. 등급은 아직 0이지만.. 꾸준히 풀다보면 올라가지 않을까? 물론 알고리즘 실력은 형편이 없어서 쉬운것부터 차근차근 풀어야한다. 오랜만에 풀어보는거기도 하고..

### MD 미리 확인하기
vs code에서도 미리보기로 확인할 수 있지만 [online markdown](https://dillinger.io/)에서도 확인 가능해 적용전에 테스트용으로 사용하면 좋을듯하다. 해당 레포는 딱히 로컬에 클론할 계획이 없어서..

## 1. 레포지토리 생성하기
- 본인 이름(github 닉네임)으로 레포지토리를 생성한다
- README.md 파일을 꼭 생성해야한다

## 2. 헤더
[capsule-render](https://github.com/kyechan99/capsule-render) 오픈 API를 이용하면 자동으로 링크를 생성해준다.  
`https://capsule-render.vercel.app/api?` 뒤쪽에 작성하면 되는데, html은 잘 모르고 md은 그나마 조금 아니깐, md으로 작성된 예시를 변경해서 작성했다.  
색상 참고는 [HTML Color Codes](https://htmlcolorcodes.com/) 사이트를 이용했고, 마크다운 미리보기로 보면서 설정을했다, 원래는 랜덤 색상을 쓸까했는데 검은색이라, 일단은 좋아하는 보라색으로..
![header](https://capsule-render.vercel.app/api?type=waving&color=A243F2&height=300&section=header&text=welcome&desc=Nuyhc%20Github%20Profile&fontSize=100&animation=scaleIn&descSize=25&descAlign=65&descAlignY=65&fontColor=FFFFFF)

## 3. 뱃지와 이모지
[Simple Icons](https://simpleicons.org/)와 [shields.io](https://shields.io/)를 이용해 뱃지를 설정하면 된다.  
어떤 뱃지들을 넣을까하다, 일단 사용할 수있는 개발툴과 언어들만 넣기로했다.  
이모지는 [emojicopy](https://www.emojicopy.com/) 이용  
처음엔 어떤 방식으로 코드가 구성되는지 몰라서 애를 정말 많이 먹음..

### 뱃지 코드
`<img src="https://img.shields.io/badge/이름-코드?style=flat-square&logo=로고명&logoColor=로고색"/>`
- 이름: 뱃지 내용
- 코드: 색상 hexcode
- 로고명: 아이콘 이름
- 로고색: 로고의 색깔
- 링크를 걸고 싶으면, ` <a href="링크"><img src="뱃지코드"/></a>`
- 링크를 거는 방법은, 그냥 md문법 링크거는 형식으로 작성해도 되는듯
- 로고는 [simple-icon](https://github.com/simple-icons/simple-icons/blob/develop/slugs.md)의 slugs.md를 참고

```
<a href="https://www.instagram.com/nuyhc_/"><img src="https://img.shields.io/badge/Instagram-E4405F?style=plastic&logo=instagram&logoColor=FFFFFF"/></a>
<a href="https://nuyhc.github.io/"><img src="https://img.shields.io/badge/Github%20Pages-222222?style=plastic&logo=githubpages&logoColor=FFFFFF"/></a>
  ```
<a href="https://www.instagram.com/nuyhc_/"><img src="https://img.shields.io/badge/Instagram-E4405F?style=plastic&logo=instagram&logoColor=FFFFFF"/></a>
<a href="https://nuyhc.github.io/"><img src="https://img.shields.io/badge/Github%20Pages-222222?style=plastic&logo=githubpages&logoColor=FFFFFF"/></a>

```
<img src="https://img.shields.io/badge/C-A8B9CC?style=plastic&logo=c&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/C++-00599C?style=plastic&logo=cplusplus&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=FFFFFF&"/>
```

<img src="https://img.shields.io/badge/C-A8B9CC?style=plastic&logo=c&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/C++-00599C?style=plastic&logo=cplusplus&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=plastic&logo=pandas&logoColor=FFFFFF&"/>

<img src="https://img.shields.io/badge/VS Code-007ACC?style=plastic&logo=visualstudiocode&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/Anaconda-44A833?style=plastic&logo=anaconda&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/Git-F05032?style=plastic&logo=git&logoColor=FFFFFF&"/>
<img src="https://img.shields.io/badge/GitHub-181717?style=plastic&logo=github&logoColor=FFFFFF&"/>


## 4. Hits
방문자 수를 보여줄 수 있는 기능  
[HITS](https://hits.seeyoufarm.com/)에서 손쉽게 설정 가능  
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fnuyhc&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23FFFFFF&title=Hits&edge_flat=false)](https://hits.seeyoufarm.com)

## 5. Git Stats
깃에서 어떤 활동을 하고 있는지 보여주는 지표, 아직은 딱히 하는게 없지만 미래를 위해..
[readme-stats](https://github.com/anuraghazra/github-readme-stats)을 참고했다.  
```
![nuyhc github stats](https://github-readme-stats.vercel.app/api?username=nuyhc&show_icons=true&count_private=true&theme=gruvbox)
```

## 6. 백준 티어
사실 이것때문에 꾸미는 이유가 큰거 같다. 아직 0렙이긴한데.. 동기부여 느낌으로..  
[mazzassumnida 프로젝트](https://github.com/mazassumnida/mazassumnida)에 적용법이 잘 나와있다.  

```
[![Solved.ac 프로필](http://mazassumnida.wtf/api/generate_badge?boj={handle})](https://solved.ac/{handle})
```
에서 {handle} 부분에 본인 백준 아이디를 기입하면 됨

## 7. 기타 위젯
다양한 위젯들이 있는데 괜찮아 보이는게 있을때마다 추가해볼 생각이다.
