---
title: "[minimal_mistakes] TOC 설정하기"

categories:
    - GitHub Page
tags:
    - GitHub
    - Git

date: 2022-05-08T07:44:54.264Z
---

# 1. TOC: Table of Contents
- 마크다운 문법에서 헤더(H1~H6) 목록을 표시해주는 기능
- minimal-mistakes 테마에서는 오른쪽 사이드 바에 표시
```yml
toc: ture
toc_sticky: true
toc_label: " "
```
- toc를 설정하기 위해서는 헤더 부분에 위 코드를 추가하면 됨
- toc_sticky는 사이드바에 고정하는 역할, 페이지가 화면에서 넘어가도 toc가 같이 넘어가지 않고 고정해줌  
- toc_label은 TOC의 제목을 직접 설정할 수 있게 해줌
