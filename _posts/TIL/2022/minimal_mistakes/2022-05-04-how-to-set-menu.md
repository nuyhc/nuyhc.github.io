---
title: "[minimal_mistakes] 블로그 메뉴 구성하기"

categories:
  - TIL
  - GitHub Page
tags:
  - GitHub
  - Git
date: 2022-05-05T12:20:40.634Z
---

# 블로그 메뉴 구성하기
- 블로그의 메뉴 구성
- 이전에 생성한 page는 주소를 직접쳐야지만 들어갈 수 있다는 문제를 해결

## navigation.yml
- _data 폴더 아래있는 **navigatioin.yml** 파일에서 메뉴를 구성하는 듯

```yml
# main links
main:
  - title: "Quick-Start Guide"
    url: https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/
  # - title: "About"
  #   url: https://mmistakes.github.io/minimal-mistakes/about/
  # - title: "Sample Posts"
  #   url: /year-archive/
  # - title: "Sample Collections"
  #   url: /collection-archive/
  # - title: "Sitemap"
  #   url: /sitemap/
  ```

  2부분만 변경해봤다.  
  예상이 맞다면 해당 부분으로 카테고리도 만들 수 있을꺼 같다.
  1. 페이지를 생성하고
  2. post들을 페이지로 연결
  이런 방식으로 카테고리를 생성하게되는게 아닐까 싶다.

```yml
main:
  - title: "Home"
    url: https://nuyhc.github.io/
  - title: "About"
    url: /about/
```

navigation.yml 파일에는 상단 메뉴바를 구성하는 설정이 들어있다.  
그럼 같은 폴더에 묶여있는 ui-text.yml이 좌측이나 우측 메뉴 구성인가 싶어서 봤더니 전혀 다른 내용이다. 이거는 나중에 필요하면 그때가서..  
  
  의문점이 하나있는데, 카테고리 안에 카테고리를 만드는 방식은 어떻게되는걸까? 카테고리를 만드는 방식이 내가 예상한게 맞으면 인덴트로 구분해서 만들어지는건가?