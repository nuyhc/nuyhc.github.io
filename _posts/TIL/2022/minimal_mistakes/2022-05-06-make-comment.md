---
title: "[minimal_mistakes] 댓글 기능 추가하기"

categories:
  - TIL
  - GitHub Page
tags:
  - GitHub
  - Git
date: 2022-05-06T10:01:07.107Z
---

깃허브 페이지는 댓글 기능을 지원하지 않는다.  
블로그에 댓글 기능을 지원하는 서비스인 **Disqus**를 이용하면 블로그에 댓글 기능을 사용할 수 있다고 한다.

# 1. Disqus 가입
- [Disqus](https://disqus.com/)는 소셜댓글 서비스 업체 중에 선두로 꼽히는 업체
- 계정은 구글 계정 연동으로 간편하게 생성했다  
- 블로그(내 사이트)에 사용할 예정이니, I want to install Disqus on my site를 이용  
- 웹 사이트 정보를 기입하고 설정을 함  
- basic으로 사용해도 크게 지장이 가지 않을꺼 같음
- 플랫폼은 지킬로 선택함 

# 2. Disqus 정보 기입
- Disqus-shortname을 확인해야함(설정에서 General에 가면 확인 가능) 
```yml
comments:
  provider               : # false (default), "disqus", "discourse", "facebook", "staticman", "staticman_v2", "utterances", "giscus", "custom"
  disqus:
    shortname            : # https://help.disqus.com/customer/portal/articles/466208-what-s-a-shortname-
  discourse:
    server               : # https://meta.discourse.org/t/embedding-discourse-comments-via-javascript/31963 , e.g.: meta.discourse.org
```
- provider에 disqus로 입력  
- shortname은 shorcut을 입력

# 3. 게시글에 댓글 기능 넣기
```yml
# Defaults
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
```
포스트 부분의 설정도 변경해줘야한다.  
해당 부분을 설정하면 comments 헤더를 작성하지 않아도 되는건지 확인해봐야겠다.

