---
title: "[minimal_mistakes] 카테고리와 테그 목록 만들기"

categories:
  - TIL
  - GitHub Page
tags:
  - GitHub
  - Git
date: 2022-05-05T12:21:18.340Z
---

# 카테고리와 테그 목록 만들기
과연 지난번에 가졌던 가설과 일치하는지 확인할 기회가 바로 찾아왔다.

## 1. 포스트 글에서 설정해야되는 것
```yml
---
layout: 
title: ""

categories:
    - 
tags:
    - []
---
```
기존에는 카테고리를 만드는 방법을 몰라, 추후에 만들면 바로 적용할 수 있게 위에처럼 글을 써왔다.  
tags의 경우 대괄호([])안에 넣어야만 하는지 알았는데, 아닌거 같다.  
대괄호 안에 넣어서 작성할 수도 있고, 분리하여 작성해도 문제가 없는거 같다.  
```yml
tags:
    - 
    -
    -
```

### 카테고리와 테그
- 카테고리
    - 게시물을 제목이나 유형으로 분류
- 테그
    - 게시물의 세부 정보를 키워드로 설명하는 것
    - 해시테그와 유사
    - 여러개 추가 가능

카테고리와 테그를 링크로 페이지와 연결해줘야 한다는데, 지난번에 세운 가설이 맞는거 같다.  

## 2. 카테고리와 테그 표시하지 않기
_config.yml 파일에서,  
```yml
# Archives
#  Type
#  - GitHub Pages compatible archive pages built with Liquid ~> type: liquid (default)
#  - Jekyll Archives plugin archive pages ~> type: jekyll-archives
#  Path (examples)
#  - Archive page should exist at path when using Liquid method or you can
#    expect broken links (especially with breadcrumbs enabled)
#  - <base_path>/tags/my-awesome-tag/index.html ~> path: /tags/
#  - <base_path>/categories/my-awesome-category/index.html ~> path: /categories/
#  - <base_path>/my-awesome-category/index.html ~> path: /
category_archive:
  type: liquid
  path: /categories/
tag_archive:
  type: liquid
  path: /tags/
```
category_archive와 tag_archive 설정 부분을 주석처리하면 더 이상 블로그에서 표시하지 않는다고 한다.

## 3. 카테고리와 테그 페이지 등록하기
이전처럼 각각을 위한 페이지를 생성했다.  

- category-archive.md
```
---
title: "Category"
layout: categories
permalink: /categories/
author_profile: true
---
```

- tag-archive.md
```
---
title: "Tags"
laygout: tags
permalink: /tags/
author_profile: true
---
```

이후 카테고리와 테그도 메뉴바에 추가해줬다.

## 3. 하위 카테고리
하위 카테고리는 주소 설정으로 완성되는거 같다.
```
---
title: "TIL"
permalink: /categories/TIL/
layout: category
author_profile: true
taxonomy: TIL
---

Today I Learned!
```
식으로 일단 TIL 카테고리 페이지를 생성했다.  
해당 페이지에서 보여질 카데로리를 `taxonomy`로 설정하는거 같은데, 그동안 게시물에 작성했던
```  
categories:
    -
```
 
게시물에 기재된 카테고리와 테그를 클릭하면 모여있는 페이지로 정상적으로 이동했다.

~~내가 원했던건 상단 Category 메뉴를 클릭하면 이동되는 방식인데 이 부분은 아직 어떻게하는건지 잘모르겠다.~~

철자를 잘못 입력한게 있어서 고생을했다.. 의도한 그대로 반영된듯

## 4. 추가
```yml
jekyll-archives:
  enabled:
    - categories
    - tags
  layouts:
    category: archive-taxonomy
    tag: archive-taxonomy
  permalinks:
    category: /categories/:name/
    tag: /tags/:name/
```
위와 같이 코드를 수정하면 따로 페이지를 만들지 않아도 자동으로 카테고리와 테그가 추가되는거 같다.  
근데 카테고리는 따로 페이지가 있는게 설명하기도 편한거 같아서 계속해서 페이지를 만드는 방식을 사용할꺼 같다.