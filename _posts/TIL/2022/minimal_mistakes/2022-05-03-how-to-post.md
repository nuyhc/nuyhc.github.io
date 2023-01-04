---
title: "[minimal_mistakes] 포스팅하기"

categories:
  - TIL
  - GitHub Page
tags:
  - GitHub
  - Git
---

# 포스팅하기
- 블로그 포스트는 _posts 폴더에 .md 확장자를 사용
- YEAR-MONTH-DAY-title.md 형식의 이름
- 초기에는 _posts 폴더가 없으니 직접 생성해야 함
      
## 포스팅 파일 형식
```
---
layout: 레이아웃 양식
title: 글제목

categories:
    - 카테고리 설정
tags:
    - [테그 설정]
---

마크다운 형식으로 작성된 본문
```

## 포스팅하는 순서
1. 주어진 형식에 맞추어 파일 작성
2. git add 파일이름
3. git commit -m 커밋내용
4. git push로 반영

꼭 git 명령어를 이용할 필요 없이 GUI나 다른 방법을 사용해도 무관한듯 싶다.
