---
title: "[minimal_mistakes] 구글 서치와 애널리틱스 등록, 다른 검색 엔진들도"

categories:
  - TIL
  - GitHub Page
tags:
  - GitHub
  - Git
date: 2022-05-06T14:18:41.204Z
---

# 1. Google Search Console과 Analytics
- Search Console과 Analytics는 비슷하지만 다른 서비스
 
## Google Search Console
- 구글 검색 엔진에 웹사이트가 검색되도록 등록
- 모니터링 결과 제공
- 크롤링으로 색인 생성을 요청 가능
- 트래픽 데이터(검색 빈도, 표시하는 검색어 등)

## Google Analytics
- 웹 분석 도구
- 웹 사이트로 유입되는 모든 방분자의 정보를 확인 가능
- 방문자의 위치, 네트워크, 기기 등의 정보도 확인 가능

# 2. Search Console 등록하기
[Google Search Console](https://search.google.com/search-console/welcome?utm_source=about-page)  
- html 파일을 커밋하는 방식으로 내 블로그의 소유권을 증명함
- sitemap 등록으로 지속적으로 데이터를 가져오기가 가능해짐

# 3. Google Analytics
[Google Analytics](https://analytics.google.com/analytics/web/provision/#/provision)
- tracking ID를 등록해야 함
```yml
# Analytics
analytics:
  provider: "google-gtag" # false (default), "google", "google-universal", "google-gtag", "custom"
  google:
    tracking_id:
    anonymize_ip: # true, false (default)
```

# 4. Naver
[네이버 서치 어드바이저](https://searchadvisor.naver.com/)
- 구글과 마찬가지로 html 파일을 커밋해 소유권을 확인  
- sitemap.xml도 제출

# 5. 빙, 다음 등
- 굳이 등록할 필요를 느끼지 못해 등록하지 않음

