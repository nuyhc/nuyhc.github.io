---
title: Markdown Mermaid
date: 2022-05-22T13:07:42.040Z

categories:
  - TIL
tags:
  - md
  - vs code
---
# Markdown Mermaid
vs code 확장 기능 구경하다가 **Markdown Preview Mermaid Support**라는 플러그인을 봤다.  
mermaid는 markdown으로 UML을 그릴 수 있게한 언어인데, 기존에 도식을 사용할 일이 있으면 아이패드로 그려서 이미지로 사용했다면,  
이제는 간단하게 마크다운 문법으로 작성할 수 있을꺼 같다.  

기본적인 작성법은 수도코드를 이용하는데, 여러번 써보다보면 익숙해지지 않을까 싶다.  
일단 [기본 문서](https://mermaid-js.github.io/mermaid/#/)를 보고 필요한것만 정리했다. 좀 복잡한 다이어그램을 그려야할 경우가 있으면 참고하면 좋을꺼 같다.

## 지원하는 다이어그램
- 플로우차트 (flowchart / graph)
- 시퀀스 다이어그램 (sequenceDiagram)
- 간트 차트 (gantt)
- 클래스 다이어그램 (classDiagram)

공식적으로는 위의 차트와 다이어그램을 지원하고, 비공식적으로 지원하는것으로는 Git Graph(gitGraph)와 ERD(erDiagram), User Journey Diagram(journey)가 있는거 같다.

## 기본 구성
```
<div class="mermaid">
차트종류 방향;
    차트내용;
    차트내용;
</div>
```
기본적인 구성은 위와 같은거 같다.

### 방향
굉장히 직관적이다.  
- TB(TD): 위에서 아래로
- BT: 아래에서 위로
- RL: 오른쪽에서 왼쪽으로
- LR: 왼쪽에서 오른쪽으로

### 노드
```md
<div class="mermaid">
graph BT;
id[사각형] --> A[(DB)];
A --> B{조건};
</div>
```

<div class="mermaid">
graph BT;
  id[사각형] --> A[(DB)];
  A --> B{조건};
</div>

### 엣지
화살표를 입력하는 형식에 따라 형식이 달라짐
```md
<div class="mermaid">
flowchart LR;
A --> B;
B -.-> C;
C --- D;
D -->|옵션도 가능|A;
E --> A;
E --> D;
</div>
```

<div class="mermaid">
flowchart LR;
A --> B;
B -.-> C;
C --- D;
D -->|옵션도 가능|A;
E --> A;
E --> D;
</div>


기본적인 문법만 알면, 프리뷰를 보면서 그리면 쉽게 그릴수 있을꺼 같다.  
에디터에서는 형식이 출력되지만, 깃 블로그에서는 지원을하지 않는거 같다..

## 깃허브 페이지에서 mermaid 표시하기
어제 포스팅하고 html형식으로 감싸거나 일반적인 방식으로 사용해도 표시가되지 않아 사용이 안되는지 알았다..  
오늘 좀 찾아보니 html 형식을 jekyll에 추가해주면 사용 가능한거 같아 사용해보니 가능했다.  
```html
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js">

</script>
<script>mermaid.initialize({startOnLoad:true});
    
</script>
```
`custom` 파일에 등록해서 사용했다.