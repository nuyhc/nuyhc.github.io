---
title: "[HTML/CSS] 단어/글자/줄 간격 조정하기"
date: 2023-02-28T09:05:44.128Z

categories: TIL
tags:
  - HTML
  - CSS
---

# [HTML/CSS] 단어/글자/줄 간격 및 크기 조정하기
`win32com`으로 Outlook 제어시, 가독성있는 본문 작성을 위해 `style` 태그에 추가 지정해서 사용하면 좋을꺼 같아서 정리함.

### 크기 단위
| 단위 | 설명 |  
| :--- | :--- |  
| `em` | 지정 폰트의 대문자 M의 너비 기준 |  
| `rem` | HTML 문서의 root 요소인 `<html>`에 지정된 크기를 기준으로 상대적인 값을 가짐 |  
| `ex` | 지정 폰트의 소문자 x의 높이 기준 |  
| `px` | 픽셀, 장치에 따라 상대적인 크기 |  
| `%` | 기본글꼴의 크기에 대해 상대적인 값 |  
| `pt` | point, 일반 문서에서 가장 많이 사용하는 단위 |  

### 자간 설정하기 (글자 간격)
```html
<style>
  letter-spacing : ~~~단위;
</style>
```

### 어간 설정하기 (단어 사이 간격)
```html
<style>
  word-spacing : ~~~단위;
</style>
```

### 행간 설정하기 (행 사이의 간격)
```html
<style>
  line-height : ~~~단위;
</style>
```
