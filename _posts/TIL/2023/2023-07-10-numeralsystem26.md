---
title: "기수법 적용"
date: 2023-07-10

categories: TIL
tags:
---

# 기수법 응용
엑셀의 열에서, Z 다음은 AA와 같은 형식으로 값이 증가한다.  
알파뱃은 26개이므로 26진법을 응용하면, 문자열 -> 정수, 정수 -> 문자열 변환이 가능한데,  
알파뱃뿐만 아니라 다른 규칙에 대해서도 유사하게 적용 가능할꺼 같다.

```python
class Convert:
    @staticmethod
    def I2L(col_idx: int)->str:
        letter = ""
        while col_idx>0:
            col_idx, remainder = divmod(col_idx-1, 26)
            letter = chr(remainder+ord("A"))+letter
        return letter
    @staticmethod
    def L2I(col_letter: str)->int:
        idx = 0
        for letter in col_letter:
            idx = (idx*26)+1+ord(letter)-ord("A")
        return idx
```
