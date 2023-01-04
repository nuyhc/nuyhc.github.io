---
title: 백준 solved.ac BRONZE 5 문제 풀이
date: 2022-05-19T13:18:22.025Z

categories:
    - Programming
    - Algorithm
tags:
  - baekjoon
---

# 백준 solved.ac BRONZE 5 문제 풀이
백준 solved.ac 티어표상 BRONZE 5 등급 문제중에 풀지 않았던 문제들을 풀었다.  
다국어 지원 문제들도 있긴했지만, 한글로 나온 문제들만 풀었다.  
단순 구현과 출력 문제들이 많아 일부만 모아서 올린다.

업데이트 된 문제들의 풀이는 해당 레포지토리에서 확인 가능  
> [Baekjoon Bronze 5 repo.](https://github.com/nuyhc/Algorithm_Practice)

## 백준 2845 파티가 끝나고 난 뒤



```python
l, p = map(int, input().split())
arr = input()

arr = arr.split()

for _ in arr:
    print(int(_)-l*p, end=" ")
```

    -1 1 900 -100 -3 

## 백준 9653 스타워즈


```python
print("    8888888888  888    88888")
print("   88     88   88 88   88  88")
print("    8888  88  88   88  88888")
print("       88 88 888888888 88   88")
print("88888888  88 88     88 88    888888")
print("")
print("88  88  88   888    88888    888888")
print("88  88  88  88 88   88  88  88")
print("88 8888 88 88   88  88888    8888")
print(" 888  888 888888888 88  88      88")
print("  88  88  88     88 88   88888888")
```

        8888888888  888    88888
       88     88   88 88   88  88
        8888  88  88   88  88888
           88 88 888888888 88   88
    88888888  88 88     88 88    888888
    
    88  88  88   888    88888    888888
    88  88  88  88 88   88  88  88
    88 8888 88 88   88  88888    8888
     888  888 888888888 88  88      88
      88  88  88     88 88   88888888
    

## 백준 3003 킹, 퀸, 룩, 비숍, 나이트, 폰


```python
piece = {"king":1, "queen":1, "rook":2, "bishop":2, "knight":2, "pawn": 8}

k, q, r, b, kn, p = map(int, input().split())

print(piece["king"]-k, piece["queen"]-q, piece["rook"]-r, piece["bishop"]-b, piece["knight"]-kn, piece["pawn"]-p)
```

    -1 0 0 1 0 7
    

## 백준 5339 콜센터


```python
print("     /~\\")
print("    ( oo|")
print("    _\=/_")
print("   /  _  \\")
print("  //|/.\|\\\\")
print(" ||  \ /  ||")
print("============")
print("|          |")
print("|          |")
print("|          |")
```

         /~\
        ( oo|
        _\=/_
       /  _  \
      //|/.\|\\
     ||  \ /  ||
    ============
    |          |
    |          |
    |          |
    

## 백준 1271 엄청난 부자2


```python
n, m = map(int, input().split())
print(n//m)
print(n%m)
```

    10
    0
    

## 백준 5554 심부름 가는 길


```python
hTs = int(input()) # 집 -> 학교
sTp = int(input()) # 학교 -> pc방
pTa = int(input()) # pc방 -> 학원
aTh = int(input()) # 학원 -> 집

total = hTs + sTp + pTa + aTh

print(total//60)
print(total%60)
```

    44
    2
    

## 백준 5522 카드 게임


```python
total = 0

for _ in range(5):
    total += int(input())

print(total)
```

    210
    

## 백준 9654 나부 함대 데이터


```python
print("SHIP NAME      CLASS          DEPLOYMENT IN SERVICE")
print("N2 Bomber      Heavy Fighter  Limited    21        ")
print("J-Type 327     Light Combat   Unlimited  1         ")
print("NX Cruiser     Medium Fighter Limited    18        ")
print("N1 Starfighter Medium Fighter Unlimited  25        ")
print("Royal Cruiser  Light Combat   Limited    4         ")
```

    SHIP NAME      CLASS          DEPLOYMENT IN SERVICE
    N2 Bomber      Heavy Fighter  Limited    21        
    J-Type 327     Light Combat   Unlimited  1         
    NX Cruiser     Medium Fighter Limited    18        
    N1 Starfighter Medium Fighter Unlimited  25        
    Royal Cruiser  Light Combat   Limited    4         
    

## 백준 10170 NFC West vs North


```python
print("NFC West       W   L  T")
print("-----------------------")
print("Seattle        13  3  0")
print("San Francisco  12  4  0")
print("Arizona        10  6  0")
print("St. Louis      7   9  0")
print("")
print("NFC North      W   L  T")
print("-----------------------")
print("Green Bay      8   7  1")
print("Chicago        8   8  0")
print("Detroit        7   9  0")
print("Minnesota      5  10  1")
```

    NFC West       W   L  T
    -----------------------
    Seattle        13  3  0
    San Francisco  12  4  0
    Arizona        10  6  0
    St. Louis      7   9  0
    
    NFC North      W   L  T
    -----------------------
    Green Bay      8   7  1
    Chicago        8   8  0
    Detroit        7   9  0
    Minnesota      5  10  1
    

## 백준 2338 긴자리 계산


```python
a = int(input())
b = int(input())
print(a+b)
print(a-b)
print(a*b)
```

    0
    2
    -1
    

## 백준 14645 와이버스 부릉부릉


```python
n, k = map(int, input().split())
print("비와이")
```

    비와이
    

## 백준 16394 홍익대학교


```python
n = int(input())
print(n-1946)
```

    72
    

## 백준 15894 수학은 체육과목 입니다


```python
n = int(input())

print(n*4)
```

    20
    

# 백준 16430 제리와 톰


```python
a, b = map(int, input().split())
print(b-a, b)
```

    5 7
    

# 백준 15964 이상한 기호


```python
a, b = map(int, input().split())
print((a+b)*(a-b))
```

    7
    

# 백준 15727 조별과제를 하려는데 조장이 사라졌다


```python
n = int(input())
if n%5==0:
    print(n//5)
else:
    print(int(n/5)+1)
```

    3
    

# 백준 14652 나는 행복합니다~


```python
N, M, K = map(int, input().split())
n = K // M
m = K % M
print(n, m)
```

    1 2
    

# 백준 20492 세금


```python
n = int(input())
print(int(n*0.78), int(n-n*0.2*0.22))
```

    780 956
    

# 백준 17256 달달함이 넘쳐흘러


```python
ax, ay, az = map(int, input().split())
cx, cy, cz = map(int, input().split())

bx = int(cx-az)
by = int(cy/ay)
bz = int(cz-ax)

print(bx, by, bz)

```

    7 1 7
    

# 백준 17496 스타후르츠


```python
N, T, C, P = map(int, input().split())
print((N-1)//T*C*P)
```

    1500000
    

# 백준 14928 큰 수 (BIG)


```python
n = int(input())
print(n%20000303)
```

    1313652
    

# 백준 11283 한글2


```python
print(ord(input())-44031)
```

    1
    

# 백준 24262 알고리즘 수업 - 알고리즘의 수행 시간 1


```python
n = int(input())
print(1)
print(0)
```

    1
    0
    
