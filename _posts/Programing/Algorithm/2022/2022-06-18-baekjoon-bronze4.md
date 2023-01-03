---
title: 백준 solved.ac BRONZE 4 문제 풀이
date: 2022-06-18T14:01:49.662Z

categories:
    - Programming
    - Algorithm
tags:
  - baekjoon
---

# 백준 solved.ac BRONZE 4 문제 풀이
백준 solved.ac 티어표상 BRONZE 4 등급 문제중에 풀지 않았던 문제들을 풀었다.  
다국어 지원 문제들도 있긴했지만, 한글로 나온 문제들만 풀었다.  
단순 구현과 출력 문제들이 많아 일부만 모아서 올린다.

업데이트 된 문제들의 풀이는 해당 레포지토리에서 확인 가능  
> [Baekjoon Bronze 4 repo.](https://github.com/nuyhc/Algorithm_Practice)

# 백준 2752 세수 정렬


```python
a, b, c = map(int, input().split())

l = [a, b, c]

for _ in sorted(l):
    print(_, end=" ")
```

    1 2 3 

# 백준 10797 10부제


```python
n = int(input())
l = map(int, input().split())
cnt = 0
for _ in l:
    if _==n: cnt += 1
    
print(cnt)
```

    2
    

# 백준 2530 인공지능 시계


```python
H, M, S = map(int, input().split())
D = int(input()) 

S += D % 60
D = D // 60
if S >= 60:
    S -= 60
    M += 1

M += D % 60
D = D // 60
if M >= 60:
    M -= 60
    H += 1

H += D % 24
if H >= 24:
    H -= 24

print(H,M,S)
```

    14 33 20
    

# 백준 10156 과자


```python
K, N, M = map(int, input().split())

s = K*N - M
if s<0: s=0

print(s)
```

    0
    

# 백준 15680 연세대학교


```python
n = int(input())

if n==0:
    print("YONSEI")
elif n==1:
    print("Leading the Way to the Future")
```

# 백준 1297 TV 크기


```python
d, h, w = map(int, input().split())
r = d/((h**2+w**2)**0.5)
print(int(h*r), int(w*r))
```

    25 45
    

# 백준 11943 파일 옮기기


```python
apple_A, orange_B = map(int, input().split())
apple_C, orange_D = map(int, input().split())

print(orange_B+apple_C if apple_A+orange_D>=orange_B+apple_C else apple_A+orange_D)
```

    5
    

# 백준 15963 CASIO


```python
n, m = map(int, input().split())

if n==m: print(1)
else: print(0)
```

    0
    

# 백준 17362 수학은 체육 과목입니다 2


```python
n = int(input())

ans = n%8

if ans==1: print(1)
elif ans==2 or ans==0: print(2)
elif ans==3 or ans==7: print(3)
elif ans==4 or ans==6: print(4)
else: print(5)
```

    2
    

# 백준 17388 와글와글 숭고한


```python
s, k, h = map(int, input().split())

if s+k+h>=100:
    print("OK")
else:
    l = [s, k, h]
    idx = l.index(min(l))
    if idx==0: print("Soongsil")
    elif idx==1: print("Korea")
    else: print("Hanyang")
```

    Korea
    

# 백준 10039 평균 점수


```python
score = []
for _ in range(5):
    score.append(int(input()))

total = 0
for _ in score:
    if _<40: total += 40
    else: total += _

print(int(total/5))
```

    68
    

# 백준 5543 상근날드


```python
menu = []
for _ in range(5):
    menu.append(int(input()))

h = min(menu[:3])
d = min(menu[3:])

print(h+d-50)
```

    848
    

# 백준 2420 사파리월드


```python
n, m = map(int, input().split())
print(abs(n-m))
```

    7
    

# 백준 10162 전자레인지


```python
t=int(input())
a = b = c = 0
if t%10 != 0:
    print(-1)
else:
    a = t // 300
    b = (t%300)//60
    c = ((t%300)%60)//10
    print(a, b, c)
```

    -1
    

# 백준 5532 방학 숙제


```python
import math

arr = []
for _ in range(5):
    arr.append(int(input()))
    
if math.ceil(arr[1]/arr[3]) > math.ceil(arr[2]/arr[4]): print(arr[0] - math.ceil(arr[1]/arr[3]))
else: print(arr[0]-math.ceil(arr[2]/arr[4]))
```

    15
    

# 백준 5596 시험 점수


```python
a = sum(map(int,input().split()))
b = sum(map(int,input().split()))
print(max(a,b))
```

    320
    

# 백준 10707 수도요금


```python
a = int(input())
b = int(input())
c = int(input())
d = int(input())
p = int(input())

vx = a*p

if p>c :
  vy = b+((p-c)*d)
else :
  vy = b

print(min(vx, vy))
```

    90
    

# 백준 10768 특별한 날


```python
m = int(input())*100
d = int(input())

if m+d>218:
    print("After")
elif m+d==218:
    print("Special")
else:
    print("Before")
```

    Special
    

# 백준 5575 타임 카드


```python
for i in range(3):
    fh, fm, fs, lh, lm, ls = map(int, input().split())
    first = (fm * 60) + (fh * 3600) + fs
    last = (lm * 60) + (lh * 3600) + ls
    time = last - first
    h = time // 3600
    m = (time % 3600) // 60
    s = (time % 3600) % 60
    print(f"{h} {m} {s}")
```

    9 0 0
    8 59 59
    0 0 38
    

# 백준 11948 과목선택


```python
a = []

for _ in range(6):
    a.append(int(input()))
    
f, s = sorted(a[:4]), sorted(a[4:])

print(sum(f[1:])+s[-1])
```

    140
    

# 백준 16486 운동장 한 바퀴


```python
d1 = int(input())
d2 = int(input())
pi = 3.141592

print(2*d1+2*pi*d2)
```

    76.265472
    

# 백준 4299 AFC 윔블던


```python
a, b=map(int,input().split())
if a < b:
    print(-1)
else:
    x=(a+b)//2
    y=(a-b)//2
    if x+y==a and x-y==b:
        print(x, y)
    else:
        print(-1)
```

    2 1
    

# 백준 15873 공백없는 A+B


```python
a = input()
if len(a)==2:
    print(int(a[0])+int(a[1]))
else:
    idx = a.find("0")
    if idx==1:
        print(int(a[:idx+1])+int(a[idx+1:]))
    else:
        print(int(a[0])+int(a[1:]))
```

    20
    

# 백준 13866 팀 나누기


```python
a, b, c, d = map(int, (input().split()))

l = [a, b, c, d]

l = sorted(l)

print(abs((l[0]+l[3])-(l[1]+l[2])))
```

    7
    

# 백준 20499 Darius님 한타 안 함?


```python
kda = input()

kda = kda.split("/")

if (int(kda[0])+int(kda[2])<int(kda[1])) or int(kda[1])==0:
    print("hasu")
else:
    print("gosu")
```

    gosu
    

# 백준 14924 폰 노이만과 파리


```python
s, t, d = map(int, input().split())

print(int(t*(d/(s*2))))
```

    150.0
    

# 백준 11365 !밀비 급일


```python
while True:
    a = input()
    if a=="END": break
    print(a[::-1])
```

    What a good code!
    

# 백준 1264 모음의 개수


```python
vow = ["a", "e", "i", "o", "u"]
while True:
    s = input().lower()
    if s=="#": break
    cnt = 0
    for _ in s:
        if _ in vow:
            cnt += 1
    print(cnt)
    
```

    7
    14
    

# 백준 11945 뜨거운 붕어빵


```python
n, m = map(int, input().split())

for _ in range(n) :
  data = input()
  print(data[::-1])
```

    0000100
    0101110
    1111111
    0101110
    0000100
    

# 백준 5893 17배


```python
n = int(input(), 2)
n *= 17
print(bin(n)[2:])
```

# 백준 19944 뉴비의 기준은 뭘까?


```python
n, m = map(int, input().split())

if m==1 or m==2:
    print("NEWBIE!")
elif n>=m:
    print("OLDBIE!")
else:
    print("TLE!")
```

    OLDBIE!
    

# 백준 15726 이칙연산


```python
a, b, c = map(int, input().split())
x = (a * b) / c
y = (a / b) * c
if x > y:
    print(int(x))
else:
    print(int(y))
```

    64
    

# 백준 10808 알파벳 개수


```python
s = input()
lst = [0]*26
for i in s:
    lst[ord(i)-97]+=1
for i in lst:
    print(i,end= ' ')
```

# 백준 14581 팬들에게 둘러싸인 홍준


```python
n = input()

print(f":fan::fan::fan:\n:fan::{n}::fan:\n:fan::fan::fan:")
```

    :fan::fan::fan:
    :fan::h0ngjun7::fan:
    :fan::fan::fan:
    

# 백준 14489 치킨 두 마리 (..)


```python
a, b = map(int, input().split())
p = int(input())

if a+b<p*2:
    print(a+b)
else:
    print((a+b)-(p*2))
```

    11000
    

# 백준 15700 타일 채우기 4


```python
n, m = map(int, input().split())
print(n*m//2)
```

    4
    

# 백준 14935 FA


```python
print("FA")
```

    FA
    

# 백준 16204 카트 뽑기


```python
n, m, k = map(int, input().split())

print(min(m,k)+min(n-m,n-k))
```

    3
    

# 백준 13136 Do Not Touch Anything


```python
r, c, n = map(int, input().split())

if r%n:
    x = r//n + 1
else:
    x = r//n
if c%n:
    y = c//n + 1
else:
    y = c//n

print(x*y)
```

    9
    

# 백준 16199 나이 계산하기


```python
y1, m1, d1 = map(int, input().split())
y2, m2, d2 = map(int, input().split())
age_man = 0

if m1 < m2:
    age_man = y2-y1
elif m1 == m2:
    if d1 <= d2:
        age_man = y2-y1
    else:
        age_man = y2-y1-1
else:
    age_man = y2-y1-1
age_count = y2-y1+1
age_year = y2-y1

print(age_man)
print(age_count)
print(age_year)
```

    0
    1
    0
    

# 백준 19698 헛간 청약


```python
n, w, h, l = map(int, input().split(" "))

r = (w//l)*(h//l)
if r<n:
    print(r)
else:
    print(n)
```

    6
    

# 백준 14623 감정이입


```python
B1 = input()
B2 = input()
print((bin(int(B1, 2) * int(B2, 2)))[2:])
```

    11110
    

# 백준 11282 한글
 유니코드에서 한글이 44031부터 시작한다는 점을 이용


```python
print(chr(44031 + int(input())))
```

    백
    

# 백준 14264 정육각형과 삼각형


```python
l = int(input())
print(3**0.5/4*l**2)
```

    10.825317547305483
    

# 백준 15921 수찬은 마린보이야!!
평균/기댓값은 항상 1이기에, 답은 항상 1이다.


```python
n = int(input())
if n == 0:
    print("divide by zero")
else:
    ln = list(map(int, input().split()))
    ans = sum(ln)/n / (sum(ln)/n)
    print("%.2f" %ans)
```

    1.00
    

# 백준 23825 SASA 모형을 만들어보자


```python
s, a = map(int, input().split())

print(min(s//2, a//2))
```

    2
    

# 백준 25191 치킨댄스를 추는 곰곰이를 본 임스


```python
n = int(input())
a, b = map(int, input().split())

if n<=(a//2+b):
    print(n)
else:
    print(a//2+b)
```

    3
    

# 백준 24883 자동완성


```python
s = input()

if (s=='N') or (s=='n'):
    print("Naver D2")
else:
    print("Naver Whale")
```

    Naver Whale
    

# 백준 24723 녹색거탑


```python
n = int(input())

print(2**n)

```

    8
    

# 백준 17356 욱 제



```python
a, b = map(int, input().split())

m = (b-a)/400
print(1/(1+10**m))
```

    0.7597469266479578
    
