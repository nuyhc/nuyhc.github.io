---
title: "Openweather API 이용해 날씨 정보 출력하기"

categories:
  - TIL
tags:
  - Openweather
  - API

date: 2022-05-08T09:17:39.496Z
---

# 1. 날씨 정보 받아오기
- API 이해와 사용법

## API key 발급 받기
- [OpenWeather](https://home.openweathermap.org/) 가입
- 발급 받은 API 키를 이용

# 2. API
- Application Programming Interface

## Interface
사람은 컴퓨터와 직접적으로 소통이 불가능하다.  
키보드와 마우스 등의 I/O 장치를 이용해 소통을하게되는데, 이때 이런 I/O 장치가 인터페이스에 해당하게 된다.  

## API
- 프로그램과 프로그램(Client와 Server)을 연결해주는 인터페이스
    - Client: 요청을 보냄
    - Server: 요청에 대한 응답을 함
- 모든 서버에서 html을 긁어오는 것이 아니라, 클라이언트와 서버의 중간에서 서로의 데이터를 잘 교환할 수 있도록 해주는 역할을 함
    - API를 만든다: 사용자가 필요한 기능을 미리 만들고 서버에 올린다
    - API를 사용한다: 누군가가 만든 규약대로 사용한다

## Openweathermap
- 미리 만들어진, 세계의 날씨 정보를 제공하는 API

## API Key
- API를 사용할 때, 본인이 누구인지 식별하게 해주는 역할
- 일반적으로, 특정 사용자만 알아볼수있는 문자열로 나타냄

# 3. API 링크 만들기
- OpenWeather API가 제공하는 API중 Current Weather Data를 이용
- 응답을 원하는 주소로 요청을 보낸다 -> API call


```python
city = "Seoul"
apikey = "발급 받은 API 키"

api = "https://api.openweathermap.org/data/2.5/weather?q={city name}&appid={API key}"
# 중괄호 안은 직접 채워줘야 함
# API 파라미터라고 함

print(api)
```

## f-string
- 사용을 원하는 문자열 앞에 f를 적어줌
- 변수로 변경하기 원하는 이름을 {}로 변경함


```python
city = "Seoul"
apikey = "발급 받은 API 키"

api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"

print(api)
```

# 4. 날씨 받아오기
`https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}`
- ?를 기점으로, 앞쪽은 공통 url, 뒤쪽은 파라미터를 의미
- 설정한 도시의 'API키를 바탕으로 정보를 제공해줘'라는 의미


```python
import requests

city = "Seoul"
apikey = "발급 받은 API 키"
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"

result = requests.get(api)

print(result)
print(result.text)
```

    <Response [200]>
    {"coord":{"lon":126.9778,"lat":37.5683},"weather":[{"id":501,"main":"Rain","description":"moderate rain","icon":"10d"}],"base":"stations","main":{"temp":289.9,"feels_like":288.62,"temp_min":289.81,"temp_max":289.93,"pressure":1018,"humidity":38},"visibility":10000,"wind":{"speed":2.57,"deg":30},"rain":{"1h":2.73},"clouds":{"all":75},"dt":1651999797,"sys":{"type":1,"id":8105,"country":"KR","sunrise":1651955385,"sunset":1652005647},"timezone":32400,"id":1835848,"name":"Seoul","cod":200}
    

`print(result)`의 결과로는 `Response [200]`이 출력됨  
이는 요청이 정상적으로 이뤄지고 응답도 정상적으로 받아왔다는 의미  
  
`print(result.text)`의 결과로는 딕셔너리 형태의 데이터를 받아옴  
이제 해당 데이터에서 필요한 데이터를 이용해 날씨 정보를 출력하는 프로그램을 작성해야 함


```python
import requests

city = "Seoul"
apikey = "발급 받은 API 키"
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"

result = requests.get(api)

print(type(result.text))
```

    <class 'str'>
    

## json
- 파이썬이 제공하는 기본 모듈 중 하나
- Lightweight Data-Interchange Format
- JavaScript Object Notation -> JSON
- 일반적으로, 데이터를 주고 받을 때 사용하는 포맷
- 파이썬의 딕셔너리와 유사한 구조


```python
import requests
import json

city = "Seoul"
apikey = "발급 받은 API 키"
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"

result = requests.get(api)

data = json.loads(result.text)

print(type(result.text))
print(type(data))
```

    <class 'str'>
    <class 'dict'>
    

받아온 데이터를 json형식으로 가독성이 좀 좋게 바꾸면 다음과 같다  
```json
{
"coord":{"lon":126.9778,"lat":37.5683},
"weather":[{"id":800,"main":"Clear","description":"clear sky","icon":"01d"}],
"base":"stations",
"main":{"temp":290.1,"feels_like":288.69,"temp_min":287.81,"temp_max":290.91,"pressure":1020,"humidity":32},
"visibility":10000,
"wind":{"speed":3.09,"deg":170},
"clouds":{"all":0},"dt":1651623938,
"sys":{"type":1,"id":8105,"country":"KR","sunrise":1651610036,"sunset":1651659827},
"timezone":32400,
"id":1835848,
"name":"Seoul",
"cod":200
}
```


```python
import requests
import json

city = "Seoul"
apikey = "발급 받은 API 키"
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}"

result = requests.get(api)

data = json.loads(result.text)

print(data["name"],"의 날씨입니다.")
print("날씨는", data["weather"][0]["description"],"입니다.")
print("현재 온도는", data["main"]["temp"],"입니다.")
print("하지만 체감 온도는", data["main"]["feels_like"],"입니다.")
print("최저 기온은", data["main"]["temp_min"],"입니다.")
print("최고 기온은", data["main"]["temp_max"],"입니다.")
print("습도는", data["main"]["humidity"],"입니다.")
print("기압은", data["main"]["pressure"],"입니다.")
print("풍향은", data["wind"]["deg"],"입니다.")
print("풍속은", data["wind"]["speed"],"입니다.")
```

    Seoul 의 날씨입니다.
    날씨는 moderate rain 입니다.
    현재 온도는 289.9 입니다.
    하지만 체감 온도는 288.62 입니다.
    최저 기온은 289.81 입니다.
    최고 기온은 289.93 입니다.
    습도는 38 입니다.
    기압은 1018 입니다.
    풍향은 30 입니다.
    풍속은 2.57 입니다.
    

# 5. 언어 및 단위 변경하기
[para info](https://openweathermap.org/current) 문서를 참고해 변경함


```python
import requests
import json

city = "Seoul"
apikey = "발급 받은 API 키"
lang = "kr"
units = "metric"
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apikey}&lang={lang}&units={units}"

result = requests.get(api)

data = json.loads(result.text)

print(data["name"],"의 날씨입니다.")
print("날씨는", data["weather"][0]["description"],"입니다.")
print("현재 온도는", data["main"]["temp"],"입니다.")
print("하지만 체감 온도는", data["main"]["feels_like"],"입니다.")
print("최저 기온은", data["main"]["temp_min"],"입니다.")
print("최고 기온은", data["main"]["temp_max"],"입니다.")
print("습도는", data["main"]["humidity"],"입니다.")
print("기압은", data["main"]["pressure"],"입니다.")
print("풍향은", data["wind"]["deg"],"입니다.")
print("풍속은", data["wind"]["speed"],"입니다.")
```

    Seoul 의 날씨입니다.
    날씨는 보통 비 입니다.
    현재 온도는 16.75 입니다.
    하지만 체감 온도는 15.47 입니다.
    최저 기온은 16.66 입니다.
    최고 기온은 16.78 입니다.
    습도는 38 입니다.
    기압은 1018 입니다.
    풍향은 30 입니다.
    풍속은 2.57 입니다.
    

# 6. 정리
- 제공되는 API는 다양함
- 기본적으로 API와 함께 기본적인 사용법을 문서 형식으로 제공하고 있음
- 어떤 프로젝트를 계획하고, 그에 맞는 API를 사용하게되면 기본적인 사용법은 익히고 사용하는것이 좋아 보임
