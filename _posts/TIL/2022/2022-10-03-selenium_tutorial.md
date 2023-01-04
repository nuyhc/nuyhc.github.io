---
title: Selenium 기초 사용
date: 2022-10-02T15:39:41.141Z

categories:
  - TIL
tags:
  - Selenium
---

# Python Selenium 사용법
[참고 | Gorio Learning Y Tech Blog](https://greeksharifa.github.io/references/2020/10/30/python-selenium-usage/)
### Chrome WebDriver
- 브라우저 버전에 맞는 웹 드라이버 다운 받아서 파일과 같은 경로에 두기 (다른 경로에 두면 경로 따로 설정해야 함)


### Import


```python
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains

from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
```

## Driver & Web Load



```python
URL = "https://securities.miraeasset.com/hki/hki3028/r01.do"

driver = webdriver.Chrome(executable_path="./chromedriver") # 드라이버 로드, 확장자는 필요 없음 (상대 경로나 절대 경로 이용 가능)
driver.get(URL)
```


```python
# 현재 url 얻기
driver.current_url
```




    'https://securities.miraeasset.com/hki/hki3028/r01.do'




```python
# 브라우저 닫기
driver.close()
```

## 로딩 대기
브라우저에서 해당 웹 페이지의 요소들을 로드하는데 시간이 걸림  
-> 요소가 존재하지 않는다는 에러를 보지 않기 위해서는 전부 로드 될때까지 대기해야 함
### Implicit Waits (암묵적 대기)
지정한 시간만큼 대기


```python
driver.implicitly_wait(time_to_wait=5) # 초 단위, default=0
```

### Explicit Waits (명시적 대기)
`time.sleep` 함수를 이용해 무조건 몇 초간 대기하는 방법  
-> 비효율적


```python
driver = webdriver.Chrome("./chromedriver")
driver.get(url="https://www.google.com/")
try:
    element = WebDriverWait(driver=driver, timeout=5).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "gLFyf"))
    )
finally:
    driver.quit()
```

`class`가 `gLFyf`인 어떤 요소를 찾을 수 있는지를 5초 동안 시도  
`EC` 객체는 해당 요소를 찾았다면 `True`를, 찾지 못했다면 `False`를 반환  
-> 다양한 조건들이 있음


```python
from selenium.webdriver.support.expected_conditions import *
```


```python
title_is
title_contains
presence_of_element_located
visibility_of
...
```

커스텀 조건 설정도 가능  
-> `__init__` 함수와 `__call__` 함수를 구현한 `class`를 작성해 사용  


- `until(method, message="")`는 `method`의 반환값이 `False`인 동안 실행
- 반대는, `until_not(method, message="")`

## 요소 찾기 (Locating Elements)
다양한 요소를 찾는 방법을 지원함 -> HTML 이용  
찾을 요소를 `ctrl + shift + c`를 눌러 확인할 수 있음


```python
URL = "https://www.google.com/"

driver = webdriver.Chrome(executable_path="./chromedriver")
driver.get(URL)
```

구글 검색창
```html
<input class="gLFyf gsfi" jsaction="paste:puy29d;" maxlength="2048" name="q" type="text" aria-autocomplete="both" aria-haspopup="false" autocapitalize="off" autocomplete="off" autocorrect="off" autofocus="" role="combobox" spellcheck="false" title="검색" value="" aria-label="검색" data-ved="0ahUKEwi1tYCC_L76AhUyBKYKHYVPA3YQ39UDCAY">
```

각 요소에는, `class, XPath, id` 등 여러 속성이 존재  
-> 특징적인 속성을 찾아 이용함


```python
search_box = driver.find_element_by_class_name("gLFyf")
# 키보드를 입력해주는 코드
search_box.send_keys("gorio")
```

`find_element`로 시작하는 함수는 조건에 맞는 요소를 **하나만** 반환하고,  
`find_elements`로 시작하는 함수는 해당 조건을 만족하는 모든 요소를 반복 가능한 (iterable) 형태로 반환

#### find_element_by_xpath
강력한 찾기 기능을 제공  
-> 웹페이지 상에서 해당 요소의 전체 경로나 상대 경로를 갖고 찾기 기능을 수행

xpath = `/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input`


```python
search_box = driver.find_element_by_xpath("/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")
search_box.send_keys("Test Text")
```


```python
search_box = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")
search_box.send_keys("Same Method")
```

| 표현식 | 설명 |
| :--- | :--- |
| nodename | `nodename`을 name으로 갖는 모든 요소 선택 |
| / | root 요소 선택 |
| // | 현재 요소의 자손 요소 선택 |
| . | 현재 요소를 선택 |
| .. | 현재 요소의 부모 요소를 선택 |
| @ | 속성 선택 |
| * | 모든 요소에 매치 |
| @* | 모든 속성 요소에 매치 |
| node() | 모든 종류의 모든 요소에 매치 |
| \| | OR |

## 텍스트 입력
`send_keys(*value)` 함수는 문자열을 그대로 받고, 엔터와 같은 특수 키는 문자열로도 처리 가능하지만 `RETURN = "\ue006"` 다음 같이 사용 가능


```python
search_box = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")
search_box.send_keys("xpath", "\ue006")
```


```python
search_box = driver.find_element(By.XPATH, "/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")
search_box.send_keys("xpath", Keys.RETURN)
```


```python
# 사용 가능 특수키 목록
dir(Keys)
```




    ['ADD',
     'ALT',
     'ARROW_DOWN',
     'ARROW_LEFT',
     'ARROW_RIGHT',
     'ARROW_UP',
     'BACKSPACE',
     'BACK_SPACE',
     'CANCEL',
     'CLEAR',
     'COMMAND',
     'CONTROL',
     'DECIMAL',
     'DELETE',
     'DIVIDE',
     'DOWN',
     'END',
     'ENTER',
     'EQUALS',
     'ESCAPE',
     'F1',
     'F10',
     'F11',
     'F12',
     'F2',
     'F3',
     'F4',
     'F5',
     'F6',
     'F7',
     'F8',
     'F9',
     'HELP',
     'HOME',
     'INSERT',
     'LEFT',
     'LEFT_ALT',
     'LEFT_CONTROL',
     'LEFT_SHIFT',
     'META',
     'MULTIPLY',
     'NULL',
     'NUMPAD0',
     'NUMPAD1',
     'NUMPAD2',
     'NUMPAD3',
     'NUMPAD4',
     'NUMPAD5',
     'NUMPAD6',
     'NUMPAD7',
     'NUMPAD8',
     'NUMPAD9',
     'PAGE_DOWN',
     'PAGE_UP',
     'PAUSE',
     'RETURN',
     'RIGHT',
     'SEMICOLON',
     'SEPARATOR',
     'SHIFT',
     'SPACE',
     'SUBTRACT',
     'TAB',
     'UP',
     '__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__']



### 텍스트 입력 지우기
특수키도 사용 가능


```python
search_box.clear()
```

### 파일 업로드
파일을 받는 요소를 찾고 사용하면 됨


```python
file_path = "../AISCHOOL7/s_code/daily_til.md"
upload = driver.find_element_by_tag_name("input")
upload.send_keys(file_path)
```

## 상호 작용
### 클릭하기
요소를 찾고 `click()` 함수 호출  
`//*[@id="rso"]/div[1]/div[2]/div/div[1]/div/a/h3` <- 검색 결과를 클릭하는 부분의 xpath


```python
posting = driver.find_element_by_xpath('//*[@id="rso"]/div[1]/div[2]/div/div[1]/div/a/h3')
posting.click()
```

### 옵션 선택 및 제출
XPath 등으로 `select` 요소를 선택한 다음에 각 옵션을 선택할 수 있지만, 아래와 같이  
`select` 객체 내에서 인덱스를 선택하거나, 옵션의 텍스트나 어떤 값을 통해 선택 가능


```python
URL = "https://phone.11st.co.kr/skbroadband/product.tmall?productNo=3901154914&internetProductId=NI00000901&btvProductId=NT00000772&setTopBoxProductId=T0027&combineLineCode=03"

driver = webdriver.Chrome(executable_path="./chromedriver")
driver.get(URL)
```


```python
from selenium.webdriver.support.ui import Select

select = Select(driver.find_element_by_name("select_name"))

select.select_by_index(index=2) # 인덱스
select.select_by_visible_text(text="option_text") # 옵션의 텍스트
select.select_by_value(value="역사 보기") # 특정 값
```

특정 선택 해제시, `deselect_~` 사용  

선택된 옵션 리스트를 얻으려면, `select.all_selected_options`으로 얻을 수 있고,  
첫 번째 선택된 옵션은 `select.first_selected_optoin`,  
가능한 옵션을 모두 보려면 `select.options` 이용


```python
# 제출 요소를 찾고 click()을 수행해도 되지만 다음도 가능
submit_btn.submit()
```

### Drag and Drop
어떤 일련의 동작을 수행하기 위해서는 `ActionChains`를 사용


```python
action_chains = ActionChains(driver)
action_chains.drag_and_drop(source=source, target=target).perform() # source 요소에서 target 요소로 Drag & Drop을 수행
```

### Window / Frame 이동
최신 웹 페이지에서는 frame 같은 것을 잘 사용하지 않지만, 예전에 만들어진 사이트라면 사용한 경우가 있음  
->  frame 안에 들어 있는 요소는 `find_element` 함수를 써도 찾아지지 않음  
-> 특정 frame으로 이동해야 함


```python
driver.switch_to_frame("frameName")
driver.switch_to_window("windowName")
# frame 내 subframe으로도 접근이 가능, .을 이용
driver.switch_to_frame("frameName.0.child")
```

windowName을 알고 싶다면 다음과 같은 링크가 있는지 찾아봐야 함  
```html
<a href="somewhere.html" target="windowName">Click here to open a new window</a>
```


```python
# frame 밖으로 나갈 때
driver.switch_to_default_content()
# 경고창으로 이동
alert = driver.switch_to.alert
```

### JavaScript 코드 실행


```python
# Name이 search_box인 요소의 값을 query의 값으로 변경하는 코드
driver.execute_script("document.getE")
```

## 브라우저 창 다루기
### 뒤로가기, 앞으로가기


```python
driver.back()
```


```python
driver.forward()
```

### 화면 이동
화면의 끝으로 내려가야 내용이 동적으로 추가되는 상황에서 사용


```python
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
```


```python
driver.execute_script("window.scrollTo(0, 100);")
```


```python
# 특정 요소까지 계속 찾는 경우
from selenium.webdriver import ActionChains

some_tag = driver.find_element_by_xpath('//*[@id="rso"]/div[7]/div/div/div[1]/div/a/h3')

ActionChains(driver=driver).move_to_element(some_tag).perform()
```

### 브라우저 최소화/최대화


```python
driver.minimize_window()
```


```python
driver.maximize_window()
```

### 스크린샷 저장


```python
driver.save_screenshot("screenshot.png")
```




    True



## 옵션
브라우저의 창 크기, 해당 기기의 정보 등을 설정 가능


```python
options = webdriver.ChromeOptions()
options.add_argument("window-size=1920, 1080")

driver = webdriver.Chrome("./chromedriver", options=options)
```

다양한 옵션을 지정해서 사용 할 수 있음

## ActionChains
마우스, 키보드 입력 등 연속 동작 실행


```python
ActionChains(driver).move_to_element(menu).click(hidden_submenu).perform()
```


```python
actions = ActionChains(driver)
actions.move_to_element(menu)
actions.click(hidden_submenu)
actions.perform()
```

## 경고 창 다루기


```python
from selenium.webdriver.common.alert import Alert

# 수락
Alert(driver).accept()
# 거절
Alert(driver).dismiss()
# 내용 출력
print(Alert(driver).text)
# 특정 키 입력 전달
Alert(driver).send_keys(keysToSend=Keys.ESCAPE)
```
