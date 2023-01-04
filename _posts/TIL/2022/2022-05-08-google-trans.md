---
title: Google Trans로 번역하기
date: 2022-05-08T10:01:33.341Z

categories:
    - TIL
tags:
    - GoogleTrans
---

# 1. Google Trans로 번역하기
## 발생 가능한 문제
- `AttributeError: 'NoneType' object has no attribute 'group'`: googletrans 버전 관련 에러
  - 버전을 명시해 설치해줘여 함: `pip install googletrans==3.1.0a0`

# 2. Googletrans
- 구글에서 제공하는 언어 감지/번역 모듈
- 일일 사용량이 정해져있다고 함
- 모듈을 큰 기능 단위로묶어둔것 -> Library
 

```python
from googletrans import Translator
print(Translator)
```

    <class 'googletrans.client.Translator'>
    

# 3. 언어 감지하기
1. 번역기를 만든다
2. 언어 감지를 원하는 문장을 설정한다
3. 언어를 감지한다


```python
# 1. 번역기를 만든다
from googletrans import Translator

translator = Translator()

# 2, 3. 언어 감지를 원하는 문장을 설정
sentence = "안녕하세요"
detected = translator.detect(sentence)

print(detected)
```

    Detected(lang=ko, confidence=1)
    

- lang: 감지한 언어
- confidence: 신뢰도


```python
from googletrans import Translator

translator = Translator()

sentence = input("번역할 문장: ")
detected = translator.detect(sentence)

print(detected.lang)
```

    ko
    

# 4. 번역하기
1. 번역기를 만든다
2. 번역을 원하는 문장을 설정한다
3. 번역을 원하는 언어를 설정한다
4. 번역을 한다

## translate
`translate(text, dest, src)`
- text: 번역을 원하는 문장
- dest: 번역할 언어
- src: (optional)text의 언어  
함수에 언어 감지 능력이 내장되어있어, src는 생략 가능

## dest code
|언어|코드|언어|코드|
|---|---|---|---|
|프랑스어|fr|아랍어|ar|
|배트남어|vi|독일어|de|
|스페인어|es|몽골어|mn|
|중국어|zh-CN|힌디어|hi|


```python
from googletrans import Translator

translator = Translator()

sentence = input("번역할 문장: ")

result = translator.translate(sentence, 'en')

print(result)
```

    Translated(src=ko, dest=en, text=Hello, this is a translation test., pronunciation=None, extra_data="{'translat...")
    


```python
from googletrans import Translator

translator = Translator()

sentence = input("번역할 문장: ")

result = translator.translate(sentence, 'en')
detedcted = translator.detect(sentence)

print(detedcted.lang, ":", sentence)
print(result.dest, ":", result.text)
```

    ko : 안녕하세요, 번역 테스트입니다.
    en : Hello, this is a translation test.
    
