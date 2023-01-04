---
title: 파이썬 가상 환경 사용하기
date: 2022-06-08T14:18:15.174Z

categories:
  - TIL
tags:
  - Git
  - pip
  - anaconda
  - homebrew
---

# 파이썬 가상 환경 사용하기
## 가상 환경(virtual environment)을 사용하는 이유
하나의 PC에서 프로젝트별로 내가 원하는 환경(파이썬 버전, 라이브러리 버전 등)을 구축하기 위해서이다.  
가상 환경을 이용하면 버전 충돌도 예방할 수 있고, 패키지의 구성도 관리할 수 있다.

## 가상 환경 설정하기
다양한 명령어들이 있지만, 기본적으로 사용하는데 있어 필요한 명령어들만 정리했다.
### 1. venv
`venv`는 파이썬 기본 모듈이며, 가상 환경을 구성할 폴더 안에서 생성해주면 된다.  
#### 생성
```bash
$ cd <프로젝트 경로>
$ python -m venv [가상환경 이름]
$ cd [가상환경 이름]
```
`venv` 관례적으로 가상환경 이름은 `.venv`를 쓰는듯 한다.

#### 활성화
| 플랫폼 | 셸 | 명령어 |
| --- | --- | --- |
| POSIX | bash/zsh | source [가상환경 이름]/bin/activate |
| - | fish | source [가상환경 이름]/bin/activate.fish |
| - | csh/tcsh | source [가상환경 이름]/bin/activate.csh |
| - | PowerShell Core | [가상환경 이름]/bin/Activate.ps1 |
| Window | cmd | [가상환경 이름]\Scripts\activate.bat |
| - | PowerShell | [가상환경 이름]\Scripts\Activate.ps1|

#### 비활성화
```bash
deactivate
```

#### 삭제
삭제 명령어는 따로 없고, 생성된 폴더를 삭제해주면 된다.

### 2. conda
개인적으로 `venv`보다는 선호하는 방식이다.  
#### 생성
```bash
conda create -n [가상환경 이름] python=버전
```
`python=버전`을 생략하면 가장 최신 버전으로 생성된다.  
예를들어, `conda create -n [가상환경 이름] python=3.7`이라고하면,  
3.7 중의 가장 최신인 3.7.13으로 설치가 된다.  

#### 환경 확인
```bash
conda info --env
conda env list
```
위 두가지 방식으로 현재 가상 환경 상태를 확인할 수 있다.  
둘 중 무엇을 쓰던 상관없다.  

#### 활성화
```bash
conda activate [가상환경 이름]
```

#### 비활성화
```bash
conda deactivate [가상환경 이름]
```

#### 삭제
```bash
conda remove -n [가상환경 이름] --all
```

## 유용한 관련 명령어
```bash
pip freeze > requirements.txt
```
위 명령어는, 해당 가상 환경에 설치되어있는 라이브러리들 `requirements.txt`에 저장해준다.  
```bash
pip install -r 파일이름.txt
pip uninstall -r 파일이름.txt
```
위 두 명령어는, `파일이름.txt`를 기반으로 라이브러리를 설치하거나 삭제한다.

## 여담
### 1. 쉘(shell)과 프롬포트(prompt)
쉘은, 운영체제의 커널과 사용자 사이의 다리 역할을하며, 사용자가 내린 명령을 해석해 프로그램을 실행한다. 종류로는 `bash, C, tsch, zsh` 등이 있다.  

프롬포트는, 사용자의 입력을 받을 준비가 되어 있다는 것을 사용자에게 알려주기 위해 화면에 나타내고 있는 신호다.  

명령어를 입력할 때는 **쉘**을 이용하자.

### 2. pip, conda, homebrew
파이썬 패키지를 설치할 때 가장 많이 사용하는 패키지 설치 도구들이다.  
새로 생성한 가상 환경에 라이브러리를 설치할 때, 무엇을 쓰던 상관없다.

정확한 비유인지는 모르겠지만, 중고나라와 당근마켓이 모두 물건을 사고 팔 수 있는 플랫폼이지만 운영자는 다르다.  
마찬가지로, `pip, conda, homebrew` 모두 라이브러리를 설치할 수 있지만 운영자는 다른 셈이다. 개인적으로는 `pip`를 선호하는 편.

### 3. CLI가 어렵다면 GUI로..
개인적으로는 `CLI`를 선호하지만, 경우에 따라서 `GUI`를 사용하고 있다.  
사실 가상 환경 설정의 경우, `CLI`에서는 복잡하지만, `GUI`에서는 굉장히 직관적인 편이다.  

vs code를 사용하는 경우, `python envirment management` 확장 기능을 이용해 가상 환경을 생성/사용/삭제가 편리하고,  
Anaconda Navigation을 이용해서도 가상 환경 관리가 편한 편이다.  
`CLI`가 익숙하지 않다면 `GUI`를 사용하는걸 추천한다.
