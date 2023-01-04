---
title: Running cells with 'Python 3.9.12 ('가상환경')' requires ipykernel package. 오류해결
date: 2022-06-10T14:13:33.274Z

categories:
  - TIL
tags:
  - vs code
  - pip
  - anaconda
  - jupyter
  - issue
---

# 1. 문제 발생
평소 `conda base`만 사용하다, 진행하는 프로젝트를 퍼블리싱 해 볼 생각으로 해당 프로젝트를 위한 가상환경을 생성했다.  
생성 후 코드를 실행 시켜보자, 다음과 같은 에러가 발생했다.
```
Running cells with 'Python 3.9.12 ('가상환경')' requires ipykernel package.
Run the following command to install 'ipykernel' into the Python environment. 
Command: 'conda install -n 가상환경 ipykernel --update-deps --force-reinstall'
```

친절하게도 마지막 줄에 해결을 위한 명령어를 알려줬지만, vsc 쉘에 매직키워드로는 해결이되지 않고 동일한 메시지만 반복해서 등장했다.

# 2. 해결
윈도우 기준 `ctrl + \``키를 이용해 터미널에서 실행하거나, Anaconda powershell 같은 터미널을 이용해  
주어진 명령어를 입력하니 해결되었다.


