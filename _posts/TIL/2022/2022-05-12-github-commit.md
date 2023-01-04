---
title: GitHub 원격 저장소에 커밋 올리고 내려받기
date: 2022-05-12T14:11:25.799Z

categories:
  - TIL
tags:
  - Git
  - GitHub
---

# 1. 원격 저장소 만들기
로컬에서만 git을 사용하는거면 앞선 내용만으로도 충분히 혼자만의 버전 관리가 가능하다.  
하지만 git을 사용하는 목적이, 다양한 사람들과 협업을 하기 위해서라는 점을 생각하다면 원격 저장소를 이용하는 편이 좋다.  
본인은 처음 git과 github를 사용하게 된 이유는, 데스크탑과 노트북 사이의 파일을 최신으로 동일하게 유지하기 위함이었다.  

깃 호스팅(GitHub 등)에 협업 할 공간을 원격 저장소라고하고, 로컬 저장소와 구별하는 개념으로 쉽게 **웹사이트에 프로젝트를 위한 공용 폴더**라고 생각하면된다.  
GitHub에서는 원격 저장소를 레포지토리(repository)라고 부른다.  
![img](https://github.com/nuyhc/github.io.archives/blob/main/create_github_repo.png?raw=true)  
GitHub 사이트에 접속해 우측 상단 `+` 버튼을 눌러 `New repoitory` 버튼을 눌러 새로운 원격 저장소를 생성한다.

# 2. 로컬 저장소와 원격 저장소 연결하기
로컬 저장소와 원격 저장소를 연결하는 방법은 크게 2가지가 있다.  
1. git clone  
미리 github에서 레포지토리를 생성하고 로컬 저장소에 클론을 받는 방식이다.  
개인적으로 선호하는 방식인데, `git clone [클론할 원격 저장소 주소]`만 입력해도 모든 설정이 끝나기 때문이다.  
    1. 원격 저장소를 생성한다
    2. 원격 저장소를 클론할 로컬 저장소를 생성한다
    3. 로컬 저장소에서 Git Bash를 열어 `git clone [클론할 원격 저장소 주소]`를 수행한다
2. git remote add  
로컬 저장소에서 원격 저장소를 설정해주는 방식이다. 
`git remote add origin ["원격 저장소 주소"]`를 수행해 연결하게 된다.  
깃 허브에 로그인하라는 창이 생성될 수도 있다.
    1. 원격 저장소를 생성한다.
    2. 로컬 저장소에서 Git Bash를 열어 `git init` 명령을 수행한다.
    3. `git remote add origin ["원격 저장소 주소"]`를 수행한다.
### clone & remote
로컬 저장소와 원격 저장소를 연결하는 명령어로 큰 차이는 없다.  
단순히 로컬에서 원격을 연결 할 것인지(`remote`), 원격에서 로컬을 연결 할 것인지(`clone`) 차이인거 같다.

# 3. 원격 저장소로 커밋 올리기
앞서 `git add`, `git commit` 등의 과정을 수행했다면 상관 없지만, 만약 수행하지 않았다면 지금이라도 수행해야한다.  
원격 저장소로 파일을 올리기 위해서는 스테이징 필드에서 `push`를 이용해 올리는데 `git add`와 `git commit`이 스테이징 필드에 파일을 올리는 과정이기 때문이다.  
  
`git push origin master` 명령어를 이용하면 로컬 저장소의 변경 사항을 원격 저장소로 반영할 수 있다.

# 3. master와 main
Git과 GitHub는 서로 다른 단체이므로 기본 브랜치가 master와 main이 있다.  
깃허브에서는 기본적으로 main을 사용하는데, `git init`으로 시작을 하게되면 master가 기본 브랜치가 된다.

# 4. pull과 fetch
로컬 환경이 변경되거나 원격 저장소에서 바로 수정한 사항이 있으면, 로컬 저장소에도 반영을 해야한다. 이때 사용하는 명령어가 `git pull`과 `git fetch`이다.  

`pull`과 `fetch`의 차이는 변경 사항을 합병(merge)을 하냐 안하냐의 차이인데, `pull`은 합병을 진행하고 `fetch`는 합병을 하지 않는다. 기본적으로 `pull`만 주로 사용한다.

`git pull` 형식으로 사용하고 변경 사항을 갱신하지 않고 새로운 업데이트를 할려고하면 오류가 발생한다.