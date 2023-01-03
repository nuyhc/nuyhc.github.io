---
title: The kernel failed to start ~ 오류
date: 2022-05-30T14:49:40.293Z

categories:
  - TIL
tags:
  - issue
  - Pandas
  - vs code
  - pip
  - jupyter

toc: true
toc_sticky: true
---

# 1.문제 발생
vs code를 이용해 ipynb 파일을 이용하고 있었다, 그러던 와중에  
`The kernel failed to start as 'path' could not be imported from ~`  
와 같은 메시지가 출력되면서 `import`를하는 첫 셀부터 작동이 되지 않았다.  
뭔가 딱히 건드린거 같지는 않은데 갑자기 커널이 죽어버려서 굉장히 당황스럽다.  

검색을하다가 안 사실은, 같은 모듈을 두 번  `import`하면 발생한다고 한다.  
나 같은 경우에는 초기에 `pandas`를 불러와 놓고 다시 한번 더 부르는 과정에서 오류가 발생한거 같다.

# 2. 해결
해결 방법은 굉장히 간단하다.  
`pip`나 `conda` 명령어로 중복된 모듈을 `uninstall`하고 재설치만 해주면 된다.

# 3. 정리
모듈은 한번만 불러와서 사용하자.. 중간에 불가피하게 모듈을 불러오게 됬는데, 그냥 상단에서 한번에 다 `import`하자..