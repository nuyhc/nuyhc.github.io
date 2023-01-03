---
title: "ImportError: cannot import name 'ABCIndexClass' from
  'pandas.core.dtypes.generic' 문제해결"
date: 2022-05-11T01:30:45.963Z

categories:
  - TIL
tags:
  - Pandas
  - ABCIndexClass
  - vs code
  - issue
---

# 1. 문제 발생
pandas profiling을 설치하는 것까지는 문제가 없었으나, 막상 import해 사용하려고하니 문제가 발생했다.   
`ImportError: cannot import name 'ABCIndexClass' from 'pandas.core.dtypes.generic'` 오류가 발생해 검색해보니 역시나 버전 문제인거 같다.  

# 2. 해결
해결하는 방법은 크게 2가지가 있는거 같다.  
1. 다운그레이드  
2.8.0 이상 버전에서 폰트 설정방법이 변경되었다고해서 2.8.0 이상을 사용하려고 시도했으나, vision 모듈이 pandas 1.2에서만 사용 가능하다고 하는거 같다.  
해결하기 위해서는 pandas를 1.3에서 1.2대 버전으로 다운그레이드하는 방법이 있었다.  
다운그레이드하기는 귀찮아서 다른 해결 방법을 찾았다.
2. 소스코드 변경  
boolean.py의 내용을 변경해주면 해결이 가능하다. 실제로 이 방법으로 정상적으로 코드가 수행되게 되었다.  
경로는 `anaconda3/[가상환경]/lib/site-packages/visions/dtypes/boolean.py`다.  
가상환경을 아직 따로 만들어 사용하고 있지는 않아서 `base`를 이용했지만, 추후에 가상환경을 만들고 사용할 일이 있을수도 있으니까...
`boolean.py` 코드에서 `from pandas.core.dtypes.generic import ABCIndexClass, ABCSeries` 부분에서 `ABCIndexClass`만 `ABCIndex`로 변경하고 저장하면 된다.
3. vision 업데이트 기다리기  
정말로 급하지 않다면 가장 편안한 방법이 아닐까..

# 3. 정리
최근 여러가지 모듈들을 사용해보고있는데, 거의 매번 문제가 발생하는거 같다..   
검색이 잘되어서 해결이 금방되면 정말 좋은데 해결이 잘 안되는 경우도 있어서 좀 그렇다.. 그래도 이번 문제는 지난번 문제보다 해결이 빠르게된 편이다.