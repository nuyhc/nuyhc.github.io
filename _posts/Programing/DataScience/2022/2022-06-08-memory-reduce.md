---
title: 메모리 부담 줄이기
date: 2022-06-07T16:01:31.156Z

categories:
  - Programming
  - DataScience
tags:
  - Pandas
  - Parquet
---

# 메모리 부담(용량) 줄이기
메모리의 사용량을 줄일 수 있는 방법은 다음과 같다.  
1. `pandas`의 `downcast`
2. `parquet` 저장 형식

## Downcast
`pandas`의 `downcast`는 자료형의 타입을 변경해 메모리의 사용량을 줄이는 방식이다.


```python
import pandas as pd
import numpy as np
import os
```


```python
df = pd.read_csv("../data/HP_T60_2020_1.CSV", encoding="cp949")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10994920 entries, 0 to 10994919
    Data columns (total 15 columns):
     #   Column       Dtype  
    ---  ------       -----  
     0   기준년도         int64  
     1   가입자 일련번호     int64  
     2   처방내역일련번호     int64  
     3   일련번호         int64  
     4   성별코드         int64  
     5   연령대코드(5세단위)  int64  
     6   시도코드         int64  
     7   요양개시일자       object 
     8   약품일반성분명코드    object 
     9   1회 투약량       float64
     10  1일투약량        int64  
     11  총투여일수        int64  
     12  단가           float64
     13  금액           int64  
     14  데이터 공개일자     object 
    dtypes: float64(2), int64(10), object(3)
    memory usage: 1.2+ GB
    

데이터는 의료 처방전 데이터로 대용량 데이터를 이용했다.  
초기 데이터를 불러와 메모리 사용량을 확인해보면 **1.2+GB**를 사용하고 있다. 


```python
df.dtypes
```




    기준년도             int64
    가입자 일련번호         int64
    처방내역일련번호         int64
    일련번호             int64
    성별코드             int64
    연령대코드(5세단위)      int64
    시도코드             int64
    요양개시일자          object
    약품일반성분명코드       object
    1회 투약량         float64
    1일투약량            int64
    총투여일수            int64
    단가             float64
    금액               int64
    데이터 공개일자        object
    dtype: object



|    | Data type   | Description                                                          |
|---:|:------------|:---------------------------------------------------------------------|
|  0 | bool        | Boolean (True or False) stored as a byte                             |
|  1 | int         | Platform integer (normally either int32 or int64)                    |
|  2 | int8        | Byte (-128 to 127)                                                   |
|  3 | int16       | Integer (-32768 to 32767)                                            |
|  4 | int32       | Integer (-2147483648 to 2147483647)                                  |
|  5 | int64       | Integer (9223372036854775808 to 9223372036854775807)                 |
|  6 | uint8       | Unsigned integer (0 to 255)                                          |
|  7 | uint16      | Unsigned integer (0 to 65535)                                        |
|  8 | uint32      | Unsigned integer (0 to 4294967295)                                   |
|  9 | uint64      | Unsigned integer (0 to 18446744073709551615)                         |
| 10 | float       | Shorthand for float64.                                               |
| 11 | float16     | Half precision float: sign bit, 5 bits exponent, 10 bits mantissa    |
| 12 | float32     | Single precision float: sign bit, 8 bits exponent, 23 bits mantissa  |
| 13 | float64     | Double precision float: sign bit, 11 bits exponent, 52 bits mantissa |
| 14 | complex     | Shorthand for complex128.                                            |
| 15 | complex64   | Complex number, represented by two 32-bit floats                     |
| 16 | complex128  | Complex number, represented by two 64-bit floats                     |

데이터 타입을보면, 타입마다 표현할 수 있는 수의 범위가 다르다.  
위 데이터에서, 1일 투약량 같은 경우는 상식적으로 `int64`의 데이터 타입을 가질 필요가 없다. `int8` 타입이나 `uint8` 타입만으로도 충분하다.  

이런식으로, 데이터가 가지는 범위를 파악해 데이터 타입을 조정해 메모리의 사용량을 줄이는 방식이 `donwcast`를 이용한 방식이다.


```python
# 처방
for col in df.columns:
    dtype_name = df[col].dtypes.name
    col_name = df[col].name
    except_col = ["처방내역일련번호"] # 해당 컬럼은 downcast를하지 않음
    if col_name in except_col:
        pass
    else:
        if dtype_name.startswith("int"):
            df[col] = pd.to_numeric(df[col], downcast="unsigned")
        elif dtype_name.startswith("float"):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif dtype_name.startswith("bool"):
            df[col] = df[col].astype("int8")
    
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10994920 entries, 0 to 10994919
    Data columns (total 15 columns):
     #   Column       Dtype  
    ---  ------       -----  
     0   기준년도         uint16 
     1   가입자 일련번호     uint32 
     2   처방내역일련번호     int64  
     3   일련번호         uint8  
     4   성별코드         uint8  
     5   연령대코드(5세단위)  uint8  
     6   시도코드         uint8  
     7   요양개시일자       object 
     8   약품일반성분명코드    object 
     9   1회 투약량       float32
     10  1일투약량        uint8  
     11  총투여일수        uint16 
     12  단가           float32
     13  금액           uint32 
     14  데이터 공개일자     object 
    dtypes: float32(2), int64(1), object(3), uint16(2), uint32(2), uint8(5)
    memory usage: 597.7+ MB
    

메모리 사용량이 `downcast` 이후 약 절반정도 줄어들었습니다.  
아래와 같이, 파일을 읽어올 때 적용하는 방법도 있습니다.


```python
df2 = pd.read_csv("../data/HP_T60_2020_1.CSV", encoding="cp949", dtype={"기준년도":"int16", "가입자 일련번호":"uint32", "일련번호":"uint8", "성별코드":"uint8", "연령대코드(5세단위)":"uint8", "시도코드":"uint8", "1회 투약량":"float32", "1일투약량":"uint8", "총투여일수":"uint16", "단가":"float32", "금액":"uint32"})
```


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10994920 entries, 0 to 10994919
    Data columns (total 15 columns):
     #   Column       Dtype  
    ---  ------       -----  
     0   기준년도         int16  
     1   가입자 일련번호     uint32 
     2   처방내역일련번호     int64  
     3   일련번호         uint8  
     4   성별코드         uint8  
     5   연령대코드(5세단위)  uint8  
     6   시도코드         uint8  
     7   요양개시일자       object 
     8   약품일반성분명코드    object 
     9   1회 투약량       float32
     10  1일투약량        uint8  
     11  총투여일수        uint16 
     12  단가           float32
     13  금액           uint32 
     14  데이터 공개일자     object 
    dtypes: float32(2), int16(1), int64(1), object(3), uint16(1), uint32(2), uint8(5)
    memory usage: 597.7+ MB
    

## Parquet
### Apache Parquet
- 효율적인 데이터 저장 및 검색을 위해 설계된 오픈 소스
  - 열 지향 데이터 파일 형식 -> 같은 데이터 형식끼리 저장되기 때문에 압축률이 높아짐
- 복잡한 데이터를 대량으로 처리하기 위해 향상된 성능과 효율적인 데이터 압축 및 인코딩 체계를 제공
- 여러 언어를 지원
- Twitter와 Cloudera의 협업으로 만들어짐


```python
df = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col1</th>
      <th>col2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
file_path_parquet = 'df.parquet.gzip'
file_path_csv  = 'df.csv'

df.to_parquet(file_path_parquet, compression="gzip")
df.to_csv(file_path_csv, index=False)
```


```python
os.stat(file_path_parquet).st_size, os.stat(file_path_csv).st_size
```




    (2353, 21)



용량이 작은 경우, `parquet`은 메타 데이터를 가지고 있기 때문에 `csv` 형식보다 용량이 큽니다.


```python
df = pd.read_csv("../data/HP_T60_2020_1.CSV", encoding="cp949")
```


```python
df.to_parquet(file_path_parquet, compression="gzip")
df.to_csv(file_path_csv, index=False)

format(os.stat(file_path_parquet).st_size, ","), format(os.stat(file_path_csv).st_size, ",")
```




    ('96,430,605', '897,541,690')




```python
# 파일 사이즈 bytes 로 표기하기
def convert_bytes(num):
    for fs in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024:
            return (f"{num:.1f} {fs}")
            break
        num /= 1024


def file_size(file_path):
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)


file_size(file_path_parquet), file_size(file_path_csv), file_size("")
```




    ('92.0 MB', '856.0 MB', None)


