---
title: 기초 SQL 정리 2
date: 2022-05-14T12:59:28.222Z

categories:
  - TIL
tags:
  - SQL
  - MySQL
---

## 1. SQL의 통계 함수
1. 전체 데이터의 갯수 출력 - COUNT()
2. 합계 -> SUM()
3. 평균 -> AVG()
4. 최소/최대 -> MIN()/MAX()
```sql
SELECT COUNT(productid), SUM(price), AVG(price), MIN(price), MAX(price)
FROM products
```  
products 테이블에 있는 데이터의 갯수, 가격 총합, 가격 평균, 최소 가격, 최대 가격을 출력해주는 쿼리

## 2. GROUP BY
그룹별로 요약하기  
- GROUP BY 절에 사용한건 SELECT 절에도 무조건 들어가햠 함
- GROUP BY 절에 사용한 칼럼과 SELECT 절에 오는 칼럼의 순서는 달라도 상관 없지만 맞춰주는게 좋음
```sql
SELECT day, time, SUM(total_bill), SUM(tip)
FROM tips
GROUP BY day, time
```
tips 테이블에서 total_bill과 tip의 총합을 day와 time으로 묶어서 결과를 출력해주는 쿼리

## 3. HAVING
요약정보를 필터링함  
- GROUP BY 이후에 필터링 함
- WHERE는 GROUP BY 이전에 필터링 
```sql
SELECT day, SUM(total_bill) AS revenue
FROM tips
GROUP BY day
HAVING revenue>=1000
```  
전체 가격이 1000 이상인 경우 일별로 그룹화해 출력하는 쿼리  
```sql
SELECT day, SUM(total_bill) AS revenue
FROM tips
WHERE sex='Female'
GROUP BY day
```
여성들의 계산 가격을 일별로 그룹화해 출력하는 쿼리
```sql
SELECT day, SUM(total_bill) AS revenue
FROM tips
WHERE sex='Female'
GROUP BY day
HAVING SUM(total_bill)>=200
```
