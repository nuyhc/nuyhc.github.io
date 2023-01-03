---
title: 군집화(Clustering) 정리
date: 2022-07-28T11:27:30.554Z

categories:
  - Programming
  - Machine Learning
tags:
  - Pandas
  - Numpy
  - Seaborn
  - matplot
  - sklearn
  - Tutorial
---

# 군집화(Clustering)

## K-평균 알고리즘
군집화에서 가장 일반적으로 사용되는 알고리즘  
1. 군집화의 기준이 되는 중심을 구성하는 임의의 점을 선택
2. 각 데이터는, 가장 가까운 곳에 위치한 중심점에 소속
3. 소속된 데이터의 평균값으로 중심점 이동
4. 위 과정 반복
5. 데이터의 중심점 소속 변경이 없으면 종로

### 장점과 단점
- 가장 많이 활용되는 알고리즘
- 알고리즘이 쉽고 간결함
- 거리 기반 알고리즘으로 속성의 개수가 많을 경우 정확도가 떨어짐 (PCA로 보완 가능)
- 반복 횟수가 많을수록 느려짐
- 몇 개의 군집(cluster)을 선택해야 할지 어려움


```python
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import pandas as pd

iris = load_iris()
irisDF = pd.DataFrame(data=iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
irisDF.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300)
kmeans.fit(irisDF)
```




    KMeans(n_clusters=3)




```python
kmeans.labels_
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
           0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
           0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2])




```python
irisDF["target"] = iris.target
irisDF["cluster"] = kmeans.labels_

iris_result = irisDF.groupby(["target", "cluster"])["sepal_length"].count()
iris_result
```




    target  cluster
    0       1          50
    1       0           2
            2          48
    2       0          36
            2          14
    Name: sepal_length, dtype: int64




```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(iris.data)

irisDF["pca_x"] = pca[:, 0]
irisDF["pca_y"] = pca[:, 1]
irisDF.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>target</th>
      <th>cluster</th>
      <th>pca_x</th>
      <th>pca_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>-2.684126</td>
      <td>0.319397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>-2.714142</td>
      <td>-0.177001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>-2.888991</td>
      <td>-0.144949</td>
    </tr>
  </tbody>
</table>
</div>




```python
marker0_ind = irisDF.loc[irisDF["cluster"]==0].index
marker1_ind = irisDF.loc[irisDF["cluster"]==1].index
marker2_ind = irisDF.loc[irisDF["cluster"]==2].index

plt.scatter(x=irisDF.loc[marker0_ind, "pca_x"], y=irisDF.loc[marker0_ind, "pca_y"], marker="o")
plt.scatter(x=irisDF.loc[marker1_ind, "pca_x"], y=irisDF.loc[marker1_ind, "pca_y"], marker="s")
plt.scatter(x=irisDF.loc[marker2_ind, "pca_x"], y=irisDF.loc[marker2_ind, "pca_y"], marker="^")

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA를 통해 2차원으로 축소한 3개의 군집화 시각화")
```




    Text(0.5, 1.0, 'PCA를 통해 2차원으로 축소한 3개의 군집화 시각화')




    
![png](/assets/images/sourceImg/clustering_files/clustering_6_1.png)
    


### 군집화 알고리즘 테스트를 위한 데이터 생성
- `make_blobs()`
- `make_classification()`


```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8)
print(X.shape, y.shape)
unique, counts = np.unique(y, return_counts=True)
print(unique, counts)
```

    (200, 2) (200,)
    [0 1 2] [67 67 66]
    


```python
clusterDF = pd.DataFrame(data=X, columns=["ftr1", "ftr2"])
clusterDF["target"] = y
clusterDF.head(3)
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
      <th>ftr1</th>
      <th>ftr2</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.703252</td>
      <td>-1.716435</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.738486</td>
      <td>-1.924543</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-5.330378</td>
      <td>7.648435</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
target_list = np.unique(y)
markers = ["o", "s", "^", "P", "D", "H", "x"]
for target in target_list:
    target_cluster = clusterDF.loc[clusterDF["target"]==target]
    plt.scatter(x=target_cluster["ftr1"], y=target_cluster["ftr2"], edgecolors="k", marker=markers[target])
plt.title("생성한 데이터 분포")
plt.show()
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_10_0.png)
    



```python
# 생성한 데이터로 군집화 수행
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=200)
cluster_labels = kmeans.fit_predict(X)
clusterDF["kmeans_label"] = cluster_labels

# 중심점 시각화
centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ["o", "s", "^", "P", "D", "H", "x"]

for label in unique_labels:
    label_cluster = clusterDF.loc[clusterDF["kmeans_label"]==label]
    center_x_y = centers[label]
    plt.scatter(x=label_cluster["ftr1"], y=label_cluster["ftr2"], edgecolors="k", marker=markers[label])
    
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color="white", alpha=0.9, edgecolors="k", marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color="k", alpha=0.9, edgecolors="k", marker="$%d$"%label)
plt.show()
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_11_0.png)
    


## 군집 평가 (Cluster Evaluation)
군집화는 분류와 비슷해보이나 성격이 다름  

### 실루엣 분석(Silhouette Analysis)
각 군집 간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타냄  
잘 분리됐다 -> 다른 군집과의 거리는 떨어져 있고, 동일 군집끼리의 데이터는서로 가깝게 잘 뭉쳐 있다  

#### 실루엣 계수(Silhouette coefficient)
개별 데이터가 갖는 군집화 지표  
해당 데이터가 같은 군집 내의 데이터와 얼마나 가깝게 군집화돼 있고, 다른 군집에 있는 데이터와는 얼마나 멀리 분리돼 있는지를 나타냄  

$s(i) = {b(i) - a(i) \over max(a(i), b(i))}$
- s(i): i번째 데이터 포인트의 실루엣 계수
- a(i): 같은 군집 내에 있는 다른 데이터 포인트와의 평균 거리
- b(i): 가장 가까운 다른 군집과의 평균 거리
- `-1에서 1` 사이의 값을 가지며, 1로 가까워질수록 근처 군집과 멀리 떨어져 있다는 것을 의미함
- 0에 가까울수록 근처 군집과 가까워지는 것을 의미함
- `-`값은 아예 다른 군집에 데이터가 존재한다는 것을 의미

#### 좋은 군집화
- 전체 실루엣 계수의 평균값(`silhoutte_score`)이 1에 가까운 경우
- 개별 군집의 평균값의 편차가 크지 않아야 함 -> 개별 군집의 실루엣 계수 평균값이 전체 실루엣 계수 평균값에서 크게 벗어나지 않는 것



```python
from sklearn.metrics import silhouette_samples, silhouette_score

iris = load_iris()
feature_names = ["sepal_length", "sepal_width", "petal_legth", "petal_width"]
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300).fit(irisDF)
irisDF["cluster"] = kmeans.labels_
```


```python
# 실루엣 계수 값
score_samples = silhouette_samples(iris.data, irisDF["cluster"])
print(score_samples.shape)
```

    (150,)
    


```python
irisDF["silhouette_coef"] = score_samples

# 모든 데이터의 평균 실루엣 계수 값
average_score = silhouette_score(iris.data, irisDF["cluster"])
print(average_score)
irisDF.head(3)
```

    0.5528190123564091
    




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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_legth</th>
      <th>petal_width</th>
      <th>cluster</th>
      <th>silhouette_coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
      <td>0.852955</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
      <td>0.815495</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
      <td>0.829315</td>
    </tr>
  </tbody>
</table>
</div>




```python
irisDF.groupby("cluster")["silhouette_coef"].mean()
```




    cluster
    0    0.798140
    1    0.417320
    2    0.451105
    Name: silhouette_coef, dtype: float64



### 최적화: 군집별 평균 실루엣 계수의 시각화를 통한 군집 개수 조정
개별 군집별로 적당히 분리된 거리를 유지하면서도 군집 내의 데이터가 서로 뭉쳐 있는 경우에 K-평균 알고리즘의 적절한 군집 개수가 설정됐다고 판단할 수 있음


```python
### 파이썬 머신러닝 완벽 가이드 부록 참고
### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features): 
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
```


```python
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, center_box=(-10.0, 10.0), shuffle=True)

visualize_silhouette([2, 3, 4, 5], X)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_19_0.png)
    


4개의 군집으로 나누는게 가장 좋음  


```python
visualize_silhouette([2, 3, 4, 5], iris.data)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_21_0.png)
    


2개의 군집으로 나누는게 가장 좋음  

각각의 데이터에서, 각각 4개와 2개로 군집한 경우가 각 군집간의 편차가 가장 적고 전체 평균과 비슷한 차이를 가지고 있기때문

## 평균 이동(Mean Shift)
`K-means`는 중심에 소속된 데이터의 평균 거리 중심으로 이동하는 반면, `평균 이동`은 중심을 데이터가 모여 있는 밀도가 가장 높은 곳으로 이동  
확률 밀도 함수(probability density function)을 이용해 군집 중심점을 찾음, 군집화의 개수를 지정하지 않음  
-> `KDE(Kernel Density Estimation)`를 이용해 확률 밀도 함수가 피크인 점을 군집 중심점으로 선정  

#### 평균 이동 절차
1. 개별 데이터의 특정 반경 내에ㅔ 주변 데이터를 포함한 데이터 분포도를 KDE 기반의 Mean Shift 알고리즘으로 계산
2. KDE로 계산된 데이터 분포도가 높은 방향으로 데이터 이동
3. 모든 데이터에 위 과정을 적용
4. 지정된 반복 횟수만큼 전테 데이터에 대해 KDE 기반 데이터를 이동시키면서 군집화 수행
5. 개별 데이터들이 모인 중심점을 군집 중심점으로 설정

#### KDE
$KDE = {1 \over n} \sum^n_{i=1}{K_h}(x-x_i) = {1 \over nh} \sum^n_{i=1}K{(x-x_i)\over h}$  
h는 대역폭(bandwidth)  
대역폭은 KDE 형태를 부드러운 형태로 평활화(smoothing)하는데 적용  
작은 h값은 오버피팅을, 큰 h값은 언더피팅을 유발할 수 있어 적절한 h를 선택하는 것이 평균 이동 군집화 알고리즘에서 중요  

**오직 대역폭의 크기에 따라 군집화를 수행**


```python
from sklearn.cluster import MeanShift

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.7)

meanshift = MeanShift(bandwidth=0.8)
cluster_labels = meanshift.fit_predict(X)
np.unique(cluster_labels)
```




    array([0, 1, 2, 3], dtype=int64)



4개의 군집으로 군집화


```python
meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
np.unique(cluster_labels)
```




    array([0, 1, 2], dtype=int64)



`h`값을 높이니 3개의 군집으로 군집화

#### 최적화: bandwidth


```python
from sklearn.cluster import estimate_bandwidth

bandwidth = estimate_bandwidth(X)
bandwidth
```




    1.876315412406745




```python
clusterDF = pd.DataFrame(data=X, columns=["ftr1", "ftr2"])
clusterDF["target"] = y

best_bandwidth = estimate_bandwidth(X)

meanshift = MeanShift(bandwidth=best_bandwidth)
cluster_labels = meanshift.fit_predict(X)

np.unique(cluster_labels)
```




    array([0, 1, 2], dtype=int64)




```python
# 시각화
clusterDF["meanshift_label"] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ["o", "s", "^", "x", "*"]

for label in unique_labels:
    label_cluster = clusterDF.loc[clusterDF["meanshift_label"]==label]
    center_x_y = centers[label]
    plt.scatter(x=label_cluster["ftr1"], y=label_cluster["ftr2"], edgecolors="k", marker=markers[label])
    
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color="gray", alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color="k", edgecolors="k", marker="$%d$"%label)

plt.show()
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_30_0.png)
    



```python
clusterDF.groupby("target")["meanshift_label"].value_counts()
```




    target  meanshift_label
    0       2                  67
    1       0                  67
    2       1                  66
    Name: meanshift_label, dtype: int64



## GMM(Gaussian Mixture Model)
군집화를 적용하고자 하는 데이터가 여러 개의 가우시안 분포(Gaussian Distribution)를 가진 데이터 집합들이 섞여서 생성된 것이라는 가정하에 군집화를 수행  
서로 다른 정규 분포에 기반해 군집화를 수행하는 방식  
-> 데이터에서 여러개의 정규 분포 곡선을 추출하고, 개별 데이터가 어떤 정규 분포에 속하는지 결정  
`n_components`로 추출할 정규 분포 곡선 개수를 지정해줘야 함


```python
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
irisDF["target"] = iris.target
```


```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3).fit(iris.data)
gmm_cluster_labels = gmm.predict(iris.data)

irisDF["gmm_cluster"] = gmm_cluster_labels
irisDF["target"] = iris.target

irisDF.groupby("target")["gmm_cluster"].value_counts()
```




    target  gmm_cluster
    0       0              50
    1       2              45
            1               5
    2       1              50
    Name: gmm_cluster, dtype: int64



K-Means 알고리즘은 원형의 범위에서 군집화를 수행 -> 데이터가 원형의 범위를 가질 경우 효과적   
데이터가 원형의 범위를 갖지 않는 경우에는, GMM이 더 효과적일 수 있음 -> GMM이 더 유연하다

## DBSCAN(Density Based Spatial Clustering od Applications with Noise)
밀도 기반 군집화의 대표적인 알고리즘  
간단하고 직관적인 알고리즘이지만, 기하학적으로 복잡한 데이터에도 효과적인 군집화가 가능  

- `epsilon`: 개별 데이터를 중심으로 입실론 반경을 갖는 원형의 영역
- `min points`: 개별 데이터의 입실론 주변 영역에 포함되는 타 데이터의 개수

### 데이터 포인트 정의
- 핵심 포인트(Core Point): 주변 영역 내에 최소 데이터 개수 이상의 타 데이터를 갖는 경우
- 이웃 포인트(Neighbor Point): 주변 영역 내에 위치한 타 데이터
- 경계 포인트(Border Point): 주변 영역 내에 최소 데이터 개수 이상의 이웃 포인트를 갖지 않지만, 핵심 포인트를 이웃 포인트로 갖는 경우
- 잡음 포인트(Noise Point): 최소 데이터 개수 이상의 이웃 포인트를 갖고 있지 않으며, 핵심 포인트도 이웃 포인트로 가지고 있지 않은 데이터

**DBSCAN은 핵심 포인트를 연결하면서 군집화를 구성하는 방식**


```python
from sklearn.cluster import DBSCAN, dbscan

dbscan = DBSCAN(eps=0.6, min_samples=8, metric="euclidean")
dbscan_labels = dbscan.fit_predict(iris.data)

irisDF = pd.DataFrame(data=iris.data, columns=feature_names)
irisDF["dbscan_cluster"] = dbscan_labels
irisDF["target"] = iris.target

irisDF.groupby("target")["dbscan_cluster"].value_counts()
```




    target  dbscan_cluster
    0        0                49
            -1                 1
    1        1                46
            -1                 4
    2        1                42
            -1                 8
    Name: dbscan_cluster, dtype: int64



`-1`에 속하는건 잡은 포인트(Noise Point)


```python
### 클러스터 결과를 담은 DataFrame과 사이킷런의 Cluster 객체등을 인자로 받아 클러스터링 결과를 시각화하는 함수  
def visualize_cluster_plot(clusterobj, dataframe, label_name, iscenter=True):
    if iscenter :
        centers = clusterobj.cluster_centers_
        
    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else :
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70,\
                    edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white',
                        alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k',\
                        edgecolor='k', marker='$%d$' % label)
    if isNoise:
        legend_loc='upper center'
    else: legend_loc='upper right'
    
    plt.legend(loc=legend_loc)
    plt.show()
```


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)
irisDF["ftr1"] = pca_transformed[:, 0]
irisDF["ftr2"] = pca_transformed[:, 1]

visualize_cluster_plot(dbscan, irisDF, "dbscan_cluster", iscenter=False)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_40_0.png)
    



```python
dbscan = DBSCAN(eps=0.8, min_samples=16, metric='euclidean')
dbscan_labels = dbscan.fit_predict(iris.data)

irisDF['dbscan_cluster'] = dbscan_labels
irisDF['target'] = iris.target

iris_result = irisDF.groupby(['target'])['dbscan_cluster'].value_counts()
print(iris_result)
visualize_cluster_plot(dbscan, irisDF, 'dbscan_cluster', iscenter=False)
```

    target  dbscan_cluster
    0        0                50
    1        1                50
    2        1                45
            -1                 5
    Name: dbscan_cluster, dtype: int64
    


    
![png](/assets/images/sourceImg/clustering_files/clustering_41_1.png)
    


`eps`의 값을 증가 시키면 잡음 포인트의 개수가 줄어듬

#### DBSCAN 적용 - make_circles()
DBSCAN와 다른 군집화 알고리즘 비교


```python
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=0, factor=0.5)
clusterDF = pd.DataFrame(data=X, columns=["ftr1", "ftr2"])
clusterDF["target"] = y

visualize_cluster_plot(None, clusterDF, "target", iscenter=False)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_43_0.png)
    


위 데이터를 군집화 알고리즘을 사용해 군집화 진행

##### K-Means


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, max_iter=1000)
kmeans_labels = kmeans.fit_predict(X)
clusterDF["kmeans_cluster"] = kmeans_labels

visualize_cluster_plot(kmeans, clusterDF, "kmeans_cluster", iscenter=True)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_45_0.png)
    


##### GMM


```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)
gmm_label = gmm.fit(X).predict(X)
clusterDF["gmm_cluster"] = gmm_label

visualize_cluster_plot(gmm, clusterDF, "gmm_cluster", iscenter=False)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_47_0.png)
    


#### DBSCAN


```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=10, metric="euclidean")
dbscan_labels = dbscan.fit_predict(X)
clusterDF["dbscan_cluster"] = dbscan_labels

visualize_cluster_plot(dbscan, clusterDF, "dbscan_cluster", iscenter=False)
```


    
![png](/assets/images/sourceImg/clustering_files/clustering_49_0.png)
    

