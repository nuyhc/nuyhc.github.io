---
title: 추천 시스템 기본 정리
date: 2022-08-08T08:28:19.267Z

categories:
  - Programming
  - Machine Learning
tags:
  - sklearn
  - Pandas
  - Numpy
  - Tutorial
---

# 추천 시스템(Recommendations)
하나의 컨텐츠를 선택했을 때, 선택된 콘텐츠와 연관된 추천 콘테츠가 얼마나 사용자의 관심을 끌고 개인에게 맞춘 콘텐트를 추천했는지가 중요함  
사용자 자신도 좋아하는지 몰랐던 취향을 시스템이 발견하고 그에 맞는 콘텐츠를 추천해주는 것  

## 추천 시스템 유형
- 콘텐츠 기반 필터링(Content Based Filtering)
- 협업 필터링(Collaborative Filtering)
  - 최근접 이웃(Nearest Neighbor) 기반 협업 필터링
  - 잠재 요인(Latent Factor) 협업 필터링

초창기에는 콘텐츠 기반 필터링이나 최근접 이웃 기반 협업 필터링이 주로 사용됐지만, 넷플릭스 추천 시스템 경연 대회에서 행렬 분해(Matrix Factorization) 기법을 이용한 잠재 요인 협업 필터링 방식이 우승하며, 잠재 요인 협업 필터링이 주류가 됨  
하지만, 서비스하는 아이템의 특성에 따라 콘텐츠 기반 필터링이나 최근접 이웃 기반 협업 필터링 방식을 유지하는 사이트도 많으며, 개인화 특성을 좀 더 강화하기 위해 하이브리드 형식으로 콘텐츠 기반과 협업 기반을 적절히 결합해 사용하는 경우도 있음  

## 콘텐츠 기반 필터링 추천 시스템
특정한 아이템을 선호하는 경우, 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템을 추천하는 방식

## 최근접 이웃 협업 필터링 (메모리 협업 필터링)
사용자 행동 양식(User Behavior)만을 기반으로 추천을 수행하는 것이 협업 필터링(Collaborative Filtering) 방식  
협업 필터링의 주요 목표는 사용자-아이틈 평점 매트릭스와 같은 축적된 사용자 행동 데이터를 기반으로 예측 평가(Predicted Rating)하는 것  

1. 사용자 기반(User-User)
2. 아이템 기반(Item-Item)

## 잠재 요인 협업 필터링
사용자-아이템 평점 매트릭스 속에 숨어 있는 잠재 요인을 추출해 추천 예측을 할 수 있게 하는 기법  
대규모 다차원 행렬을 SVD와 같은 차원 축소 기법으로 분해하는 과정에서 잠재 요인을 추출 -> 행렬 분해(Matrix Factorization)

### 행렬 분해의 이해
다차원의 매트릭스를 저차원 매트릭스로 분해하는 기법  
- SVD(Singluar Vector Decomposition)
- NMF(Non-Negative Matrix Factorization)

M개의 사용자 행과 N개의 아이템 열을 가진 평점 행렬 R은, 행렬 분해를 통해 사용자-K 차원 잠재 요인 행렬 P(M\*K)와 K 차원 잠재 요인 - 아이템 행렬 Q.T(K\*N)로 분해 가능  
$R = P*Q.T$  
$M*N = (M*K)*(K*N)$  
- M: 총 사용자 수
- N: 총 아이템 수
- K: 잠재 요인의 차원 수
- R: M*N 차원의 사용자-아이템 평점 행렬
- P: M*K 차원의 사용자-잠재 요인 행렬
- Q: N*K 차원의 아이템-잠재 요인 행렬

$평점 데이터 = r_(u, i) = p_u * q^t_i$  

SVD는 널(NaN)값이 없는 행렬에만 적용 가능 -> SGD나 ALS 방식을 이용해 SVD를 수행하면 널값이 있어도 행렬 분해 가능

### 확률적 경사 하강법(SGD)을 이용한 행렬 분해
P와 Q 행렬로 계산된 예측 R 행렬 값이 실제 R 행렬 값과 가장 최소한의 오류를 가질 수 있도록 반복적인 비용 함수 최적화를 통해 P와 Q를 유추해내는 것

1. P와 Q를 임의의 값을 가진 행렬로 설정
2. P와 Q.T 값을 곱해 예측 R 행렬을 계산하고 오류 계산
3. 오류를 최소화할 수 있도록 P와 Q 행렬을 각각 업데이트
4. 지속적인 업데이트로 근사화


```python
import numpy as np

R = np.array([[4, np.NaN, np.NaN, 2, np.NaN],
              [np.NaN, 5, np.NaN, 3, 1],
              [np.NaN, np.NaN, 3, 4, 4],
              [5, 2, 1, 2, np.NaN]
              ])

num_users, num_items = R.shape
K=3

P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))
```


```python
from sklearn.metrics import mean_squared_error

def get_rmse(R, P, Q, non_zeros):
    error = 0
    full_pred_matrix = np.dot(P, Q.T)
    
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    return rmse
```


```python
non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j]>0]

steps = 1000
lr = 0.01
r_lambda = 0.01

for step in range(steps):
    for i, j, r in non_zeros:
        eij = r - np.dot(P[i, :], Q[j, :].T)
        P[i, :] = P[i, :] + lr*(eij*Q[j, :] - r_lambda*P[i, :])
        Q[j, :] = Q[j, :] + lr*(eij*P[i, :] - r_lambda*Q[j, :])
    rmse = get_rmse(R, P, Q, non_zeros)
    if step%50==0:
        print("### iteration step: ", step, "rmse: ", rmse)
```

    ### iteration step:  0 rmse:  3.1345564002413893
    ### iteration step:  50 rmse:  0.39438701928608744
    ### iteration step:  100 rmse:  0.16952770328264244
    ### iteration step:  150 rmse:  0.09016691819192837
    ### iteration step:  200 rmse:  0.0509578535617624
    ### iteration step:  250 rmse:  0.03100211778544041
    ### iteration step:  300 rmse:  0.021267015824906324
    ### iteration step:  350 rmse:  0.0169231089073883
    ### iteration step:  400 rmse:  0.015132166119500107
    ### iteration step:  450 rmse:  0.014405749540362495
    ### iteration step:  500 rmse:  0.01409838092830005
    ### iteration step:  550 rmse:  0.013959127186243741
    ### iteration step:  600 rmse:  0.013891452315489872
    ### iteration step:  650 rmse:  0.013856370174916146
    ### iteration step:  700 rmse:  0.013836987935265613
    ### iteration step:  750 rmse:  0.013825474052221563
    ### iteration step:  800 rmse:  0.013818001162102246
    ### iteration step:  850 rmse:  0.013812630563516705
    ### iteration step:  900 rmse:  0.013808358582966862
    ### iteration step:  950 rmse:  0.013804659578060416
    


```python
# 예측 행렬
np.dot(P, Q.T)
```




    array([[3.99236006e+00, 1.39588064e+00, 1.37018804e+00, 1.99271581e+00,
            2.81828405e+00],
           [1.28744560e-03, 4.97970705e+00, 2.26698810e+00, 2.98589620e+00,
            1.00935671e+00],
           [4.47454811e+00, 4.10541953e+00, 2.98564029e+00, 3.98554835e+00,
            3.98210226e+00],
           [4.97524652e+00, 1.99127333e+00, 1.00705429e+00, 1.99858986e+00,
            2.70502898e+00]])



## 콘텐츠 기반 필터링 실습 - TMDB 5000 영화 데이터 셋
[Kaggle TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
사용자가 특정 영화를 감상하고, 그 영화를 좋아했다면 그 영화와 비슷한 특성/속성, 구성 요소 등을 가진 다른 영화를 추천하는 것  
장르 값의 유사도를 비교한 뒤 그 중 높은 평점을 갖는 영화를 추천하는 방식


```python
import pandas as pd
import numpy as np

movies = pd.read_csv("./tmdb_5000_movies.csv")
print(movies.shape)
movies.head(3)
```

    (4803, 20)
    




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
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies_df = movies[["id", "title", "genres", "vote_average", "vote_count", "popularity", "keywords", "overview"]]
```


```python
from ast import literal_eval
movies_df["genres"] = movies_df["genres"].apply(literal_eval)
movies_df["keywords"] = movies_df["keywords"].apply(literal_eval)
```


```python
movies_df["genres"] = movies_df["genres"].apply(lambda x: [y["name"] for y in x])
movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [y["name"] for y in x])
movies_df.head(2)
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
      <th>id</th>
      <th>title</th>
      <th>genres</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>popularity</th>
      <th>keywords</th>
      <th>overview</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>[Action, Adventure, Fantasy, Science Fiction]</td>
      <td>7.2</td>
      <td>11800</td>
      <td>150.437577</td>
      <td>[culture clash, future, space war, space colon...</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>6.9</td>
      <td>4500</td>
      <td>139.082615</td>
      <td>[ocean, drug abuse, exotic island, east india ...</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
    </tr>
  </tbody>
</table>
</div>



### 장르 콘텐츠 유사도 측정
- Count 기반으로 피처 벡터화
- 코사인 유사도를 이용해 비교


```python
from sklearn.feature_extraction.text import CountVectorizer

movies_df["genres_literal"] = movies_df["genres"].apply(lambda x: (" ").join(x))
count_vect = CountVectorizer(min_df=0, ngram_range=(1, 2))
genre_mat = count_vect.fit_transform(movies_df["genres_literal"])

genre_mat.shape
```




    (4803, 276)




```python
from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)
genre_sim[:1]
```




    array([[1.        , 0.59628479, 0.4472136 , ..., 0.        , 0.        ,
            0.        ]])



### 장르 콘텐츠 필터링을 이용한 영화 추천



```python
def find_sim_movies(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df["title"]==title_name]
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :top_n]
    
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]
```


```python
find_sim_movies(movies_df, genre_sim.argsort()[:, ::-1], "The Godfather", 10)[["title", "vote_average"]]
```

    [[2731 1243 3636 1946 2640 4065 1847 4217  883 3866]]
    




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
      <th>title</th>
      <th>vote_average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2731</th>
      <td>The Godfather: Part II</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>Mean Streets</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>3636</th>
      <td>Light Sleeper</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>The Bad Lieutenant: Port of Call - New Orleans</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>Things to Do in Denver When You're Dead</td>
      <td>6.7</td>
    </tr>
    <tr>
      <th>4065</th>
      <td>Mi America</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1847</th>
      <td>GoodFellas</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>4217</th>
      <td>Kids</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>883</th>
      <td>Catch Me If You Can</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>3866</th>
      <td>City of God</td>
      <td>8.1</td>
    </tr>
  </tbody>
</table>
</div>



## 최근접 이웃 협업 필터링(아이템 기반)
[Movie lens Dataset](https://grouplens.org/datasets/movielens/latest/)  


```python
movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")

movies.shape, ratings.shape
```




    ((9742, 3), (100836, 4))




```python
ratings = ratings[["userId", "movieId", "rating"]]
ratings_matrix = ratings.pivot_table("rating", index="userId", columns="movieId")
ratings_matrix.head(3)
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
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>193565</th>
      <th>193567</th>
      <th>193571</th>
      <th>193573</th>
      <th>193579</th>
      <th>193581</th>
      <th>193583</th>
      <th>193585</th>
      <th>193587</th>
      <th>193609</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9724 columns</p>
</div>




```python
rating_movies = pd.merge(ratings, movies, on="movieId")

rating_matrix = rating_movies.pivot_table("rating", index="userId", columns="title")
rating_matrix.fillna(0, inplace=True)
rating_matrix.head(3)
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>



### 영화 간 유사도 산출


```python
ratings_matrix_T = rating_matrix.transpose()
ratings_matrix_T.head(3)
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
      <th>userId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>601</th>
      <th>602</th>
      <th>603</th>
      <th>604</th>
      <th>605</th>
      <th>606</th>
      <th>607</th>
      <th>608</th>
      <th>609</th>
      <th>610</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 610 columns</p>
</div>




```python
item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

item_sim_df = pd.DataFrame(data=item_sim, index=rating_matrix.columns, columns=rating_matrix.columns)
```


```python
item_sim_df.shape
```




    (9719, 9719)




```python
item_sim_df.head(3)
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.141653</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.342055</td>
      <td>0.543305</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.139431</td>
      <td>0.327327</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.707107</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>0.0</td>
      <td>0.707107</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.176777</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>




```python
# 유사한 상위 5개 영화
item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[:5]
```




    title
    Godfather, The (1972)                        1.000000
    Godfather: Part II, The (1974)               0.821773
    Goodfellas (1990)                            0.664841
    One Flew Over the Cuckoo's Nest (1975)       0.620536
    Star Wars: Episode IV - A New Hope (1977)    0.595317
    Name: Godfather, The (1972), dtype: float64



## 잠재 요인 협업 필터링 (행렬 분해)


```python
def matrix_factorization(R, K, steps=200, lr=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))
    
    prev_rmse = 10000
    break_count = 0
    
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j]>0]

    for step in range(steps):
        for i, j, r in non_zeros:
            eij = r - np.dot(P[i, :], Q[j, :].T)
            P[i, :] = P[i, :] + lr*(eij*Q[j, :] - r_lambda*P[i, :])
            Q[j, :] = Q[j, :] + lr*(eij*P[i, :] - r_lambda*Q[j, :])
        rmse = get_rmse(R, P, Q, non_zeros)
        if step%10==0:
            print("### iteration step: ", step, "rmse: ", rmse)
    return P, Q
```


```python
movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")

ratings = ratings[["userId", "movieId", "rating"]]
ratings_matrix = ratings.pivot_table("rating", index="userId", columns="movieId")
rating_movies = pd.merge(ratings, movies, on="movieId")
ratings_matrix = rating_movies.pivot_table("rating", index="userId", columns="title")
```


```python
P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200, lr=0.01, r_lambda=0.01)

pred_matrix = np.dot(P, Q.T)
```

    ### iteration step:  0 rmse:  2.93786091768772
    ### iteration step:  10 rmse:  0.7312209418714728
    ### iteration step:  20 rmse:  0.5100850016439248
    ### iteration step:  30 rmse:  0.3712333270639945
    ### iteration step:  40 rmse:  0.29399664330651887
    ### iteration step:  50 rmse:  0.24971866468792214
    ### iteration step:  60 rmse:  0.22262249831448883
    ### iteration step:  70 rmse:  0.2047490033063493
    ### iteration step:  80 rmse:  0.19216254497786234
    ### iteration step:  90 rmse:  0.18282943339050187
    ### iteration step:  100 rmse:  0.175627293491817
    ### iteration step:  110 rmse:  0.16989575037186763
    ### iteration step:  120 rmse:  0.16522354862893715
    ### iteration step:  130 rmse:  0.16134152805789648
    ### iteration step:  140 rmse:  0.15806580560470823
    ### iteration step:  150 rmse:  0.15526609841105868
    ### iteration step:  160 rmse:  0.15284723767625422
    ### iteration step:  170 rmse:  0.1507379084664755
    ### iteration step:  180 rmse:  0.14888351579956635
    ### iteration step:  190 rmse:  0.1472415023306992
    


```python
ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index=ratings_matrix.index, columns=rating_matrix.columns)

ratings_pred_matrix.head(3)
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
      <th>title</th>
      <th>'71 (2014)</th>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <th>'Round Midnight (1986)</th>
      <th>'Salem's Lot (2004)</th>
      <th>'Til There Was You (1997)</th>
      <th>'Tis the Season for Love (2015)</th>
      <th>'burbs, The (1989)</th>
      <th>'night Mother (1986)</th>
      <th>(500) Days of Summer (2009)</th>
      <th>*batteries not included (1987)</th>
      <th>...</th>
      <th>Zulu (2013)</th>
      <th>[REC] (2007)</th>
      <th>[REC]² (2009)</th>
      <th>[REC]³ 3 Génesis (2012)</th>
      <th>anohana: The Flower We Saw That Day - The Movie (2013)</th>
      <th>eXistenZ (1999)</th>
      <th>xXx (2002)</th>
      <th>xXx: State of the Union (2005)</th>
      <th>¡Three Amigos! (1986)</th>
      <th>À nous la liberté (Freedom for Us) (1931)</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.248835</td>
      <td>4.022864</td>
      <td>3.603369</td>
      <td>4.636573</td>
      <td>4.211315</td>
      <td>1.257738</td>
      <td>4.349091</td>
      <td>2.169613</td>
      <td>4.104452</td>
      <td>4.108343</td>
      <td>...</td>
      <td>1.274139</td>
      <td>4.171452</td>
      <td>3.704318</td>
      <td>2.869537</td>
      <td>2.584760</td>
      <td>4.726538</td>
      <td>3.122263</td>
      <td>2.114118</td>
      <td>3.894503</td>
      <td>0.906326</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.016236</td>
      <td>3.662808</td>
      <td>3.194755</td>
      <td>4.146299</td>
      <td>3.988296</td>
      <td>1.148734</td>
      <td>3.866164</td>
      <td>1.918793</td>
      <td>3.189712</td>
      <td>3.621542</td>
      <td>...</td>
      <td>1.012644</td>
      <td>3.941292</td>
      <td>3.227885</td>
      <td>2.555684</td>
      <td>2.289547</td>
      <td>4.200528</td>
      <td>1.586022</td>
      <td>1.667428</td>
      <td>4.242065</td>
      <td>0.815590</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.143055</td>
      <td>1.846009</td>
      <td>1.608932</td>
      <td>2.171609</td>
      <td>2.219850</td>
      <td>0.783532</td>
      <td>1.432963</td>
      <td>1.219582</td>
      <td>1.431676</td>
      <td>2.425546</td>
      <td>...</td>
      <td>0.706200</td>
      <td>2.383612</td>
      <td>2.264150</td>
      <td>1.840257</td>
      <td>1.603285</td>
      <td>2.972241</td>
      <td>1.532865</td>
      <td>1.094112</td>
      <td>2.960689</td>
      <td>0.471577</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 9719 columns</p>
</div>


