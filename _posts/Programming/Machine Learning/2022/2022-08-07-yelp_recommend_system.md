---
title: Yelp 데이터셋과 텍스트 유사도를 이용한 추천 시스템
date: 2022-08-07T10:36:16.047Z

categories:
  - Programming
  - Machine Learning
tags:
  - Pandas
  - Numpy
  - sklearn
---

# Yelp Recommender Systems
[참고 노트북 | Yelp Dataset: SurpriseMe Recommendation System](https://www.kaggle.com/code/fahd09/yelp-dataset-surpriseme-recommendation-system)
### 사용 라이브러리


```python
import os
import re
import string

import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
```

### Data Load


```python
df_yelp_business = pd.read_json('./data//yelp_academic_dataset_business.json', lines=True)
df_yelp_business.fillna('NA', inplace=True)
df_yelp_business = df_yelp_business[df_yelp_business['categories'].str.contains('Restaurants')]
print('Final Shape: ',df_yelp_business.shape)
```

    Final Shape:  (52268, 14)
    


```python
df_yelp_review_iter = pd.read_json("./data/yelp_academic_dataset_review.json", chunksize=100000, lines=True)

df_yelp_review = pd.DataFrame()
i=0
for df in df_yelp_review_iter:
    df = df[df['business_id'].isin(df_yelp_business['business_id'])]
    df_yelp_review = pd.concat([df_yelp_review, df])
    i=i+1
    print(i)
    if i==4: break
```

    1
    2
    3
    4
    


```python
df_yelp_business = df_yelp_business[df_yelp_business['business_id'].isin(df_yelp_review['business_id'])]

print('Final businesses shape: ', df_yelp_business.shape)
print('Final review shape: ', df_yelp_review.shape)
```

    Final businesses shape:  (4937, 14)
    Final review shape:  (283029, 9)
    

### Preprocessing


```python
def clean_text(text):
    ## 구두점 제거
    text = text.translate(string.punctuation)
    
    ## 소문자 변경 후 분리
    text = text.lower().split()
    
    ## 불용어 제거
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    return text
```


```python
%%time
df_yelp_review['text'] = df_yelp_review['text'].apply(clean_text)
```

    CPU times: total: 1min 9s
    Wall time: 1min 9s
    

### Top 100 Vocabularies


```python
vectorizer_reviews = CountVectorizer(min_df = .01,max_df = .99, tokenizer = WordPunctTokenizer().tokenize)
vectorized_reviews = vectorizer_reviews.fit_transform(df_yelp_review['text'])

vectorized_reviews.shape
```




    (283029, 886)




```python
' | '.join(vectorizer_reviews.get_feature_names_out()[:100])
```




    '! | + | - | 00 | 1 | 10 | 12 | 15 | 2 | 20 | 3 | 30 | 4 | 5 | 50 | 6 | 7 | 8 | : | ; | a | able | about | absolutely | accommodating | across | actually | add | added | addition | afternoon | again | ago | all | almost | along | already | also | although | always | am | amazing | ambiance | american | amount | and | another | anyone | anything | anyway | anywhere | appetizer | appetizers | are | area | around | arrived | as | ask | asked | ate | atmosphere | attention | attentive | authentic | available | average | avocado | away | awesome | awful | back | bacon | bad | baked | bar | bartender | based | basically | bbq | be | beans | beautiful | beef | beer | beers | before | behind | believe | best | better | beyond | big | bill | birthday | bit | bite | black | bland | blue'



### Top 100 Categoreis


```python
vectorizer_categories = CountVectorizer(min_df = 1, max_df = 1., tokenizer = lambda x: x.split(', '))
vectorized_categories = vectorizer_categories.fit_transform(df_yelp_business['categories'])

vectorized_categories.shape
```




    (4937, 387)




```python
' | '.join(vectorizer_categories.get_feature_names_out()[:100])
```




    "acai bowls | accessories | active life | adult entertainment | afghan | african | american (new) | american (traditional) | amusement parks | appliances & repair | arabic | arcades | argentine | armenian | art galleries | arts & crafts | arts & entertainment | asian fusion | austrian | auto detailing | auto glass services | auto repair | automotive | bagels | bakeries | banks & credit unions | bar crawl | barbeque | barbers | bars | bartenders | basque | battery stores | batting cages | beaches | beauty & spas | bed & breakfast | beer | beer bar | beer gardens | beer tours | beverage store | bistros | boat charters | boat tours | boating | body shops | books | bookstores | bowling | brasseries | brazilian | breakfast & brunch | breweries | brewpubs | british | bubble tea | buffets | building supplies | burgers | burmese | business consulting | butcher | cabaret | cafes | cafeteria | cajun/creole | calabrian | cambodian | canadian (new) | candy stores | cannabis dispensaries | cantonese | car dealers | car stereo installation | cards & stationery | caribbean | casinos | caterers | cheese shops | cheesesteaks | chicken shop | chicken wings | child care & day care | children's clothing | chinese | chiropractors | chocolatiers & shops | christmas trees | churches | cinema | club crawl | cocktail bars | coffee & tea | coffee roasteries | coffeeshops | colombian | comedy clubs | comfort food | community service/non-profit"



### 희소 행렬 생성


```python
%%time
from scipy import sparse
businessxreview = sparse.csr_matrix(pd.get_dummies(df_yelp_review['business_id']).values)
```

    CPU times: total: 14.6 s
    Wall time: 14.7 s
    


```python
print('restuarants x categories: \t', vectorized_categories.shape) 
print('restuarants x reviews: \t\t' , businessxreview.shape) 
print('reviews x words: \t\t', vectorized_reviews.shape)
```

    restuarants x categories: 	 (4937, 387)
    restuarants x reviews: 		 (283029, 4937)
    reviews x words: 		 (283029, 886)
    

### 리뷰와 평점이 좋은 다른 식당 추천


```python
df_yelp_business.sample(5)
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
      <th>business_id</th>
      <th>name</th>
      <th>address</th>
      <th>city</th>
      <th>state</th>
      <th>postal_code</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>stars</th>
      <th>review_count</th>
      <th>is_open</th>
      <th>attributes</th>
      <th>categories</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6273</th>
      <td>DptW6vZmrd7ttS0RCaWx2w</td>
      <td>Xwrecks Restaurant &amp; Lounge</td>
      <td>9303 50th Street NW</td>
      <td>Edmonton</td>
      <td>AB</td>
      <td>T6B 2L5</td>
      <td>53.530919</td>
      <td>-113.417837</td>
      <td>2.0</td>
      <td>7</td>
      <td>0</td>
      <td>{'Alcohol': 'u'full_bar'', 'RestaurantsPriceRa...</td>
      <td>Restaurants, Bars, Nightlife, American (Tradit...</td>
      <td>{'Monday': '11:0-0:0', 'Tuesday': '11:0-0:0', ...</td>
    </tr>
    <tr>
      <th>12352</th>
      <td>4w6Z5v0uVt08oSBaA3342A</td>
      <td>Wawa</td>
      <td>600 Cinnaminson Ave</td>
      <td>Palmyra</td>
      <td>NJ</td>
      <td>08065</td>
      <td>39.998409</td>
      <td>-75.035118</td>
      <td>3.5</td>
      <td>5</td>
      <td>1</td>
      <td>{'RestaurantsPriceRange2': '4', 'BusinessAccep...</td>
      <td>Convenience Stores, Automotive, Coffee &amp; Tea, ...</td>
      <td>{'Monday': '0:0-0:0', 'Tuesday': '0:0-0:0', 'W...</td>
    </tr>
    <tr>
      <th>3136</th>
      <td>N44roXfLNkBdpINQDjEFOQ</td>
      <td>Carisilo's Mexican Restaurant</td>
      <td>1978 Vandalia St</td>
      <td>Collinsville</td>
      <td>IL</td>
      <td>62234</td>
      <td>38.695337</td>
      <td>-89.966691</td>
      <td>4.0</td>
      <td>65</td>
      <td>1</td>
      <td>{'RestaurantsDelivery': 'False', 'Alcohol': ''...</td>
      <td>Mexican, Restaurants</td>
      <td>{'Monday': '11:0-21:0', 'Tuesday': '11:0-21:0'...</td>
    </tr>
    <tr>
      <th>9834</th>
      <td>-SFSt3FkjGfavnyMpHsZPA</td>
      <td>Enjoi Sweets &amp; Company</td>
      <td>4707 W Gandy Blvd, Ste 7</td>
      <td>Tampa</td>
      <td>FL</td>
      <td>33611</td>
      <td>27.893760</td>
      <td>-82.525167</td>
      <td>4.5</td>
      <td>9</td>
      <td>0</td>
      <td>{'NoiseLevel': 'u'quiet'', 'BusinessAcceptsBit...</td>
      <td>Desserts, Food, Cafes, Restaurants, Food Truck...</td>
      <td>{'Thursday': '12:0-21:0', 'Friday': '12:0-21:0...</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>jLaPtjlLfRSaoBWIcHcSQg</td>
      <td>The Mad Crab</td>
      <td>8080 Olive Blvd</td>
      <td>University City</td>
      <td>MO</td>
      <td>63130</td>
      <td>38.672734</td>
      <td>-90.345018</td>
      <td>3.5</td>
      <td>156</td>
      <td>1</td>
      <td>{'Caters': 'False', 'Alcohol': 'u'beer_and_win...</td>
      <td>Seafood, Cajun/Creole, American (New), Restaur...</td>
      <td>{'Monday': '15:0-22:0', 'Tuesday': '15:0-22:0'...</td>
    </tr>
  </tbody>
</table>
</div>




```python
business_choose = '-SFSt3FkjGfavnyMpHsZPA' # Desserts, Food, Cafes, Restaurants ...
```


```python
new_reviews = df_yelp_review.loc[df_yelp_review['business_id'] == business_choose, 'text']
print('\n'.join([r[:100] for r in new_reviews.tolist()]))
```

    wow probably best cupcakes i have since moved tampa + + i stopped guys came flicks food trucks heard
    pleasure experiencing enjoi sweets recent food truck rally work later day dessert truck best place e
    delicious cupcakes review say much liked place went tried red velvet chocolate chip brownie fresh yu
    one word delectable ! + + stumbled upon food truck which also storefront flicks food trucks past mon
    tried cupcakes food truck family ordered following : + + chocolate chocolate delicious moist cake ch
    unable contact month left facebook review told anything nice say keep myself understand things come 
    used enjoi sweets company event fantastic ! everything setting event food itself joi jon pleasure wo
    tried italian mango drink super delicious got get enough ! 
    enjoi sweets one favorite food trucks love design course delicious cupcakes catered events say serve
    


```python
new_categories = df_yelp_business.loc[df_yelp_business['business_id'] == business_choose, 'categories']
new_categories.tolist()
```




    ['Desserts, Food, Cafes, Restaurants, Food Trucks, American (Traditional)']



### 유사도 계산


```python
from scipy.spatial.distance import cdist
# find most similar reviews
dists1 = cdist(vectorizer_reviews.transform(new_reviews).todense().mean(axis=0), 
              vectorized_reviews.T.dot(businessxreview).T.todense(), 
               metric='correlation')
# find most similar categories
dists2 = cdist(vectorizer_categories.transform(new_categories).todense().mean(axis=0), 
              vectorized_categories.todense(), 
               metric='correlation')
```


```python
dists_together = np.vstack([dists1.ravel(), dists2.ravel()]).T

dists = dists_together.mean(axis=1)
dists
```




    array([0.54952985, 0.50191353, 0.56616524, ..., 0.69466944, 0.64917578,
           0.4334572 ])




```python
# 가장 유사한 10개의 레스토랑
closest = dists.argsort().ravel()[:10]
```

#### 기준 레스토랑


```python
df_yelp_business.loc[df_yelp_business['business_id']== business_choose, ['business_id', 'categories', 'name', 'stars']]
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
      <th>business_id</th>
      <th>categories</th>
      <th>name</th>
      <th>stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9834</th>
      <td>-SFSt3FkjGfavnyMpHsZPA</td>
      <td>Desserts, Food, Cafes, Restaurants, Food Truck...</td>
      <td>Enjoi Sweets &amp; Company</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>



#### 추천된 레스토랑 목록


```python
df_yelp_business.loc[df_yelp_business['business_id'].isin(df_yelp_business['business_id'].iloc[closest]), ['business_id', 'categories', 'name', 'stars']]
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
      <th>business_id</th>
      <th>categories</th>
      <th>name</th>
      <th>stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>742</th>
      <td>dD2p903p8lU0IgXT3OFluA</td>
      <td>Breakfast &amp; Brunch, Restaurants, Food, Cafes, ...</td>
      <td>Edgehill Cafe</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2548</th>
      <td>dcpWZ6Yk_S0HqTlNBi8jiA</td>
      <td>Food, Coffee &amp; Tea, Restaurants, Desserts, Cafes</td>
      <td>The Woodrack Cafe</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4710</th>
      <td>qLrTiIPDlnNX6FYTs29rmg</td>
      <td>Restaurants, American (Traditional)</td>
      <td>Buddy's Grill</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>6720</th>
      <td>jVdYRED2iztNaNCoTAhVMA</td>
      <td>Restaurants, Salad, Food, Desserts</td>
      <td>Have A Greener Day</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>8244</th>
      <td>iHTL6BPlaPK6xvOa5MIKaQ</td>
      <td>American (Traditional), Restaurants, Food, Ame...</td>
      <td>Essentially Fries</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9834</th>
      <td>-SFSt3FkjGfavnyMpHsZPA</td>
      <td>Desserts, Food, Cafes, Restaurants, Food Truck...</td>
      <td>Enjoi Sweets &amp; Company</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>10337</th>
      <td>hQcAPRwuYFPAbhbpeNPEgA</td>
      <td>Bakeries, American (Traditional), Food, Restau...</td>
      <td>Apple Farm Diner and Bakery</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>11701</th>
      <td>tYCok-NtWvg8_k7woeB83w</td>
      <td>Desserts, American (Traditional), Cafes, Resta...</td>
      <td>Grand Lux Cafe</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>11748</th>
      <td>newkruvn1rhEvueEc9y1Mw</td>
      <td>Food, Restaurants, Desserts, Ice Cream &amp; Froze...</td>
      <td>Moo Moo Milk Bar</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>12506</th>
      <td>9dW3CVyvnTXdkXg2AOyBfw</td>
      <td>Desserts, Coffee &amp; Tea, Cafes, Donuts, Food, S...</td>
      <td>Birds Nest Cafe</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>
</div>


