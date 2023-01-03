---
title: DACON 음향 데이터 COVID-19 검출 AI 경진대회 - DNN 모델
date: 2022-07-29T05:53:31.346Z

categories:
  - Deep Learning
tags:
  - DataScience
  - pandas
  - numpy
  - matplotlib
  - tensorflow
---

# DACON 음향 데이터 COVID-19 검출 AI 경진대회
[참고 | [Private 6위, 0.60553] DNN 코드 공유](https://dacon.io/competitions/official/235910/codeshare/5484?page=1&dtype=recent)

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import os
import librosa

import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
```

### Data Load


```python
base_path = "./open/"
train = pd.read_csv(base_path+"train_data.csv")
test = pd.read_csv(base_path+"test_data.csv")

train.shape, test.shape
```




    ((3805, 6), (5732, 5))



### Pre-Processing


```python
features = []
log_specgrams_hp = []
feature_delta = [] # feature 3
melspectrogram_list = [] # feature 1
harmonic_percussive_list = [] # feature 2

for uid in tqdm(train["id"]):
    root_path = os.path.join(base_path, "train")
    path = os.path.join(root_path, str(uid).zfill(5)+".wav")
    
    # wav 파일 load
    y, sr = librosa.load(path, sr=16000)
    
    # melspectrogram 추출
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=32)
    # 로그스케일링 된 melspectrogram의 델타값
    delta = []
    for e in melspectrogram:
        delta.append(np.mean(librosa.feature.delta(e)))
    feature_delta.append(delta)
    # 로그 스케일링
    feature_log_scale = librosa.power_to_db(S=melspectrogram, ref=1.0)
    # feature 1: 추출된 melspectrogram드르이 평균
    temp = []
    for e in feature_log_scale:
        temp.append(np.mean(e))
    melspectrogram_list.append(temp)
    
    # feature 2: (운율적 소리(harmonic)+두드리는 소리(percussive)의 구성 요소) 평균
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    melspectrogram_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels=32)
    melspectrogram_percussive = librosa.feature.melspectrogram(y_percussive, n_mels=32)
    log_harmonic = librosa.amplitude_to_db(melspectrogram_harmonic)
    log_percussive = librosa.amplitude_to_db(melspectrogram_percussive)
    log_hp = np.average([log_harmonic, log_percussive], axis=0)
    
    temp = []
    for e in log_hp:
        temp.append(np.mean(e))
    harmonic_percussive_list.append(temp)    
```


```python
# 오디오 Feature 추가
f_list1 = pd.DataFrame(melspectrogram_list)
f_list2 = pd.DataFrame(harmonic_percussive_list)
f_list3 = pd.DataFrame(feature_delta)
```


```python
f_list1.columns = ["melspectrogram_1_"+str(x) for x in range(1, 33)]
f_list2.columns = ["melspectrogram_2_"+str(x) for x in range(1, 33)]
f_list3.columns = ["melspectrogram_3_"+str(x) for x in range(1, 33)]
```


```python
melspectrogram_train_df = pd.concat([f_list1,f_list2,f_list3], axis=1)
train_df = pd.concat([train, melspectrogram_train_df], axis=1)
```


```python
train_df.head(5)
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
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>covid19</th>
      <th>melspectrogram_1_1</th>
      <th>melspectrogram_1_2</th>
      <th>melspectrogram_1_3</th>
      <th>melspectrogram_1_4</th>
      <th>...</th>
      <th>melspectrogram_3_23</th>
      <th>melspectrogram_3_24</th>
      <th>melspectrogram_3_25</th>
      <th>melspectrogram_3_26</th>
      <th>melspectrogram_3_27</th>
      <th>melspectrogram_3_28</th>
      <th>melspectrogram_3_29</th>
      <th>melspectrogram_3_30</th>
      <th>melspectrogram_3_31</th>
      <th>melspectrogram_3_32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>-31.943413</td>
      <td>-27.335651</td>
      <td>-23.652283</td>
      <td>-20.332117</td>
      <td>...</td>
      <td>1.551721e-05</td>
      <td>5.001302e-05</td>
      <td>7.300529e-05</td>
      <td>6.368942e-05</td>
      <td>8.467526e-05</td>
      <td>2.460767e-05</td>
      <td>1.935706e-05</td>
      <td>3.276046e-05</td>
      <td>1.080332e-05</td>
      <td>9.988543e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>51</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-18.772213</td>
      <td>-17.970600</td>
      <td>-19.152706</td>
      <td>-20.261648</td>
      <td>...</td>
      <td>4.808880e-07</td>
      <td>3.241044e-07</td>
      <td>2.775657e-07</td>
      <td>2.403397e-07</td>
      <td>1.801275e-07</td>
      <td>1.468669e-07</td>
      <td>1.040892e-07</td>
      <td>3.793937e-08</td>
      <td>2.792552e-08</td>
      <td>-5.082057e-09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>22</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-41.755840</td>
      <td>-37.208641</td>
      <td>-36.034973</td>
      <td>-36.101788</td>
      <td>...</td>
      <td>-2.077132e-08</td>
      <td>-1.249289e-08</td>
      <td>-1.005954e-08</td>
      <td>-6.208817e-09</td>
      <td>-1.373466e-09</td>
      <td>-1.762301e-09</td>
      <td>-2.323603e-09</td>
      <td>-1.048914e-09</td>
      <td>-2.308708e-10</td>
      <td>-4.273706e-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>29</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-36.795872</td>
      <td>-28.668598</td>
      <td>-27.015005</td>
      <td>-27.315031</td>
      <td>...</td>
      <td>-8.004472e-08</td>
      <td>-6.886885e-08</td>
      <td>-4.426233e-08</td>
      <td>-2.452074e-08</td>
      <td>-3.148728e-08</td>
      <td>-4.078213e-08</td>
      <td>-1.627670e-08</td>
      <td>-1.823432e-08</td>
      <td>-7.824335e-09</td>
      <td>-9.221319e-10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-50.650784</td>
      <td>-46.894409</td>
      <td>-47.190170</td>
      <td>-48.408485</td>
      <td>...</td>
      <td>-5.674101e-10</td>
      <td>-8.946563e-10</td>
      <td>5.834516e-10</td>
      <td>-1.668314e-10</td>
      <td>1.389479e-09</td>
      <td>-1.114242e-09</td>
      <td>2.274453e-11</td>
      <td>3.102881e-10</td>
      <td>2.110314e-10</td>
      <td>8.379827e-11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 102 columns</p>
</div>




```python
# test
features = []
log_specgrams_hp = []
feature_delta = [] # feature 3
melspectrogram_list = [] # feature 1
harmonic_percussive_list = [] # feature 2

for uid in tqdm(test["id"]):
    root_path = os.path.join(base_path, "test")
    path = os.path.join(root_path, str(uid).zfill(5)+".wav")
    
    # wav 파일 load
    y, sr = librosa.load(path, sr=16000)
    
    # melspectrogram 추출
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=32)
    # 로그스케일링 된 melspectrogram의 델타값
    delta = []
    for e in melspectrogram:
        delta.append(np.mean(librosa.feature.delta(e)))
    feature_delta.append(delta)
    # 로그 스케일링
    feature_log_scale = librosa.power_to_db(S=melspectrogram, ref=1.0)
    # feature 1: 추출된 melspectrogram드르이 평균
    temp = []
    for e in feature_log_scale:
        temp.append(np.mean(e))
    melspectrogram_list.append(temp)
    
    # feature 2: (운율적 소리(harmonic)+두드리는 소리(percussive)의 구성 요소) 평균
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    melspectrogram_harmonic = librosa.feature.melspectrogram(y_harmonic, n_mels=32)
    melspectrogram_percussive = librosa.feature.melspectrogram(y_percussive, n_mels=32)
    log_harmonic = librosa.amplitude_to_db(melspectrogram_harmonic)
    log_percussive = librosa.amplitude_to_db(melspectrogram_percussive)
    log_hp = np.average([log_harmonic, log_percussive], axis=0)
    
    temp = []
    for e in log_hp:
        temp.append(np.mean(e))
    harmonic_percussive_list.append(temp)    
```


```python
# 오디오 Feature 추가
f_list1 = pd.DataFrame(melspectrogram_list)
f_list2 = pd.DataFrame(harmonic_percussive_list)
f_list3 = pd.DataFrame(feature_delta)
```


```python
f_list1.columns = ["melspectrogram_1_"+str(x) for x in range(1, 33)]
f_list2.columns = ["melspectrogram_2_"+str(x) for x in range(1, 33)]
f_list3.columns = ["melspectrogram_3_"+str(x) for x in range(1, 33)]
```


```python
melspectrogram_test_df = pd.concat([f_list1,f_list2,f_list3], axis=1)
test_df = pd.concat([test, melspectrogram_test_df], axis=1)
```


```python
test_df.head()
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
      <th>age</th>
      <th>gender</th>
      <th>respiratory_condition</th>
      <th>fever_or_muscle_pain</th>
      <th>melspectrogram_1_1</th>
      <th>melspectrogram_1_2</th>
      <th>melspectrogram_1_3</th>
      <th>melspectrogram_1_4</th>
      <th>melspectrogram_1_5</th>
      <th>...</th>
      <th>melspectrogram_3_23</th>
      <th>melspectrogram_3_24</th>
      <th>melspectrogram_3_25</th>
      <th>melspectrogram_3_26</th>
      <th>melspectrogram_3_27</th>
      <th>melspectrogram_3_28</th>
      <th>melspectrogram_3_29</th>
      <th>melspectrogram_3_30</th>
      <th>melspectrogram_3_31</th>
      <th>melspectrogram_3_32</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3806</td>
      <td>48</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>-55.853134</td>
      <td>-55.158051</td>
      <td>-54.891762</td>
      <td>-54.874367</td>
      <td>-54.994320</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>3.043538e-12</td>
      <td>7.608845e-13</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-3.043538e-12</td>
      <td>-7.608845e-13</td>
      <td>-7.608845e-13</td>
      <td>3.091093e-13</td>
      <td>-5.944410e-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3807</td>
      <td>24</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>-46.948128</td>
      <td>-46.494526</td>
      <td>-46.730667</td>
      <td>-46.501385</td>
      <td>-46.066654</td>
      <td>...</td>
      <td>4.068705e-10</td>
      <td>-3.715323e-10</td>
      <td>-7.672049e-10</td>
      <td>-1.703404e-10</td>
      <td>1.796148e-09</td>
      <td>2.624169e-10</td>
      <td>1.179117e-10</td>
      <td>-4.361051e-11</td>
      <td>1.998812e-11</td>
      <td>-1.802107e-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3808</td>
      <td>29</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>-99.176407</td>
      <td>-93.005913</td>
      <td>-91.820778</td>
      <td>-92.210541</td>
      <td>-93.268394</td>
      <td>...</td>
      <td>-4.520774e-13</td>
      <td>-3.750947e-13</td>
      <td>-7.825289e-13</td>
      <td>-1.284776e-12</td>
      <td>-6.738704e-13</td>
      <td>-9.572461e-13</td>
      <td>-1.115856e-12</td>
      <td>-6.283318e-13</td>
      <td>-1.451943e-12</td>
      <td>-3.377899e-13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3809</td>
      <td>39</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>-33.959797</td>
      <td>-27.807213</td>
      <td>-26.108114</td>
      <td>-28.486483</td>
      <td>-29.118986</td>
      <td>...</td>
      <td>-9.174222e-10</td>
      <td>2.168453e-09</td>
      <td>-4.170101e-10</td>
      <td>5.560135e-11</td>
      <td>-6.950168e-12</td>
      <td>-5.125749e-11</td>
      <td>-4.969370e-10</td>
      <td>2.293556e-10</td>
      <td>1.390034e-10</td>
      <td>4.734802e-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3810</td>
      <td>34</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>-33.321808</td>
      <td>-29.476269</td>
      <td>-30.429527</td>
      <td>-30.836193</td>
      <td>-31.185854</td>
      <td>...</td>
      <td>-5.356627e-10</td>
      <td>7.791457e-10</td>
      <td>3.895728e-10</td>
      <td>0.000000e+00</td>
      <td>1.217415e-10</td>
      <td>8.521906e-11</td>
      <td>6.087076e-12</td>
      <td>0.000000e+00</td>
      <td>-2.434830e-11</td>
      <td>-1.217415e-11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 101 columns</p>
</div>



### Train


```python
x_train = train_df.drop(columns=["id", "covid19"])
y_train = train_df["covid19"]
```


```python
x_train = pd.get_dummies(x_train, columns=["gender"], drop_first=False)
```


```python
model = Sequential(name="Covid19_DNN")
model.add(Dense(512, activation="relu", input_shape=(x_train.shape[0], x_train.shape[1])))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation="relu"))
model.add(Dense(2, activation="sigmoid"))
```


```python
model.summary()
```

    Model: "Covid19_DNN"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_13 (Dense)             (None, 3805, 512)         52736     
    _________________________________________________________________
    dense_14 (Dense)             (None, 3805, 256)         131328    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 3805, 256)         0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 3805, 128)         32896     
    _________________________________________________________________
    dense_16 (Dense)             (None, 3805, 64)          8256      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 3805, 64)          0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 3805, 10)          650       
    _________________________________________________________________
    dense_18 (Dense)             (None, 3805, 2)           22        
    =================================================================
    Total params: 225,888
    Trainable params: 225,888
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)
```


```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
```


```python
with tf.device("/device:GPU:0"):
    histroy = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=64, callbacks=[early_stop])
```

    Epoch 1/100
    48/48 [==============================] - 2s 9ms/step - loss: 0.7508 - accuracy: 0.8811 - val_loss: 0.3203 - val_accuracy: 0.9080
    Epoch 2/100
    48/48 [==============================] - 0s 6ms/step - loss: 0.3339 - accuracy: 0.9149 - val_loss: 0.3312 - val_accuracy: 0.9080
    Epoch 3/100
    48/48 [==============================] - 0s 6ms/step - loss: 0.3159 - accuracy: 0.9215 - val_loss: 0.3192 - val_accuracy: 0.9080
    Epoch 4/100
    48/48 [==============================] - 0s 6ms/step - loss: 0.
    ...
    Epoch 66/100
    48/48 [==============================] - 0s 5ms/step - loss: 0.2662 - accuracy: 0.9231 - val_loss: 0.2988 - val_accuracy: 0.9080
    

### Evalutate


```python
test_loss, test_acc = model.evaluate(x_train, y_train)
```

    119/119 [==============================] - 0s 3ms/step - loss: 0.2735 - accuracy: 0.9198
    


```python
hist_df = pd.DataFrame(histroy.history)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
hist_df[["accuracy", "val_accuracy"]].plot(ax=ax[0])
hist_df[["loss", "val_loss"]].plot(ax=ax[1])
plt.show()
```


    
![png](/assets\images\sourceImg\covid19_DNN_sound_data_files\covid19_DNN_sound_data_25_0.png)
    

### Submission


```python
x_test = test_df.drop(columns=["id"])
x_test = pd.get_dummies(x_test, columns=["gender"], drop_first=False)
```


```python
pred = model.predict(x_test)
```
    


```python
y_pred = pd.DataFrame(np.where((pred>0.84), 0, 1))
y_pred = pd.DataFrame(y_pred.iloc[:, 0])
```


```python
sub = pd.read_csv("./open/sample_submission.csv")
sub["covid19"] = y_pred
sub.to_csv("sub_dnn.csv", index=False)
```
### 마무리
음향 데이터를 사용해 볼 목적으로 비슷한 경진대회를 필사해봤다.  
처음 다뤄보는 데이터라 막막했지만, 막상 진행해보니 음향 데이터에서 피처를 뽑는 부분만 새롭고 이후 부분들은 일반적인 방식과 동일했다.
음향 데이터를 위해 사용한 `librosa` 라이브러리와 DSP 같은 이론을 좀 더 알면 음향 데이터도 쉽게 사용할 수 있을꺼 같다.
DACON 제출시 정확도 50%로 170등대에 랭크되었다. 참고한 노트북에서는 `softmax`를 출력층 활성 함수로 사용했는데, 왜 `softmax`를 사용한지 모르겠다. 확진 여부를 분리하는 모델이므로 나는 `sigmoid`를 사용했는데 성능 차이가 이 부분에서 발생한거 같다.  
`softmax`로 활성 함수를 변경하면 58% 정확도로 59등까지 등수가 올라간다. (22-07-29일 기준)  
validation에서는 크게 성능 차이가 안나는데 제출시에는 성능 차이가 많이 발생한다.  
![png](/assets\images\sourceImg\covid19_DNN_sound_data_files\covid19_DNN_sound_data_25_1.png)  
++ 분류 문제가 아니라 회귀 문제라 끝단에 `softmax`를 사용한거 같다.