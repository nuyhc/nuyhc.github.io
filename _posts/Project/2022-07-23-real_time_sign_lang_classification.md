---
title: 영문 수어 실시간 분류 모델
date: 2022-07-22T15:21:12.450Z
img_path: /nuyhc.github.io/assets/img
categories:
  - Project
tags:
  - Tensorflow
  - Pandas
  - Seaborn
  - matplot
  - sklearn
  - Deep Learning
---

# 실시간 수어 분류 모델

## 한국 숫자 분류 모델
### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from PIL import Image
import pillow_heif
import cv2

import splitfolders

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report

import glob
import warnings
warnings.filterwarnings("ignore")
```

### Data Load


```python
train_img_path = glob.glob("archive/train/*/*")
test_img_path = glob.glob("archive/test/*/*")

len(train_img_path), len(test_img_path)
```




    (824, 626)



.heic 파일을 변환해줘야 함


```python
# for filename in test_img_path: # test_img_path는 이미지 저장한 경로
#     if filename.lower().endswith(".heic"): # 파일 경로를 모두 소문자로 바꾸고, .heic로 끝나는 경우(.heic 확장자인 경우)
#         heif_file = pillow_heif.read_heif(filename) # pillow_heif 라이브러리를 이용해 불러오고
#         img = Image.frombytes(
#             heif_file.mode,
#             heif_file.size,
#             heif_file.data,
#             "raw"
#         )
#         new_name = f"{filename.split('.')[0]}.png" # 경로를 그대로 유지하기 위해 만든 변수
#         print(new_name)
#         img.save(new_name, format="png") # png 확장자로 변환해 저장
#     else: # .heic 확장자가 아닌경우 (.jpg, .jpeg 등), 아무런 처리도 안하지만 진행 상황을 보기 위해 경로 그대로 출력
#         print(filename)
```

이미지 데이터를 변환하면서 확인해보니, 이상하게 찍힌 사진들이 존재해서 제거할 필요가 있어보임  
초기 데이터의 경우 train과 test가 각각 777, 330장이 있었음

### Data Frame


```python
train_img = pd.Series(train_img_path)
test_img = pd.Series(test_img_path)
```


```python
train_df = pd.DataFrame()
train_df["path"] = train_img.map(lambda x: x)
train_df["ClassId"] = train_img.map(lambda x: x.split("\\")[1])
train_df["FileName"] = train_img.map(lambda x: x.split("\\")[2])

test_df = pd.DataFrame()
test_df["path"] = test_img.map(lambda x: x)
test_df["ClassId"] = test_img.map(lambda x: x.split("\\")[1])
test_df["FileName"] = test_img.map(lambda x: x.split("\\")[2])
```

.heic 파일들을 제거해줘야 함


```python
train_df = train_df[~train_df["FileName"].str.contains(".HEIC|.heic")].reset_index()
test_df = test_df[~test_df["FileName"].str.contains(".HEIC|.heic")].reset_index()
```


```python
train_df.shape, test_df.shape
```




    ((777, 4), (330, 4))



초기 데이터와 동일한 개수의 이미지가 있음

### 데이터 분포 확인


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].pie(train_df["ClassId"].value_counts().sort_index().values, labels=train_df["ClassId"].value_counts().sort_index().index, autopct="%.2f%%")
ax[1].pie(test_df["ClassId"].value_counts().sort_index().values, labels=test_df["ClassId"].value_counts().sort_index().index, autopct="%.2f%%")
plt.show()
```


    
![png](proj_sign/kor_num_sign_files/kor_num_sign_15_0.png)
    


`train`과 `test` 모두 데이터가 고르게 잘 들어있음  
다만, 10을 나타내는 수어가 2 종류인 점을 감안하면, 10은 다른 숫자보다 약 2배 더 많음


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
sns.countplot(x=train_df["ClassId"], ax=ax[0]).set_title("Train ClassId Distribution")
sns.countplot(x=test_df["ClassId"], ax=ax[1]).set_title("Test ClassId Distribution")
plt.show()
```


    
![png](proj_sign/kor_num_sign_files/kor_num_sign_17_0.png)
    


데이터의 절대량은, `train`은 70개 전후, `test`는 30개 전후여서 데이터 증강이 필요할꺼라 생각됨

### 예시 이미지 출력


```python
def img_resize_to_gray(fpath):
    """파일 경로를 입력 받아 사이즈 조정과 그레이로 변환하는 함수

    Args:
        fpath (str): 파일 경로
    Returns:
        arr (np.array)
    """
    img = cv2.imread(fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (300, 300))
    return img
```


```python
plot_df = train_df.sample(25)
```


```python
fig, ax = plt.subplots(5, 5, figsize=(20, 15))
for idx, fpath in enumerate(plot_df["path"]):
    classid = fpath.split("\\")[1]
    plt.subplot(5, 5, idx+1)
    plt.imshow(img_resize_to_gray(fpath))
    plt.title(classid)
    plt.xticks([])
    plt.yticks([])
```


    
![png](proj_sign/kor_num_sign_files/kor_num_sign_22_0.png)
    


### Image Data Generator


```python
splitfolders.ratio(input="./archive/train/", output="./archive/kor_number", ratio=(0.9, 0.05, 0.05))
```

    Copying files: 824 files [00:04, 196.93 files/s]
    


```python
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory("./archive/kor_number/train/", target_size=(300, 300), batch_size=32, shuffle=True, class_mode='sparse')
test_generator = test_datagen.flow_from_directory("./archive/kor_number/test/", target_size=(300,300), batch_size=32, shuffle=False, class_mode='sparse')
val_generator = val_datagen.flow_from_directory("./archive/kor_number/val/", target_size=(300,300), batch_size=32, shuffle=False, class_mode='sparse')
```

    Found 689 images belonging to 11 classes.
    Found 54 images belonging to 11 classes.
    Found 34 images belonging to 11 classes.
    

### Model: Efficient Net


```python
from tensorflow.keras.applications import EfficientNetB0

model = EfficientNetB0(
    input_shape=(300, 300, 3),
    include_top=False,
    weights="imagenet"
)
```

### Fine Tuning


```python
model.trainable = True

for layer in model.layers[:-15]:
    layer.trainable = False
    
x = tf.keras.layers.Flatten()(model.output)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(11, activation="softmax")(x)

model = tf.keras.Model(model.input, x)

model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)
```

### Train


```python
early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", verbose=0, patience=10)
model_check = ModelCheckpoint("model_kor_num_no_augmentation.h5", monitor="val_accuracy", mode="max", save_best_only=True)
```


```python
with tf.device("/device:GPU:0"):
    history = model.fit(train_generator, validation_data=val_generator, epochs=50, verbose=1, callbacks=[early_stopping, model_check])
```

    Epoch 1/50
    22/22 [==============================] - 76s 2s/step - loss: 4.1347 - accuracy: 0.4731 - val_loss: 1.7526 - val_accuracy: 0.7353
    Epoch 2/50
    22/22 [==============================] - 75s 3s/step - loss: 0.8099 - accuracy: 0.8462 - val_loss: 2.8852 - val_accuracy: 0.7647
    Epoch 3/50
    22/22 [==============================] - 63s 3s/step - loss: 0.4242 - accuracy: 0.9303 - val_loss: 3.3324 - val_accuracy: 0.7941
    ...
    Epoch 22/50
    22/22 [==============================] - 116s 5s/step - loss: 0.3746 - accuracy: 0.9710 - val_loss: 9.3713 - val_accuracy: 0.8235
    Epoch 23/50
    22/22 [==============================] - 89s 4s/step - loss: 0.7543 - accuracy: 0.9550 - val_loss: 10.8085 - val_accuracy: 0.7353
    


```python
hist_df = pd.DataFrame(history.history)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
hist_df[["accuracy", "val_accuracy"]].plot(ax=ax[0])
hist_df[["loss", "val_loss"]].plot(ax=ax[1])
plt.show()
```


    
![png](proj_sign/kor_num_sign_files/kor_num_sign_33_0.png)
    


### Model Evaluation


```python
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
print('Loss: %.3f' % (test_loss * 100.0))
print('Accuracy: %.3f' % (test_acc * 100.0)) 
```

    2/2 [==============================] - 6s 2s/step - loss: 8.0187 - accuracy: 0.8148
    Loss: 801.866
    Accuracy: 81.481
    


```python
y_val = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred,axis=1)

print(classification_report(y_val, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      0.80      0.89         5
               1       1.00      0.75      0.86         4
               2       0.83      1.00      0.91         5
               3       0.50      1.00      0.67         5
               4       0.50      0.20      0.29         5
               5       0.80      0.80      0.80         5
               6       1.00      0.60      0.75         5
               7       0.71      1.00      0.83         5
               8       1.00      0.80      0.89         5
               9       1.00      1.00      1.00         5
              10       1.00      1.00      1.00         5
    
        accuracy                           0.81        54
       macro avg       0.85      0.81      0.81        54
    weighted avg       0.85      0.81      0.81        54
    

## 알파벳 수어 분류 모델
### Sign Language MNIST
z와 j는 유사한 수어가 존재하고 움직이는 방식으로 구분하는거 같음

### 사용 라이브러리


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, ImageDraw
import tkinter as tk
import operator
import os
import glob

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Data Load


```python
train = pd.read_csv("./eng_sign/sign_mnist_train.csv")
test = pd.read_csv("./eng_sign/sign_mnist_test.csv")

train.shape, test.shape
```




    ((27455, 785), (7172, 785))




```python
display(train.sample(3))
display(test.sample(3))
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
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2227</th>
      <td>5</td>
      <td>143</td>
      <td>147</td>
      <td>153</td>
      <td>159</td>
      <td>163</td>
      <td>166</td>
      <td>170</td>
      <td>171</td>
      <td>173</td>
      <td>...</td>
      <td>107</td>
      <td>99</td>
      <td>100</td>
      <td>100</td>
      <td>99</td>
      <td>96</td>
      <td>112</td>
      <td>206</td>
      <td>221</td>
      <td>205</td>
    </tr>
    <tr>
      <th>15930</th>
      <td>4</td>
      <td>160</td>
      <td>160</td>
      <td>159</td>
      <td>161</td>
      <td>161</td>
      <td>161</td>
      <td>160</td>
      <td>159</td>
      <td>159</td>
      <td>...</td>
      <td>185</td>
      <td>188</td>
      <td>188</td>
      <td>185</td>
      <td>184</td>
      <td>184</td>
      <td>182</td>
      <td>180</td>
      <td>179</td>
      <td>177</td>
    </tr>
    <tr>
      <th>4648</th>
      <td>13</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>222</td>
      <td>223</td>
      <td>224</td>
      <td>225</td>
      <td>224</td>
      <td>224</td>
      <td>...</td>
      <td>164</td>
      <td>197</td>
      <td>132</td>
      <td>111</td>
      <td>86</td>
      <td>231</td>
      <td>255</td>
      <td>232</td>
      <td>253</td>
      <td>255</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 785 columns</p>
</div>



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
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3840</th>
      <td>24</td>
      <td>157</td>
      <td>159</td>
      <td>162</td>
      <td>162</td>
      <td>163</td>
      <td>164</td>
      <td>165</td>
      <td>165</td>
      <td>166</td>
      <td>...</td>
      <td>193</td>
      <td>192</td>
      <td>189</td>
      <td>189</td>
      <td>188</td>
      <td>187</td>
      <td>186</td>
      <td>185</td>
      <td>183</td>
      <td>181</td>
    </tr>
    <tr>
      <th>2611</th>
      <td>4</td>
      <td>179</td>
      <td>181</td>
      <td>181</td>
      <td>182</td>
      <td>184</td>
      <td>185</td>
      <td>187</td>
      <td>188</td>
      <td>187</td>
      <td>...</td>
      <td>93</td>
      <td>75</td>
      <td>124</td>
      <td>214</td>
      <td>218</td>
      <td>218</td>
      <td>218</td>
      <td>217</td>
      <td>216</td>
      <td>216</td>
    </tr>
    <tr>
      <th>942</th>
      <td>12</td>
      <td>96</td>
      <td>101</td>
      <td>108</td>
      <td>116</td>
      <td>134</td>
      <td>150</td>
      <td>159</td>
      <td>167</td>
      <td>174</td>
      <td>...</td>
      <td>115</td>
      <td>94</td>
      <td>217</td>
      <td>250</td>
      <td>244</td>
      <td>248</td>
      <td>249</td>
      <td>250</td>
      <td>251</td>
      <td>251</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 785 columns</p>
</div>


### Label


```python
labels = train["label"].values
unique_val = np.array(labels)
```


```python
np.unique(unique_val)
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24], dtype=int64)



J와 Z는 해당 데이터셋에 존재하지 않음

### Data Distribution


```python
_ = sns.countplot(x=labels)
```


    
![png](proj_sign/eng_real_time_files/eng_real_time_11_0.png)
    


### Data Preprocessing


```python
train.drop(columns="label", axis=1, inplace=True)
```


```python
imgs = train.values
imgs = np.array([np.reshape(i, (28, 28)) for i in imgs]) # 28 by 28로 사이즈 변경
imgs = np.array([i.flatten() for i in imgs])
```


```python
# 라벨 인코딩
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
```

### Display Images (Data)


```python
plt.figure(figsize=(12, 12))
for idx in range(25):
    plt.subplot(5, 5, idx+1)
    plt.imshow(imgs[idx].reshape(28, 28))
```


    
![png](proj_sign/eng_real_time_files/eng_real_time_17_0.png)
    



```python
# # openCV로 확인
# for idx in range(25):
#     rand = np.random.randint(0, len(imgs))
#     sample_img = imgs[rand]
#     sample_img = sample_img.reshape(28, 28).astype(np.uint8)
#     sample_img = cv2.resize(sample_img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
#     cv2.imshow("Sample", sample_img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
```

### Data Split


```python
x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2)

print(f"x_train: {x_train.shape}\ny_train: {y_train.shape}\nx_test: {x_test.shape}\ny_test: {y_test.shape}")
```

    x_train: (21964, 784)
    y_train: (21964, 24)
    x_test: (5491, 784)
    y_test: (5491, 24)
    


```python
x_train = x_train/255
x_test = x_test/255
```

### Data Augmentation

### Modeling


```python
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
```


```python
# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         # rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         # horizontal_flip=False,  # randomly flip images
#         # vertical_flip=False  # randomly flip images
# )


# datagen.fit(x_train)
```

데이터 증강시 성능은 좋아지지만, 실시간 분류 능력이 떨어지는데,  
증강시 사용되는 회전, 반전 등이 영향이 큰 것 같음  

```python
model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
```

전이 학습을 시도했으나, 이미지 사이즈가 작아 전이 학습이 불가능해  직접 층을 쌓았다..

```python
model.compile(
    optimizer="adam",
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)
```


```python
# model.summary()
```


```python
early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", verbose=0, patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
```


```python
# with tf.device("/device:GPU:0"):
#     history = model.fit(datagen.flow(x_train, y_train, batch_size=200), epochs=50, validation_data=(x_test, y_test), callbacks=[early_stopping, learning_rate_reduction], verbose=1)
```

데이터 증강을 한 경우 위와 같은 코드를 이용했다.

```python
with tf.device("/device:GPU:0"):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128, verbose=1, callbacks=[early_stopping])
```

    Epoch 1/50
    172/172 [==============================] - 6s 16ms/step - loss: 1.8222 - accuracy: 0.4408 - val_loss: 0.4805 - val_accuracy: 0.8428
    Epoch 2/50
    172/172 [==============================] - 5s 28ms/step - loss: 0.2936 - accuracy: 0.9027 - val_loss: 0.0825 - val_accuracy: 0.9831
    Epoch 3/50
    172/172 [==============================] - 4s 26ms/step - loss: 0.0820 - accuracy: 0.9760 - val_loss: 0.0258 - val_accuracy: 0.9942
    ...
    Epoch 15/50
    172/172 [==============================] - 4s 22ms/step - loss: 0.0042 - accuracy: 0.9988 - val_loss: 8.6977e-04 - val_accuracy: 0.9998
    Epoch 16/50
    172/172 [==============================] - 4s 24ms/step - loss: 7.8023e-04 - accuracy: 0.9999 - val_loss: 6.1924e-05 - val_accuracy: 1.0000
    


```python
model.save("eng_sign_lang_cnn_model.h5")
```


```python
df_hist = pd.DataFrame(history.history)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
_ = df_hist[["loss", "val_loss"]].plot(ax=ax[0])
_ = df_hist[["accuracy", "val_accuracy"]].plot(ax=ax[1])
```


    
![png](proj_sign/eng_real_time_files/eng_real_time_33_0.png)
    


### Evaluation


```python
test_labels = test["label"]
test.drop(columns="label", axis=1, inplace=True)

test_img = test.values
test_img = np.array([np.reshape(i, (28, 28)) for i in test_img])
test_img = np.array([i.flatten() for i in test_img])

test_labels = label_binrizer.transform(test_labels)

test_img = test_img.reshape(test_img.shape[0], 28, 28, 1)

y_pred = model.predict(test_img)
```


```python
accuracy_score(test_labels, y_pred.round())
```




    0.9213608477412158



### Matching func.


```python
alpha = [chr(x).upper() for x in range(97, 123)]
alpha.remove("J")
alpha.remove("Z")
idx = [x for x in range(0, 24)]
```


```python
def convert_letter(result):
    classLabels = {idx:c for idx, c in zip(idx, alpha)}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "err"
```


### Test on Real Time


```python
model = tf.keras.models.load_model("./eng_sign_lang_cnn_model.h5")
```


```python
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    roi = frame[100:400, 320:620]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # roi = cv2.GaussianBlur(roi, (0, 0), 1)
    roi = cv2.Sobel(roi, -1, 1, 0, delta=128)
    # roi = cv2.Canny(roi, 60, 90)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    # cv2.imshow("roi", roi)
    roi = roi.reshape(1, 28, 28, 1)
    
    result = convert_letter(np.argmax(model.predict(roi)))
    # result = str(model.predict_classes(roi, 1, verbose=0)[0])
    
    cv2.putText(frame, result, (300, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
    cv2.rectangle(frame, pt1=(320, 100), pt2=(620, 400), color=(255, 0, 0), thickness=3)
    # frame = cv2.Canny(frame, 60, 90)
    # frame = cv2.GaussianBlur(frame, (0, 0), 1)
    frame = cv2.Sobel(frame, -1, 1, 0, delta=128)
    cv2.imshow("Sign Translator", frame)
    
    if cv2.waitKey(1)==ord("q"): break
    
cap.release()
cv2.destroyAllWindows()
```
roi 영역에서 엣지 검출 등 다양한 방식을 시도해봤는데, 인식률이 매우 낮고 배경의 영향을 많이 탄다는 문제점이 있었다. 이런 문제를 해결하기 위해 결국 검출 모델을 이용하고 관절을 키 포인트로 학습 시켜 접근했다.

## 알파벳 수어 분류 모델 - 디텍션 이용
```python
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
```

### 데이터 수집


```python
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize =300

folder = "Data/Y"
counter = 0

while True:
    try:
        ret, img = cap.read()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]
            
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 # 저장되는 창
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] # 유동적인 창
            
            # 메인 창
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
            
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImgWhite", imgWhite)
            
        cv2.imshow("Image", img)
        
        k = cv2.waitKey(1)
        if k==ord("s"): # s를 누르면 이미지 저장
            counter += 1
            cv2.imwrite(f"./{folder}/Image_{time.time()}.jpg", imgWhite)
            print(counter)
        if k==ord("q"): # q를 누르면 프로그램 종료
            break
    except: # 경계선 밖으로 나가면 충돌 남
        break
    
cap.release()
cv2.destroyAllWindows()
```
관절이 찍혀있는데 데이터 셋을 찾지 못해서 직접 데이터를 수집했다.  
![png](proj_sign/b.png)  
![png](proj_sign/f.png)  
![png](proj_sign/n.png)

위와 같이 각 레이블에 대해서 400~500장 정도 수집하고 구글의 Teachable Machine을 이용해 모델을 생성했다.

### Real Time


```python
try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
# hand detection    
detector = HandDetector(maxHands=1)
classifier = Classifier("./model/keras_model.h5", "./model/labels.txt")
 
offset = 20
imgSize = 300
 
labels = [chr(x).upper() for x in range(97, 123)]
labels.remove("J")
labels.remove("Z")
 
while True:
    try:
        ret, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            x, y, w, h = hands[0]['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                # print(prediction, index)
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
    
            cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 139), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 139), 4)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
        if cv2.waitKey(1)==ord("q"): break
    except:
        print("카메라가 경계선 밖으로 나갔습니다.")
        break
    cv2.imshow("Sign Detectoin", imgOutput)
    
cap.release()
cv2.destroyAllWindows
```