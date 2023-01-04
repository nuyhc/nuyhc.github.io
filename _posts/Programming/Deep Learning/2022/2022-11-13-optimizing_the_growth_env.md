---
title: DACON 생육 환경 최적화 경진대회
date: 2022-11-13T11:36:53.086Z

categories:
  - Deep Learning
tags:
  - Pandas
  - Numpy
  - matplot
  - Tensorflow
  - sklearn
  - PyTorch
  - CatBoost
---

# [DACON] 생육 환경 최적화 경진대회
[참고 | CNN+CatBoost+ANN](https://dacon.io/competitions/official/235897/codeshare/5017?page=1&dtype=recent)  
- DL과 ML을 섞어 사용했다는 점이 흥미로웠음

### Abstraction
- CNN: 이미지 밝기 조절 -> 마스킹(HSV) -> 픽셀 비율 추출 -> 이상치 처리 -> 학습
- CatBoost, ANN : 이미지 픽셀 비율 Feature 변수로 포함 -> 전처리 -> 학습

### Library


```python
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby
import random

import seaborn as sns

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

import matplotlib.pyplot as plt
import koreanize_matplotlib
```


```python
main_path = "./open"
train_imgs = glob(main_path + "/train/*/*/*.png") + glob(main_path + "/train/*/*/*.jpg")
train_imgs = sorted(train_imgs)
test_imgs = glob(main_path + "/test/image/*.png") + glob(main_path + "/test/image/*.jpg")
test_imgs = sorted(test_imgs)
train_data = glob(main_path + "/train/*/meta/*.csv")
train_data = sorted(train_data)
train_label = glob(main_path + "/train/*/*.csv")
train_label = sorted(train_label)
test_data = glob(main_path + "/test/meta/*.csv")
test_data = sorted(test_data)

preprocessing_train_imgs = main_path + "/PREPROCESSING-TRAIN"
preprocessing_test_imgs = main_path + "/PREPROCESSING-TEST"

if not os.path.exists(preprocessing_train_imgs):
    os.mkdir(preprocessing_train_imgs)
if not os.path.exists(preprocessing_test_imgs):
    os.mkdir(preprocessing_test_imgs)
```

### Image Augmentation
Grayscale 히스토그램을 이용해 모든 이미지 밝기 자동 조절


```python
def automatic_brightness_and_contrast(img, clip_hist_pct=0.025):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate grayscale hist
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    # Calculate cumulative distribution from the hist
    accumulator = []
    accumulator.append(float(hist[0]))
    for idx in range(1, hist_size):
        accumulator.append(accumulator[idx-1]+float(hist[idx]))
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_pct *= (maximum/100.0)
    clip_hist_pct /= 2.0
    # Locate Left Cut
    minimum_gray = 0
    while accumulator[minimum_gray]<clip_hist_pct: minimum_gray += 1
    # Locate Right Cut
    maximum_gray = hist_size-1
    while accumulator[maximum_gray]>=(maximum-clip_hist_pct): maximum_gray -= 1
    # Calculate alpha and beta val.
    alpha = 255 / (maximum_gray-minimum_gray)
    beta = -minimum_gray*alpha

    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return (auto_result)
```

### Data Processing
- 이미지 밝기 조절 후 HSV를 이용해 특정 색상 오브젝트 추출
- 마스킹 된 이미지 픽셀 비율 값 추출 (W / W + B)


```python
def get_image_data(dir_in, dir_out):
    ratio_list = []
    for i in tqdm(dir_in):
        name = i.split("\\")[-1]
        img = cv2.imread(i, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (615, 462))
        brightscale = automatic_brightness_and_contrast(img)
        imgcopy = brightscale.copy()
        hsvimg = cv2.cvtColor(brightscale, cv2.COLOR_BGR2HSV)
        lower = np.array([22, 40, 0])
        upper = np.array([85, 255, 245])
        mask = cv2.inRange(hsvimg, lower, upper)
        number_of_white_pix = np.sum(mask==255)
        number_of_black_pix = np.sum(mask==0)
        ratio = number_of_white_pix / (number_of_white_pix + number_of_black_pix)
        ratio_list.append(ratio)
        result = cv2.bitwise_and(imgcopy, imgcopy, mask=mask)
        cv2.imwrite(os.path.join(dir_out, name), result)
    return ratio_list
```


```python
ratio_train = get_image_data(train_imgs, preprocessing_train_imgs)
ratio_test = get_image_data(test_imgs, preprocessing_test_imgs)

processed_train_imgs = glob(main_path+"/PREPROCESSING-TRAIN/*.png") + glob(main_path+"/PREPROCESSING-TRAIN/*.jpg")
processed_train_imgs = sorted(processed_train_imgs)

processed_test_imgs = glob(main_path+"/PREPROCESSING-TEST/*.png") + glob(main_path+"/PREPROCESSING-TEST/*.jpg")
processed_test_imgs = sorted(processed_test_imgs)
```

    100%|██████████| 1592/1592 [04:44<00:00,  5.59it/s]
    100%|██████████| 460/460 [01:24<00:00,  5.44it/s]
    

- 정량이 정해져 있는 변수는 `bfill, ffill`로 결측값 대체 (최근 분무량)
- 정량이 정해져 있지 않은 변수는 보간법 이용
- 최근 분무량 데이터는, 일간 누적 분무량으로 측정이 되어 모든 최근 분무량 데이터를 일간 누적 분무량으로 변환
- 메타데이터에 픽셀 비율 변수로 포함


```python
train_df = []
for i in tqdm(train_data):
    name = i.split("\\")[-1].split(".")[0]
    df = pd.read_csv(i)
    df = df.drop('시간', axis = 1)
    case = name.split("_")[0]
    label = pd.read_csv(main_path + f"/train/{case}/label.csv")
    label_name = [i.split(".")[0] for i in label.img_name]
    label.img_name = label_name
    leaf_weight = label[label.img_name == name].leaf_weight.values[0]
    df["무게"] = leaf_weight
    df["최근분무량"] = df["최근분무량"].fillna(method='bfill', limit=1)
    df["최근분무량"] = df["최근분무량"].fillna(method='ffill', limit=1)
    df = df.interpolate()
    water = df['최근분무량'].round(2).tolist()
    if np.mean(water) > 1000:
        nums = [list(v) for k,v in groupby(water, key = lambda x: x != 0) if k != 0]
        if len(nums) == 2:
            cumulative = nums[0][-1] - nums[0][0] + nums[1][-1]
        else:
            cumulative = nums[0][-1] - nums[0][0]
            
    elif 1000 > np.mean(water) > 0:
        nums = [key for key, _ in groupby(water)]
        cumulative = sum(nums[1:])
    else:
        cumulative = 0

    df = df.mean()
    df = df.to_frame().T
    df["이미지"] = name
    df['최근분무량'] = cumulative

    train_df.append(df)

train_df = pd.concat(train_df, ignore_index=True)
train_df['비율'] = ratio_train
train_df.head()
```

    100%|██████████| 1592/1592 [00:45<00:00, 35.23it/s]
    




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
      <th>내부온도관측치</th>
      <th>외부온도관측치</th>
      <th>내부습도관측치</th>
      <th>외부습도관측치</th>
      <th>CO2관측치</th>
      <th>EC관측치</th>
      <th>최근분무량</th>
      <th>화이트 LED동작강도</th>
      <th>레드 LED동작강도</th>
      <th>블루 LED동작강도</th>
      <th>...</th>
      <th>냉방부하</th>
      <th>난방온도</th>
      <th>난방부하</th>
      <th>총추정광량</th>
      <th>백색광추정광량</th>
      <th>적색광추정광량</th>
      <th>청색광추정광량</th>
      <th>무게</th>
      <th>이미지</th>
      <th>비율</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.236181</td>
      <td>43.868542</td>
      <td>77.740695</td>
      <td>4.679291</td>
      <td>487.226389</td>
      <td>19.594792</td>
      <td>0.0</td>
      <td>200.720833</td>
      <td>201.000000</td>
      <td>0.139583</td>
      <td>...</td>
      <td>179.460356</td>
      <td>18.854103</td>
      <td>10.228598</td>
      <td>145.944829</td>
      <td>12.396061</td>
      <td>21.119466</td>
      <td>NaN</td>
      <td>49.193</td>
      <td>CASE01_01</td>
      <td>0.099845</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.011319</td>
      <td>44.573750</td>
      <td>77.645972</td>
      <td>4.537500</td>
      <td>480.144444</td>
      <td>20.855555</td>
      <td>0.0</td>
      <td>200.861111</td>
      <td>200.861111</td>
      <td>0.139583</td>
      <td>...</td>
      <td>179.471631</td>
      <td>18.853965</td>
      <td>13.709128</td>
      <td>145.980283</td>
      <td>12.391464</td>
      <td>21.099885</td>
      <td>NaN</td>
      <td>59.764</td>
      <td>CASE01_02</td>
      <td>0.120072</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.931111</td>
      <td>39.537708</td>
      <td>77.514931</td>
      <td>4.886111</td>
      <td>489.068750</td>
      <td>20.748611</td>
      <td>0.0</td>
      <td>200.651042</td>
      <td>200.373264</td>
      <td>0.139583</td>
      <td>...</td>
      <td>179.523570</td>
      <td>18.854171</td>
      <td>13.348331</td>
      <td>146.015736</td>
      <td>12.374227</td>
      <td>21.133608</td>
      <td>NaN</td>
      <td>72.209</td>
      <td>CASE01_03</td>
      <td>0.141682</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.027986</td>
      <td>58.497500</td>
      <td>80.069930</td>
      <td>3.908333</td>
      <td>481.378472</td>
      <td>18.195278</td>
      <td>0.0</td>
      <td>200.025000</td>
      <td>200.163889</td>
      <td>0.139583</td>
      <td>...</td>
      <td>179.495845</td>
      <td>18.854174</td>
      <td>7.520480</td>
      <td>145.997472</td>
      <td>12.370205</td>
      <td>21.128169</td>
      <td>NaN</td>
      <td>85.737</td>
      <td>CASE01_04</td>
      <td>0.166269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21.874305</td>
      <td>67.058819</td>
      <td>81.349792</td>
      <td>3.908333</td>
      <td>490.568750</td>
      <td>19.400486</td>
      <td>0.0</td>
      <td>200.861111</td>
      <td>201.000000</td>
      <td>0.139583</td>
      <td>...</td>
      <td>179.488241</td>
      <td>18.854140</td>
      <td>10.943552</td>
      <td>145.971688</td>
      <td>12.394912</td>
      <td>21.121642</td>
      <td>NaN</td>
      <td>102.537</td>
      <td>CASE01_05</td>
      <td>0.191539</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
test_df = []
for i in tqdm(test_data):
    name = i.split("\\")[-1].split(".")[0]
    df = pd.read_csv(i)
    df = df.drop("시간", axis=1)
    df["최근분무량"] = df["최근분무량"].fillna(method="bfill", limit=1)
    df["최근분무량"] = df["최근분무량"].fillna(method="ffill", limit=1)
    df = df.interpolate()
    water = df["최근분무량"].round(2).tolist()
    if np.mean(water)>1000:
        nums = [list(v) for k,v in groupby(water, key = lambda x: x!=0) if k != 0]
        if len(nums)==2: cumulative = nums[0][-1] - nums[0][0] + nums[1][-1]
        else: cumulative = nums[0][-1] - nums[0][0]
    elif 1000>np.mean(water)>0:
        nums = [key for key, _group in groupby(water)]
        cumulative = sum(nums[1:])
    else:
        cumulative = 0
    
    df = df.mean()
    df = df.to_frame().T
    df["이미지"] = name
    df["최근분무량"] = cumulative

    test_df.append(df)

test_df = pd.concat(test_df, ignore_index=True)
test_df["비율"] = ratio_test
test_df.head()
```

    100%|██████████| 460/460 [00:12<00:00, 37.79it/s]
    




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
      <th>내부온도관측치</th>
      <th>외부온도관측치</th>
      <th>내부습도관측치</th>
      <th>외부습도관측치</th>
      <th>CO2관측치</th>
      <th>EC관측치</th>
      <th>최근분무량</th>
      <th>화이트 LED동작강도</th>
      <th>레드 LED동작강도</th>
      <th>블루 LED동작강도</th>
      <th>냉방온도</th>
      <th>냉방부하</th>
      <th>난방온도</th>
      <th>난방부하</th>
      <th>총추정광량</th>
      <th>백색광추정광량</th>
      <th>적색광추정광량</th>
      <th>청색광추정광량</th>
      <th>이미지</th>
      <th>비율</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.634861</td>
      <td>22.564583</td>
      <td>69.500486</td>
      <td>34.499792</td>
      <td>400.265278</td>
      <td>0.000000</td>
      <td>11509.28</td>
      <td>47.195486</td>
      <td>7.484722</td>
      <td>13.488194</td>
      <td>22.395829</td>
      <td>6.198268</td>
      <td>20.395829</td>
      <td>0.000000</td>
      <td>179.576697</td>
      <td>146.045817</td>
      <td>12.398359</td>
      <td>21.132520</td>
      <td>001</td>
      <td>0.162271</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27.170347</td>
      <td>28.738472</td>
      <td>63.881805</td>
      <td>50.837708</td>
      <td>505.823611</td>
      <td>0.000000</td>
      <td>4234.63</td>
      <td>33.496181</td>
      <td>9.506597</td>
      <td>4.722222</td>
      <td>23.396007</td>
      <td>23.875345</td>
      <td>22.396007</td>
      <td>0.000000</td>
      <td>126.724400</td>
      <td>103.602930</td>
      <td>15.725197</td>
      <td>7.396273</td>
      <td>002</td>
      <td>0.526755</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25.999340</td>
      <td>25.541111</td>
      <td>79.197812</td>
      <td>65.936597</td>
      <td>498.623611</td>
      <td>1.778872</td>
      <td>610.71</td>
      <td>47.185417</td>
      <td>7.487153</td>
      <td>13.476736</td>
      <td>20.291920</td>
      <td>42.908664</td>
      <td>17.416844</td>
      <td>0.606961</td>
      <td>179.532997</td>
      <td>146.026479</td>
      <td>12.390315</td>
      <td>21.116202</td>
      <td>003</td>
      <td>0.011583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22.956944</td>
      <td>22.361667</td>
      <td>70.809792</td>
      <td>45.378646</td>
      <td>394.612500</td>
      <td>0.592409</td>
      <td>12271.85</td>
      <td>47.183333</td>
      <td>7.483333</td>
      <td>13.479514</td>
      <td>22.395793</td>
      <td>2.911434</td>
      <td>20.395793</td>
      <td>0.000000</td>
      <td>179.449739</td>
      <td>145.950201</td>
      <td>12.392039</td>
      <td>21.107500</td>
      <td>004</td>
      <td>0.149083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.014757</td>
      <td>22.531736</td>
      <td>73.886944</td>
      <td>33.534167</td>
      <td>418.561806</td>
      <td>0.541303</td>
      <td>13040.85</td>
      <td>47.185417</td>
      <td>7.466667</td>
      <td>13.478125</td>
      <td>22.395846</td>
      <td>3.179260</td>
      <td>20.395846</td>
      <td>0.000000</td>
      <td>179.480870</td>
      <td>145.998546</td>
      <td>12.365033</td>
      <td>21.117290</td>
      <td>005</td>
      <td>0.168687</td>
    </tr>
  </tbody>
</table>
</div>



### Image EDA
- 산점도 플롯을 이용해 무게, 픽셀 비율 관계 확인
- 회귀선에서 멀리 떨어져 있는 값은 이상치로 처리 (제거)
- CASE59와 CASE58은 중복 데이터임


```python
_ = sns.scatterplot(data=train_df, x="무게", y="비율")
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/optimizing_the_growth_env_files/optimizing_the_growth_env_12_0.png?raw=true)
    



```python
image_outliers = ['CASE05_21','CASE05_22','CASE05_23', 'CASE07_07', 'CASE07_08', 'CASE16_03', 'CASE23_01', 'CASE23_02', 
'CASE23_03', 'CASE23_04', 'CASE23_05', 'CASE23_06', 'CASE23_07', 'CASE23_08', 'CASE23_09', 'CASE45_16', 'CASE45_17',
'CASE72_06',  'CASE73_10', 'CASE59_01','CASE59_02','CASE59_03','CASE59_04','CASE59_05','CASE59_06',
'CASE59_07','CASE59_08','CASE59_09','CASE59_10','CASE59_11','CASE59_12','CASE59_13','CASE59_14','CASE59_15','CASE59_16','CASE59_17','CASE59_18',
'CASE59_19','CASE59_20','CASE59_21','CASE59_22','CASE59_23','CASE59_24','CASE59_25','CASE59_26','CASE59_27','CASE59_28','CASE59_29','CASE59_30',
'CASE59_31','CASE59_32', 'CASE59_33']

train_df_image = train_df[~train_df["이미지"].isin(image_outliers)]
train_imgs_removed = [x for x in processed_train_imgs if x.split(".")[1].split("\\")[1] not in image_outliers]
```

### CNN


```python
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
```

    GPU
    

이전에는 토치 GPU 구동 확인을 했었는데 왜 CPU로 잡히는지 모르겠음...


```python
CFG = {
    "IMG_SIZE" : 128,
    "EPOCHS" : 80,
    "LEARNIING_RATE" : 1e-3,
    "BATCH_SIZE" : 32,
    "SEED" : 42
}
```


```python
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG["SEED"])
```


```python
train_len = int(len(train_imgs_removed)*0.8)
weight = train_df_image['무게'].round(3).tolist()

train_img_path = train_imgs_removed[:train_len]
train_label = weight[:train_len]

val_img_path = train_imgs_removed[train_len:]
val_label = weight[train_len:]
```


```python
len(train_img_path), len(train_label), len(val_img_path), len(val_label)
```




    (1232, 1232, 308, 308)




```python
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)
        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else: return image
    def __len__(self):
        return len(self.img_path_list)
```


```python
train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])

test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((CFG['IMG_SIZE'], CFG['IMG_SIZE'])),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ])
```


```python
train_dataset = CustomDataset(train_img_path, train_label, train_mode=True, transforms=train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_img_path, val_label, train_mode=True, transforms=test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
```


```python
class CNNRegressor(torch.nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.regressor = nn.Linear(3136, 1)
    def forward(self, x):
        # (Batch, 3, 128, 128) -> (Batch, 64, 7, 7)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        out = self.regressor(x)
        return out

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    # Loss Func.
    criterion = nn.L1Loss().to(device)
    best_mae = 9999
    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.float().to(device)
            optimizer.zero_grad()
            logit = model(img)
            loss = criterion(logit.squeeze(1), label)
            # backpropagation
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        if scheduler is not None:
            scheduler.step()
        
        # Evaluation
        val_mae = validation(model, val_loader, criterion, device)
        print(f"Epoch [{epoch}] Train MAE: [{np.mean(train_loss):.5f}] Val MAE: [{val_mae:.5f}]\n")
        if best_mae > val_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "./best_model.pth")
            print("Model Saved.")

def validation(model, val_loader, criterion, device):
    model.eval() # Evaluation
    val_loss = []
    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):
            img, label = img.float().to(device), label.float().to(device)

            logit = model(img)
            loss = criterion(logit.squeeze(1), label)
            
            val_loss.append(loss.item())

    val_mae_loss = np.mean(val_loss)
    return val_mae_loss
```

#### Train and Validation


```python
CNNmodel = CNNRegressor().to(device)

optimizer = torch.optim.SGD(params=CNNmodel.parameters(), lr=CFG["LEARNIING_RATE"])
scheduler = None

train(CNNmodel, optimizer, train_loader, val_loader, scheduler, device)
```

    100%|██████████| 39/39 [00:20<00:00,  1.90it/s]
    100%|██████████| 10/10 [00:04<00:00,  2.39it/s]
    

    Epoch [1] Train MAE: [77.63346] Val MAE: [85.81707]
    
    Model Saved.
    

    100%|██████████| 39/39 [00:09<00:00,  4.21it/s]
    100%|██████████| 10/10 [00:02<00:00,  4.28it/s]
    

    Epoch [2] Train MAE: [77.04038] Val MAE: [81.62168]
    
    Model Saved.
    

    100%|██████████| 39/39 [00:09<00:00,  4.24it/s]
    100%|██████████| 10/10 [00:02<00:00,  4.85it/s]
    

   ...
    

    

    100%|██████████| 39/39 [00:09<00:00,  4.27it/s]
    100%|██████████| 10/10 [00:02<00:00,  4.58it/s]

    Epoch [80] Train MAE: [7.41040] Val MAE: [13.52668]
    
    

    
    

#### Predict


```python
def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)
            pred_logit = model(img)
            pred_logit = pred_logit.squeeze(1).detach().cpu()
            model_pred.extend(pred_logit.tolist())
    return model_pred
```


```python
test_dataset = CustomDataset(processed_test_imgs, None, train_mode=False, transforms=test_transform)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

checkpoint = torch.load('./best_model.pth')
CNNmodel = CNNRegressor().to(device)
CNNmodel.load_state_dict(checkpoint)

# Inference
preds = predict(CNNmodel, test_loader, device)
```

    100%|██████████| 15/15 [00:05<00:00,  2.74it/s]
    


```python
submission = pd.read_csv('./open/sample_submission.csv')
submission['leaf_weight'] = preds
submission.to_csv('./CNNsubmit.csv', index=False)
```

### Meatadata EDA
- 각 환경 변수 시각화
- 이상치 판단
- CASE01, CASE02 경우 EC 관측치, 외부 온도 값이 다른 케이스에 비해 다르므로 제외
- 음수값이 나오는 최근 분무량 제외
- CO2 관측치가 0인 케이스는 누락 데이터로 판단 -> 제외


```python
firstfeats = ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', 'EC관측치','최근분무량']
secondfeats = ['냉방온도', '냉방부하','난방온도', '난방부하', '비율']
thirdfeats = ['화이트 LED동작강도', '레드 LED동작강도', '블루 LED동작강도', '총추정광량', '백색광추정광량', '적색광추정광량', '청색광추정광량']

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

idx = 0
for row in range(3):
    for col in range(3):
        try:
            sns.scatterplot(x=train_df[firstfeats[idx]], y=train_df[firstfeats[idx]], ax=ax[row][col], hue=train_df[firstfeats[idx]])
            idx += 1
        except:
            pass
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/optimizing_the_growth_env_files/optimizing_the_growth_env_32_0.png?raw=true)
    



```python
meta_outliers = ['CASE01_01','CASE01_02','CASE01_03','CASE01_04','CASE01_05','CASE01_06','CASE01_07',
'CASE01_08','CASE01_09','CASE02_01','CASE02_02','CASE02_03','CASE02_04','CASE02_05','CASE02_06','CASE02_07',
'CASE02_08','CASE02_09','CASE02_10','CASE02_11']

train_df_meta = train_df_image[~train_df_image['이미지'].isin(meta_outliers)]

train_df_meta = train_df_meta[train_df_meta['CO2관측치'] > 0]
train_df_meta = train_df_meta[train_df_meta['최근분무량'] >= 0]
```

- 특성간 상관관계가 높은 변수는 제거 (LED 동작 강도)
- 총추정광량은 백색추정광량, 적생광추정과량, 청색광추정광량 합이므로 총추정광향 변수 제거


```python
corr = train_df_meta.corr()
corr.style.background_gradient(cmap='coolwarm')
```




<style type="text/css">
#T_6bf28_row0_col0, #T_6bf28_row1_col1, #T_6bf28_row2_col2, #T_6bf28_row3_col3, #T_6bf28_row4_col4, #T_6bf28_row5_col5, #T_6bf28_row6_col6, #T_6bf28_row7_col7, #T_6bf28_row8_col8, #T_6bf28_row8_col16, #T_6bf28_row9_col9, #T_6bf28_row9_col17, #T_6bf28_row10_col10, #T_6bf28_row11_col11, #T_6bf28_row12_col12, #T_6bf28_row13_col13, #T_6bf28_row14_col14, #T_6bf28_row15_col15, #T_6bf28_row16_col8, #T_6bf28_row16_col16, #T_6bf28_row17_col9, #T_6bf28_row17_col17, #T_6bf28_row18_col18, #T_6bf28_row19_col19 {
  background-color: #b40426;
  color: #f1f1f1;
}
#T_6bf28_row0_col1 {
  background-color: #c83836;
  color: #f1f1f1;
}
#T_6bf28_row0_col2, #T_6bf28_row1_col8, #T_6bf28_row1_col9, #T_6bf28_row1_col14, #T_6bf28_row1_col16, #T_6bf28_row1_col17, #T_6bf28_row2_col0, #T_6bf28_row2_col5, #T_6bf28_row5_col3, #T_6bf28_row7_col13, #T_6bf28_row7_col18, #T_6bf28_row7_col19, #T_6bf28_row9_col1, #T_6bf28_row10_col4, #T_6bf28_row11_col10, #T_6bf28_row11_col12, #T_6bf28_row12_col11, #T_6bf28_row13_col7, #T_6bf28_row13_col15, #T_6bf28_row17_col1, #T_6bf28_row18_col6 {
  background-color: #3b4cc0;
  color: #f1f1f1;
}
#T_6bf28_row0_col3 {
  background-color: #5977e3;
  color: #f1f1f1;
}
#T_6bf28_row0_col4, #T_6bf28_row4_col10 {
  background-color: #82a6fb;
  color: #f1f1f1;
}
#T_6bf28_row0_col5, #T_6bf28_row8_col14, #T_6bf28_row12_col1, #T_6bf28_row13_col2, #T_6bf28_row16_col14 {
  background-color: #dbdcde;
  color: #000000;
}
#T_6bf28_row0_col6, #T_6bf28_row2_col1, #T_6bf28_row10_col7, #T_6bf28_row19_col7 {
  background-color: #3f53c6;
  color: #f1f1f1;
}
#T_6bf28_row0_col7, #T_6bf28_row7_col4, #T_6bf28_row9_col10, #T_6bf28_row14_col4, #T_6bf28_row17_col10 {
  background-color: #90b2fe;
  color: #000000;
}
#T_6bf28_row0_col8, #T_6bf28_row0_col16 {
  background-color: #4055c8;
  color: #f1f1f1;
}
#T_6bf28_row0_col9, #T_6bf28_row0_col17 {
  background-color: #506bda;
  color: #f1f1f1;
}
#T_6bf28_row0_col10, #T_6bf28_row4_col2 {
  background-color: #afcafc;
  color: #000000;
}
#T_6bf28_row0_col11 {
  background-color: #f59c7d;
  color: #000000;
}
#T_6bf28_row0_col12, #T_6bf28_row8_col2, #T_6bf28_row10_col13, #T_6bf28_row16_col2 {
  background-color: #e8d6cc;
  color: #000000;
}
#T_6bf28_row0_col13, #T_6bf28_row12_col8, #T_6bf28_row12_col16 {
  background-color: #5875e1;
  color: #f1f1f1;
}
#T_6bf28_row0_col14, #T_6bf28_row5_col6 {
  background-color: #4f69d9;
  color: #f1f1f1;
}
#T_6bf28_row0_col15, #T_6bf28_row3_col6 {
  background-color: #7b9ff9;
  color: #f1f1f1;
}
#T_6bf28_row0_col18, #T_6bf28_row3_col12, #T_6bf28_row7_col3, #T_6bf28_row11_col14 {
  background-color: #7699f6;
  color: #f1f1f1;
}
#T_6bf28_row0_col19, #T_6bf28_row9_col11, #T_6bf28_row10_col14, #T_6bf28_row17_col11 {
  background-color: #7295f4;
  color: #f1f1f1;
}
#T_6bf28_row1_col0 {
  background-color: #c73635;
  color: #f1f1f1;
}
#T_6bf28_row1_col2, #T_6bf28_row6_col19, #T_6bf28_row15_col8, #T_6bf28_row15_col16 {
  background-color: #4c66d6;
  color: #f1f1f1;
}
#T_6bf28_row1_col3, #T_6bf28_row19_col4 {
  background-color: #6a8bef;
  color: #f1f1f1;
}
#T_6bf28_row1_col4, #T_6bf28_row3_col7, #T_6bf28_row13_col6 {
  background-color: #5f7fe8;
  color: #f1f1f1;
}
#T_6bf28_row1_col5 {
  background-color: #c3d5f4;
  color: #000000;
}
#T_6bf28_row1_col6, #T_6bf28_row6_col4, #T_6bf28_row14_col1 {
  background-color: #4a63d3;
  color: #f1f1f1;
}
#T_6bf28_row1_col7, #T_6bf28_row14_col10 {
  background-color: #7da0f9;
  color: #f1f1f1;
}
#T_6bf28_row1_col10, #T_6bf28_row6_col3 {
  background-color: #a5c3fe;
  color: #000000;
}
#T_6bf28_row1_col11 {
  background-color: #f59f80;
  color: #000000;
}
#T_6bf28_row1_col12, #T_6bf28_row6_col2, #T_6bf28_row10_col8, #T_6bf28_row10_col16 {
  background-color: #dcdddd;
  color: #000000;
}
#T_6bf28_row1_col13 {
  background-color: #536edd;
  color: #f1f1f1;
}
#T_6bf28_row1_col15, #T_6bf28_row3_col0, #T_6bf28_row12_col2, #T_6bf28_row14_col18 {
  background-color: #6b8df0;
  color: #f1f1f1;
}
#T_6bf28_row1_col18 {
  background-color: #8db0fe;
  color: #000000;
}
#T_6bf28_row1_col19, #T_6bf28_row2_col15, #T_6bf28_row13_col14, #T_6bf28_row15_col1 {
  background-color: #8badfd;
  color: #000000;
}
#T_6bf28_row2_col3 {
  background-color: #e7745b;
  color: #f1f1f1;
}
#T_6bf28_row2_col4 {
  background-color: #5b7ae5;
  color: #f1f1f1;
}
#T_6bf28_row2_col6, #T_6bf28_row4_col15, #T_6bf28_row5_col8, #T_6bf28_row5_col16 {
  background-color: #b9d0f9;
  color: #000000;
}
#T_6bf28_row2_col7, #T_6bf28_row6_col0, #T_6bf28_row9_col4, #T_6bf28_row17_col4 {
  background-color: #7ea1fa;
  color: #f1f1f1;
}
#T_6bf28_row2_col8, #T_6bf28_row2_col16, #T_6bf28_row13_col3 {
  background-color: #d8dce2;
  color: #000000;
}
#T_6bf28_row2_col9, #T_6bf28_row2_col17 {
  background-color: #d4dbe6;
  color: #000000;
}
#T_6bf28_row2_col10, #T_6bf28_row3_col9, #T_6bf28_row3_col17, #T_6bf28_row13_col5 {
  background-color: #9fbfff;
  color: #000000;
}
#T_6bf28_row2_col11, #T_6bf28_row19_col5 {
  background-color: #85a8fc;
  color: #f1f1f1;
}
#T_6bf28_row2_col12, #T_6bf28_row8_col4, #T_6bf28_row16_col4 {
  background-color: #6180e9;
  color: #f1f1f1;
}
#T_6bf28_row2_col13, #T_6bf28_row4_col5, #T_6bf28_row13_col19 {
  background-color: #c5d6f2;
  color: #000000;
}
#T_6bf28_row2_col14, #T_6bf28_row10_col5 {
  background-color: #c0d4f5;
  color: #000000;
}
#T_6bf28_row2_col18, #T_6bf28_row18_col13 {
  background-color: #c7d7f0;
  color: #000000;
}
#T_6bf28_row2_col19, #T_6bf28_row6_col15, #T_6bf28_row8_col19, #T_6bf28_row16_col19 {
  background-color: #d3dbe7;
  color: #000000;
}
#T_6bf28_row3_col1, #T_6bf28_row11_col6, #T_6bf28_row14_col0 {
  background-color: #6c8ff1;
  color: #f1f1f1;
}
#T_6bf28_row3_col2 {
  background-color: #e46e56;
  color: #f1f1f1;
}
#T_6bf28_row3_col4, #T_6bf28_row12_col4, #T_6bf28_row18_col15 {
  background-color: #465ecf;
  color: #f1f1f1;
}
#T_6bf28_row3_col5 {
  background-color: #3d50c3;
  color: #f1f1f1;
}
#T_6bf28_row3_col8, #T_6bf28_row3_col16, #T_6bf28_row5_col1, #T_6bf28_row9_col6, #T_6bf28_row13_col11, #T_6bf28_row17_col6, #T_6bf28_row18_col3 {
  background-color: #c4d5f3;
  color: #000000;
}
#T_6bf28_row3_col10, #T_6bf28_row5_col11, #T_6bf28_row10_col1 {
  background-color: #a9c6fd;
  color: #000000;
}
#T_6bf28_row3_col11, #T_6bf28_row4_col1, #T_6bf28_row10_col18, #T_6bf28_row18_col12, #T_6bf28_row19_col12 {
  background-color: #a7c5fe;
  color: #000000;
}
#T_6bf28_row3_col13, #T_6bf28_row8_col18, #T_6bf28_row16_col18 {
  background-color: #ccd9ed;
  color: #000000;
}
#T_6bf28_row3_col14, #T_6bf28_row12_col18 {
  background-color: #86a9fc;
  color: #f1f1f1;
}
#T_6bf28_row3_col15 {
  background-color: #6485ec;
  color: #f1f1f1;
}
#T_6bf28_row3_col18 {
  background-color: #adc9fd;
  color: #000000;
}
#T_6bf28_row3_col19 {
  background-color: #b6cefa;
  color: #000000;
}
#T_6bf28_row4_col0, #T_6bf28_row5_col15 {
  background-color: #cad8ef;
  color: #000000;
}
#T_6bf28_row4_col3, #T_6bf28_row6_col8, #T_6bf28_row6_col16, #T_6bf28_row11_col2 {
  background-color: #8fb1fe;
  color: #000000;
}
#T_6bf28_row4_col6, #T_6bf28_row5_col18, #T_6bf28_row8_col0, #T_6bf28_row16_col0 {
  background-color: #6788ee;
  color: #f1f1f1;
}
#T_6bf28_row4_col7, #T_6bf28_row19_col9, #T_6bf28_row19_col17 {
  background-color: #bad0f8;
  color: #000000;
}
#T_6bf28_row4_col8, #T_6bf28_row4_col16 {
  background-color: #93b5fe;
  color: #000000;
}
#T_6bf28_row4_col9, #T_6bf28_row4_col17, #T_6bf28_row18_col10 {
  background-color: #bed2f6;
  color: #000000;
}
#T_6bf28_row4_col11 {
  background-color: #d7dce3;
  color: #000000;
}
#T_6bf28_row4_col12, #T_6bf28_row4_col19, #T_6bf28_row6_col12, #T_6bf28_row9_col13, #T_6bf28_row12_col15, #T_6bf28_row15_col4, #T_6bf28_row17_col13 {
  background-color: #94b6ff;
  color: #000000;
}
#T_6bf28_row4_col13, #T_6bf28_row6_col1 {
  background-color: #7a9df8;
  color: #f1f1f1;
}
#T_6bf28_row4_col14, #T_6bf28_row13_col18 {
  background-color: #c1d4f4;
  color: #000000;
}
#T_6bf28_row4_col18, #T_6bf28_row10_col9, #T_6bf28_row10_col17, #T_6bf28_row12_col7 {
  background-color: #96b7ff;
  color: #000000;
}
#T_6bf28_row5_col0 {
  background-color: #e1dad6;
  color: #000000;
}
#T_6bf28_row5_col2 {
  background-color: #4961d2;
  color: #f1f1f1;
}
#T_6bf28_row5_col4, #T_6bf28_row5_col13, #T_6bf28_row6_col10, #T_6bf28_row12_col14, #T_6bf28_row14_col11 {
  background-color: #88abfd;
  color: #000000;
}
#T_6bf28_row5_col7, #T_6bf28_row14_col2, #T_6bf28_row18_col8, #T_6bf28_row18_col16 {
  background-color: #d2dbe8;
  color: #000000;
}
#T_6bf28_row5_col9, #T_6bf28_row5_col17, #T_6bf28_row7_col9, #T_6bf28_row7_col17, #T_6bf28_row9_col5, #T_6bf28_row17_col5 {
  background-color: #eed0c0;
  color: #000000;
}
#T_6bf28_row5_col10, #T_6bf28_row10_col0 {
  background-color: #bcd2f7;
  color: #000000;
}
#T_6bf28_row5_col12, #T_6bf28_row18_col2 {
  background-color: #e0dbd8;
  color: #000000;
}
#T_6bf28_row5_col14 {
  background-color: #e9d5cb;
  color: #000000;
}
#T_6bf28_row5_col19 {
  background-color: #6687ed;
  color: #f1f1f1;
}
#T_6bf28_row6_col5, #T_6bf28_row13_col0, #T_6bf28_row14_col13, #T_6bf28_row15_col3, #T_6bf28_row18_col14, #T_6bf28_row19_col14 {
  background-color: #80a3fa;
  color: #f1f1f1;
}
#T_6bf28_row6_col7, #T_6bf28_row14_col8, #T_6bf28_row14_col16, #T_6bf28_row19_col8, #T_6bf28_row19_col16 {
  background-color: #d6dce4;
  color: #000000;
}
#T_6bf28_row6_col9, #T_6bf28_row7_col5 {
  background-color: #dedcdb;
  color: #000000;
}
#T_6bf28_row6_col11, #T_6bf28_row11_col7, #T_6bf28_row19_col0 {
  background-color: #9dbdff;
  color: #000000;
}
#T_6bf28_row6_col13 {
  background-color: #7597f6;
  color: #f1f1f1;
}
#T_6bf28_row6_col14, #T_6bf28_row6_col17 {
  background-color: #dddcdc;
  color: #000000;
}
#T_6bf28_row6_col18, #T_6bf28_row13_col4 {
  background-color: #485fd1;
  color: #f1f1f1;
}
#T_6bf28_row7_col0 {
  background-color: #b2ccfb;
  color: #000000;
}
#T_6bf28_row7_col1 {
  background-color: #97b8ff;
  color: #000000;
}
#T_6bf28_row7_col2, #T_6bf28_row9_col12, #T_6bf28_row11_col4, #T_6bf28_row15_col0, #T_6bf28_row17_col12 {
  background-color: #a3c2fe;
  color: #000000;
}
#T_6bf28_row7_col6 {
  background-color: #cbd8ee;
  color: #000000;
}
#T_6bf28_row7_col8, #T_6bf28_row7_col16, #T_6bf28_row10_col15, #T_6bf28_row19_col15 {
  background-color: #445acc;
  color: #f1f1f1;
}
#T_6bf28_row7_col10 {
  background-color: #5572df;
  color: #f1f1f1;
}
#T_6bf28_row7_col11 {
  background-color: #b5cdfa;
  color: #000000;
}
#T_6bf28_row7_col12, #T_6bf28_row10_col2, #T_6bf28_row11_col13 {
  background-color: #aec9fc;
  color: #000000;
}
#T_6bf28_row7_col14 {
  background-color: #e97a5f;
  color: #f1f1f1;
}
#T_6bf28_row7_col15, #T_6bf28_row15_col7 {
  background-color: #c12b30;
  color: #f1f1f1;
}
#T_6bf28_row8_col1, #T_6bf28_row16_col1 {
  background-color: #5470de;
  color: #f1f1f1;
}
#T_6bf28_row8_col3, #T_6bf28_row16_col3 {
  background-color: #d1dae9;
  color: #000000;
}
#T_6bf28_row8_col5, #T_6bf28_row15_col6, #T_6bf28_row16_col5, #T_6bf28_row19_col3, #T_6bf28_row19_col13 {
  background-color: #c9d7f0;
  color: #000000;
}
#T_6bf28_row8_col6, #T_6bf28_row16_col6 {
  background-color: #799cf8;
  color: #f1f1f1;
}
#T_6bf28_row8_col7, #T_6bf28_row16_col7 {
  background-color: #4257c9;
  color: #f1f1f1;
}
#T_6bf28_row8_col9, #T_6bf28_row8_col17, #T_6bf28_row16_col9, #T_6bf28_row16_col17 {
  background-color: #f7b79b;
  color: #000000;
}
#T_6bf28_row8_col10, #T_6bf28_row9_col15, #T_6bf28_row16_col10, #T_6bf28_row17_col15 {
  background-color: #e4d9d2;
  color: #000000;
}
#T_6bf28_row8_col11, #T_6bf28_row14_col3, #T_6bf28_row16_col11 {
  background-color: #92b4fe;
  color: #000000;
}
#T_6bf28_row8_col12, #T_6bf28_row16_col12 {
  background-color: #7396f5;
  color: #f1f1f1;
}
#T_6bf28_row8_col13, #T_6bf28_row16_col13 {
  background-color: #f7a98b;
  color: #000000;
}
#T_6bf28_row8_col15, #T_6bf28_row16_col15 {
  background-color: #455cce;
  color: #f1f1f1;
}
#T_6bf28_row9_col0, #T_6bf28_row17_col0 {
  background-color: #5d7ce6;
  color: #f1f1f1;
}
#T_6bf28_row9_col2, #T_6bf28_row15_col5, #T_6bf28_row17_col2 {
  background-color: #dadce0;
  color: #000000;
}
#T_6bf28_row9_col3, #T_6bf28_row9_col18, #T_6bf28_row17_col3, #T_6bf28_row17_col18 {
  background-color: #9bbcff;
  color: #000000;
}
#T_6bf28_row9_col7, #T_6bf28_row17_col7 {
  background-color: #e5d8d1;
  color: #000000;
}
#T_6bf28_row9_col8, #T_6bf28_row9_col16, #T_6bf28_row17_col8, #T_6bf28_row17_col16 {
  background-color: #f5c0a7;
  color: #000000;
}
#T_6bf28_row9_col14, #T_6bf28_row17_col14 {
  background-color: #df634e;
  color: #f1f1f1;
}
#T_6bf28_row9_col19, #T_6bf28_row11_col3, #T_6bf28_row17_col19, #T_6bf28_row18_col0 {
  background-color: #a2c1ff;
  color: #000000;
}
#T_6bf28_row10_col3 {
  background-color: #aac7fd;
  color: #000000;
}
#T_6bf28_row10_col6, #T_6bf28_row15_col10 {
  background-color: #5e7de7;
  color: #f1f1f1;
}
#T_6bf28_row10_col11, #T_6bf28_row15_col18, #T_6bf28_row15_col19, #T_6bf28_row18_col7 {
  background-color: #4358cb;
  color: #f1f1f1;
}
#T_6bf28_row10_col12 {
  background-color: #f6a283;
  color: #000000;
}
#T_6bf28_row10_col19, #T_6bf28_row13_col9, #T_6bf28_row13_col17, #T_6bf28_row18_col1 {
  background-color: #abc8fd;
  color: #000000;
}
#T_6bf28_row11_col0 {
  background-color: #f39778;
  color: #000000;
}
#T_6bf28_row11_col1 {
  background-color: #f5a081;
  color: #000000;
}
#T_6bf28_row11_col5, #T_6bf28_row15_col11, #T_6bf28_row18_col11, #T_6bf28_row19_col1 {
  background-color: #a6c4fe;
  color: #000000;
}
#T_6bf28_row11_col8, #T_6bf28_row11_col16 {
  background-color: #779af7;
  color: #f1f1f1;
}
#T_6bf28_row11_col9, #T_6bf28_row11_col17, #T_6bf28_row12_col3, #T_6bf28_row14_col19 {
  background-color: #6f92f3;
  color: #f1f1f1;
}
#T_6bf28_row11_col15, #T_6bf28_row12_col19, #T_6bf28_row18_col5 {
  background-color: #89acfd;
  color: #000000;
}
#T_6bf28_row11_col18 {
  background-color: #84a7fc;
  color: #f1f1f1;
}
#T_6bf28_row11_col19 {
  background-color: #81a4fb;
  color: #f1f1f1;
}
#T_6bf28_row12_col0 {
  background-color: #ebd3c6;
  color: #000000;
}
#T_6bf28_row12_col5 {
  background-color: #dfdbd9;
  color: #000000;
}
#T_6bf28_row12_col6 {
  background-color: #6384eb;
  color: #f1f1f1;
}
#T_6bf28_row12_col9, #T_6bf28_row12_col17, #T_6bf28_row19_col11 {
  background-color: #a1c0ff;
  color: #000000;
}
#T_6bf28_row12_col10 {
  background-color: #f6a586;
  color: #000000;
}
#T_6bf28_row12_col13, #T_6bf28_row19_col6 {
  background-color: #3c4ec2;
  color: #f1f1f1;
}
#T_6bf28_row13_col1, #T_6bf28_row18_col4 {
  background-color: #6e90f2;
  color: #f1f1f1;
}
#T_6bf28_row13_col8, #T_6bf28_row13_col16 {
  background-color: #f7a889;
  color: #000000;
}
#T_6bf28_row13_col10, #T_6bf28_row15_col9, #T_6bf28_row15_col17 {
  background-color: #efcfbf;
  color: #000000;
}
#T_6bf28_row13_col12 {
  background-color: #5a78e4;
  color: #f1f1f1;
}
#T_6bf28_row14_col5 {
  background-color: #edd1c2;
  color: #000000;
}
#T_6bf28_row14_col6 {
  background-color: #cdd9ec;
  color: #000000;
}
#T_6bf28_row14_col7 {
  background-color: #eb7d62;
  color: #f1f1f1;
}
#T_6bf28_row14_col9, #T_6bf28_row14_col17 {
  background-color: #dd5f4b;
  color: #f1f1f1;
}
#T_6bf28_row14_col12 {
  background-color: #9abbff;
  color: #000000;
}
#T_6bf28_row14_col15 {
  background-color: #e36c55;
  color: #f1f1f1;
}
#T_6bf28_row15_col2, #T_6bf28_row15_col12 {
  background-color: #b1cbfc;
  color: #000000;
}
#T_6bf28_row15_col13 {
  background-color: #3e51c5;
  color: #f1f1f1;
}
#T_6bf28_row15_col14 {
  background-color: #e26952;
  color: #f1f1f1;
}
#T_6bf28_row18_col9, #T_6bf28_row18_col17 {
  background-color: #b7cff9;
  color: #000000;
}
#T_6bf28_row18_col19, #T_6bf28_row19_col18 {
  background-color: #b50927;
  color: #f1f1f1;
}
#T_6bf28_row19_col2 {
  background-color: #e7d7ce;
  color: #000000;
}
#T_6bf28_row19_col10 {
  background-color: #bfd3f6;
  color: #000000;
}
</style>
<table id="T_6bf28">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_6bf28_level0_col0" class="col_heading level0 col0" >내부온도관측치</th>
      <th id="T_6bf28_level0_col1" class="col_heading level0 col1" >외부온도관측치</th>
      <th id="T_6bf28_level0_col2" class="col_heading level0 col2" >내부습도관측치</th>
      <th id="T_6bf28_level0_col3" class="col_heading level0 col3" >외부습도관측치</th>
      <th id="T_6bf28_level0_col4" class="col_heading level0 col4" >CO2관측치</th>
      <th id="T_6bf28_level0_col5" class="col_heading level0 col5" >EC관측치</th>
      <th id="T_6bf28_level0_col6" class="col_heading level0 col6" >최근분무량</th>
      <th id="T_6bf28_level0_col7" class="col_heading level0 col7" >화이트 LED동작강도</th>
      <th id="T_6bf28_level0_col8" class="col_heading level0 col8" >레드 LED동작강도</th>
      <th id="T_6bf28_level0_col9" class="col_heading level0 col9" >블루 LED동작강도</th>
      <th id="T_6bf28_level0_col10" class="col_heading level0 col10" >냉방온도</th>
      <th id="T_6bf28_level0_col11" class="col_heading level0 col11" >냉방부하</th>
      <th id="T_6bf28_level0_col12" class="col_heading level0 col12" >난방온도</th>
      <th id="T_6bf28_level0_col13" class="col_heading level0 col13" >난방부하</th>
      <th id="T_6bf28_level0_col14" class="col_heading level0 col14" >총추정광량</th>
      <th id="T_6bf28_level0_col15" class="col_heading level0 col15" >백색광추정광량</th>
      <th id="T_6bf28_level0_col16" class="col_heading level0 col16" >적색광추정광량</th>
      <th id="T_6bf28_level0_col17" class="col_heading level0 col17" >청색광추정광량</th>
      <th id="T_6bf28_level0_col18" class="col_heading level0 col18" >무게</th>
      <th id="T_6bf28_level0_col19" class="col_heading level0 col19" >비율</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_6bf28_level0_row0" class="row_heading level0 row0" >내부온도관측치</th>
      <td id="T_6bf28_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_6bf28_row0_col1" class="data row0 col1" >0.914757</td>
      <td id="T_6bf28_row0_col2" class="data row0 col2" >-0.592165</td>
      <td id="T_6bf28_row0_col3" class="data row0 col3" >-0.348850</td>
      <td id="T_6bf28_row0_col4" class="data row0 col4" >0.093577</td>
      <td id="T_6bf28_row0_col5" class="data row0 col5" >0.233087</td>
      <td id="T_6bf28_row0_col6" class="data row0 col6" >-0.260451</td>
      <td id="T_6bf28_row0_col7" class="data row0 col7" >-0.021698</td>
      <td id="T_6bf28_row0_col8" class="data row0 col8" >-0.366392</td>
      <td id="T_6bf28_row0_col9" class="data row0 col9" >-0.412055</td>
      <td id="T_6bf28_row0_col10" class="data row0 col10" >0.028192</td>
      <td id="T_6bf28_row0_col11" class="data row0 col11" >0.608341</td>
      <td id="T_6bf28_row0_col12" class="data row0 col12" >0.300661</td>
      <td id="T_6bf28_row0_col13" class="data row0 col13" >-0.250316</td>
      <td id="T_6bf28_row0_col14" class="data row0 col14" >-0.341991</td>
      <td id="T_6bf28_row0_col15" class="data row0 col15" >-0.089971</td>
      <td id="T_6bf28_row0_col16" class="data row0 col16" >-0.366396</td>
      <td id="T_6bf28_row0_col17" class="data row0 col17" >-0.412047</td>
      <td id="T_6bf28_row0_col18" class="data row0 col18" >-0.096137</td>
      <td id="T_6bf28_row0_col19" class="data row0 col19" >-0.122599</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row1" class="row_heading level0 row1" >외부온도관측치</th>
      <td id="T_6bf28_row1_col0" class="data row1 col0" >0.914757</td>
      <td id="T_6bf28_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_6bf28_row1_col2" class="data row1 col2" >-0.495556</td>
      <td id="T_6bf28_row1_col3" class="data row1 col3" >-0.277455</td>
      <td id="T_6bf28_row1_col4" class="data row1 col4" >-0.025140</td>
      <td id="T_6bf28_row1_col5" class="data row1 col5" >0.104751</td>
      <td id="T_6bf28_row1_col6" class="data row1 col6" >-0.218209</td>
      <td id="T_6bf28_row1_col7" class="data row1 col7" >-0.096884</td>
      <td id="T_6bf28_row1_col8" class="data row1 col8" >-0.394604</td>
      <td id="T_6bf28_row1_col9" class="data row1 col9" >-0.520013</td>
      <td id="T_6bf28_row1_col10" class="data row1 col10" >-0.017607</td>
      <td id="T_6bf28_row1_col11" class="data row1 col11" >0.596009</td>
      <td id="T_6bf28_row1_col12" class="data row1 col12" >0.229053</td>
      <td id="T_6bf28_row1_col13" class="data row1 col13" >-0.270891</td>
      <td id="T_6bf28_row1_col14" class="data row1 col14" >-0.439107</td>
      <td id="T_6bf28_row1_col15" class="data row1 col15" >-0.151600</td>
      <td id="T_6bf28_row1_col16" class="data row1 col16" >-0.394598</td>
      <td id="T_6bf28_row1_col17" class="data row1 col17" >-0.519996</td>
      <td id="T_6bf28_row1_col18" class="data row1 col18" >-0.005515</td>
      <td id="T_6bf28_row1_col19" class="data row1 col19" >-0.029887</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row2" class="row_heading level0 row2" >내부습도관측치</th>
      <td id="T_6bf28_row2_col0" class="data row2 col0" >-0.592165</td>
      <td id="T_6bf28_row2_col1" class="data row2 col1" >-0.495556</td>
      <td id="T_6bf28_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_6bf28_row2_col3" class="data row2 col3" >0.752177</td>
      <td id="T_6bf28_row2_col4" class="data row2 col4" >-0.037328</td>
      <td id="T_6bf28_row2_col5" class="data row2 col5" >-0.517403</td>
      <td id="T_6bf28_row2_col6" class="data row2 col6" >0.201364</td>
      <td id="T_6bf28_row2_col7" class="data row2 col7" >-0.091622</td>
      <td id="T_6bf28_row2_col8" class="data row2 col8" >0.276760</td>
      <td id="T_6bf28_row2_col9" class="data row2 col9" >0.189064</td>
      <td id="T_6bf28_row2_col10" class="data row2 col10" >-0.040844</td>
      <td id="T_6bf28_row2_col11" class="data row2 col11" >-0.185409</td>
      <td id="T_6bf28_row2_col12" class="data row2 col12" >-0.344317</td>
      <td id="T_6bf28_row2_col13" class="data row2 col13" >0.194068</td>
      <td id="T_6bf28_row2_col14" class="data row2 col14" >0.135976</td>
      <td id="T_6bf28_row2_col15" class="data row2 col15" >-0.031516</td>
      <td id="T_6bf28_row2_col16" class="data row2 col16" >0.276777</td>
      <td id="T_6bf28_row2_col17" class="data row2 col17" >0.189070</td>
      <td id="T_6bf28_row2_col18" class="data row2 col18" >0.225323</td>
      <td id="T_6bf28_row2_col19" class="data row2 col19" >0.270407</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row3" class="row_heading level0 row3" >외부습도관측치</th>
      <td id="T_6bf28_row3_col0" class="data row3 col0" >-0.348850</td>
      <td id="T_6bf28_row3_col1" class="data row3 col1" >-0.277455</td>
      <td id="T_6bf28_row3_col2" class="data row3 col2" >0.752177</td>
      <td id="T_6bf28_row3_col3" class="data row3 col3" >1.000000</td>
      <td id="T_6bf28_row3_col4" class="data row3 col4" >-0.114504</td>
      <td id="T_6bf28_row3_col5" class="data row3 col5" >-0.500501</td>
      <td id="T_6bf28_row3_col6" class="data row3 col6" >-0.023921</td>
      <td id="T_6bf28_row3_col7" class="data row3 col7" >-0.219330</td>
      <td id="T_6bf28_row3_col8" class="data row3 col8" >0.181613</td>
      <td id="T_6bf28_row3_col9" class="data row3 col9" >-0.062676</td>
      <td id="T_6bf28_row3_col10" class="data row3 col10" >0.002730</td>
      <td id="T_6bf28_row3_col11" class="data row3 col11" >-0.031856</td>
      <td id="T_6bf28_row3_col12" class="data row3 col12" >-0.250606</td>
      <td id="T_6bf28_row3_col13" class="data row3 col13" >0.225217</td>
      <td id="T_6bf28_row3_col14" class="data row3 col14" >-0.102444</td>
      <td id="T_6bf28_row3_col15" class="data row3 col15" >-0.178893</td>
      <td id="T_6bf28_row3_col16" class="data row3 col16" >0.181610</td>
      <td id="T_6bf28_row3_col17" class="data row3 col17" >-0.062680</td>
      <td id="T_6bf28_row3_col18" class="data row3 col18" >0.117638</td>
      <td id="T_6bf28_row3_col19" class="data row3 col19" >0.141438</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row4" class="row_heading level0 row4" >CO2관측치</th>
      <td id="T_6bf28_row4_col0" class="data row4 col0" >0.093577</td>
      <td id="T_6bf28_row4_col1" class="data row4 col1" >-0.025140</td>
      <td id="T_6bf28_row4_col2" class="data row4 col2" >-0.037328</td>
      <td id="T_6bf28_row4_col3" class="data row4 col3" >-0.114504</td>
      <td id="T_6bf28_row4_col4" class="data row4 col4" >1.000000</td>
      <td id="T_6bf28_row4_col5" class="data row4 col5" >0.112437</td>
      <td id="T_6bf28_row4_col6" class="data row4 col6" >-0.099955</td>
      <td id="T_6bf28_row4_col7" class="data row4 col7" >0.141059</td>
      <td id="T_6bf28_row4_col8" class="data row4 col8" >-0.019642</td>
      <td id="T_6bf28_row4_col9" class="data row4 col9" >0.077924</td>
      <td id="T_6bf28_row4_col10" class="data row4 col10" >-0.163204</td>
      <td id="T_6bf28_row4_col11" class="data row4 col11" >0.201229</td>
      <td id="T_6bf28_row4_col12" class="data row4 col12" >-0.116181</td>
      <td id="T_6bf28_row4_col13" class="data row4 col13" >-0.109370</td>
      <td id="T_6bf28_row4_col14" class="data row4 col14" >0.140163</td>
      <td id="T_6bf28_row4_col15" class="data row4 col15" >0.153977</td>
      <td id="T_6bf28_row4_col16" class="data row4 col16" >-0.019634</td>
      <td id="T_6bf28_row4_col17" class="data row4 col17" >0.077936</td>
      <td id="T_6bf28_row4_col18" class="data row4 col18" >0.024076</td>
      <td id="T_6bf28_row4_col19" class="data row4 col19" >0.009779</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row5" class="row_heading level0 row5" >EC관측치</th>
      <td id="T_6bf28_row5_col0" class="data row5 col0" >0.233087</td>
      <td id="T_6bf28_row5_col1" class="data row5 col1" >0.104751</td>
      <td id="T_6bf28_row5_col2" class="data row5 col2" >-0.517403</td>
      <td id="T_6bf28_row5_col3" class="data row5 col3" >-0.500501</td>
      <td id="T_6bf28_row5_col4" class="data row5 col4" >0.112437</td>
      <td id="T_6bf28_row5_col5" class="data row5 col5" >1.000000</td>
      <td id="T_6bf28_row5_col6" class="data row5 col6" >-0.195874</td>
      <td id="T_6bf28_row5_col7" class="data row5 col7" >0.251206</td>
      <td id="T_6bf28_row5_col8" class="data row5 col8" >0.131422</td>
      <td id="T_6bf28_row5_col9" class="data row5 col9" >0.357484</td>
      <td id="T_6bf28_row5_col10" class="data row5 col10" >0.087924</td>
      <td id="T_6bf28_row5_col11" class="data row5 col11" >-0.026613</td>
      <td id="T_6bf28_row5_col12" class="data row5 col12" >0.256154</td>
      <td id="T_6bf28_row5_col13" class="data row5 col13" >-0.057690</td>
      <td id="T_6bf28_row5_col14" class="data row5 col14" >0.349921</td>
      <td id="T_6bf28_row5_col15" class="data row5 col15" >0.225883</td>
      <td id="T_6bf28_row5_col16" class="data row5 col16" >0.131423</td>
      <td id="T_6bf28_row5_col17" class="data row5 col17" >0.357497</td>
      <td id="T_6bf28_row5_col18" class="data row5 col18" >-0.153996</td>
      <td id="T_6bf28_row5_col19" class="data row5 col19" >-0.171411</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row6" class="row_heading level0 row6" >최근분무량</th>
      <td id="T_6bf28_row6_col0" class="data row6 col0" >-0.260451</td>
      <td id="T_6bf28_row6_col1" class="data row6 col1" >-0.218209</td>
      <td id="T_6bf28_row6_col2" class="data row6 col2" >0.201364</td>
      <td id="T_6bf28_row6_col3" class="data row6 col3" >-0.023921</td>
      <td id="T_6bf28_row6_col4" class="data row6 col4" >-0.099955</td>
      <td id="T_6bf28_row6_col5" class="data row6 col5" >-0.195874</td>
      <td id="T_6bf28_row6_col6" class="data row6 col6" >1.000000</td>
      <td id="T_6bf28_row6_col7" class="data row6 col7" >0.274187</td>
      <td id="T_6bf28_row6_col8" class="data row6 col8" >-0.038251</td>
      <td id="T_6bf28_row6_col9" class="data row6 col9" >0.245956</td>
      <td id="T_6bf28_row6_col10" class="data row6 col10" >-0.136779</td>
      <td id="T_6bf28_row6_col11" class="data row6 col11" >-0.082902</td>
      <td id="T_6bf28_row6_col12" class="data row6 col12" >-0.118560</td>
      <td id="T_6bf28_row6_col13" class="data row6 col13" >-0.130924</td>
      <td id="T_6bf28_row6_col14" class="data row6 col14" >0.283771</td>
      <td id="T_6bf28_row6_col15" class="data row6 col15" >0.265799</td>
      <td id="T_6bf28_row6_col16" class="data row6 col16" >-0.038245</td>
      <td id="T_6bf28_row6_col17" class="data row6 col17" >0.245923</td>
      <td id="T_6bf28_row6_col18" class="data row6 col18" >-0.284305</td>
      <td id="T_6bf28_row6_col19" class="data row6 col19" >-0.276562</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row7" class="row_heading level0 row7" >화이트 LED동작강도</th>
      <td id="T_6bf28_row7_col0" class="data row7 col0" >-0.021698</td>
      <td id="T_6bf28_row7_col1" class="data row7 col1" >-0.096884</td>
      <td id="T_6bf28_row7_col2" class="data row7 col2" >-0.091622</td>
      <td id="T_6bf28_row7_col3" class="data row7 col3" >-0.219330</td>
      <td id="T_6bf28_row7_col4" class="data row7 col4" >0.141059</td>
      <td id="T_6bf28_row7_col5" class="data row7 col5" >0.251206</td>
      <td id="T_6bf28_row7_col6" class="data row7 col6" >0.274187</td>
      <td id="T_6bf28_row7_col7" class="data row7 col7" >1.000000</td>
      <td id="T_6bf28_row7_col8" class="data row7 col8" >-0.347680</td>
      <td id="T_6bf28_row7_col9" class="data row7 col9" >0.354309</td>
      <td id="T_6bf28_row7_col10" class="data row7 col10" >-0.359168</td>
      <td id="T_6bf28_row7_col11" class="data row7 col11" >0.023614</td>
      <td id="T_6bf28_row7_col12" class="data row7 col12" >-0.003158</td>
      <td id="T_6bf28_row7_col13" class="data row7 col13" >-0.383132</td>
      <td id="T_6bf28_row7_col14" class="data row7 col14" >0.745572</td>
      <td id="T_6bf28_row7_col15" class="data row7 col15" >0.950771</td>
      <td id="T_6bf28_row7_col16" class="data row7 col16" >-0.347626</td>
      <td id="T_6bf28_row7_col17" class="data row7 col17" >0.354327</td>
      <td id="T_6bf28_row7_col18" class="data row7 col18" >-0.343482</td>
      <td id="T_6bf28_row7_col19" class="data row7 col19" >-0.360687</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row8" class="row_heading level0 row8" >레드 LED동작강도</th>
      <td id="T_6bf28_row8_col0" class="data row8 col0" >-0.366392</td>
      <td id="T_6bf28_row8_col1" class="data row8 col1" >-0.394604</td>
      <td id="T_6bf28_row8_col2" class="data row8 col2" >0.276760</td>
      <td id="T_6bf28_row8_col3" class="data row8 col3" >0.181613</td>
      <td id="T_6bf28_row8_col4" class="data row8 col4" >-0.019642</td>
      <td id="T_6bf28_row8_col5" class="data row8 col5" >0.131422</td>
      <td id="T_6bf28_row8_col6" class="data row8 col6" >-0.038251</td>
      <td id="T_6bf28_row8_col7" class="data row8 col7" >-0.347680</td>
      <td id="T_6bf28_row8_col8" class="data row8 col8" >1.000000</td>
      <td id="T_6bf28_row8_col9" class="data row8 col9" >0.496119</td>
      <td id="T_6bf28_row8_col10" class="data row8 col10" >0.299690</td>
      <td id="T_6bf28_row8_col11" class="data row8 col11" >-0.131863</td>
      <td id="T_6bf28_row8_col12" class="data row8 col12" >-0.263039</td>
      <td id="T_6bf28_row8_col13" class="data row8 col13" >0.597813</td>
      <td id="T_6bf28_row8_col14" class="data row8 col14" >0.269544</td>
      <td id="T_6bf28_row8_col15" class="data row8 col15" >-0.311848</td>
      <td id="T_6bf28_row8_col16" class="data row8 col16" >1.000000</td>
      <td id="T_6bf28_row8_col17" class="data row8 col17" >0.496131</td>
      <td id="T_6bf28_row8_col18" class="data row8 col18" >0.246122</td>
      <td id="T_6bf28_row8_col19" class="data row8 col19" >0.268795</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row9" class="row_heading level0 row9" >블루 LED동작강도</th>
      <td id="T_6bf28_row9_col0" class="data row9 col0" >-0.412055</td>
      <td id="T_6bf28_row9_col1" class="data row9 col1" >-0.520013</td>
      <td id="T_6bf28_row9_col2" class="data row9 col2" >0.189064</td>
      <td id="T_6bf28_row9_col3" class="data row9 col3" >-0.062676</td>
      <td id="T_6bf28_row9_col4" class="data row9 col4" >0.077924</td>
      <td id="T_6bf28_row9_col5" class="data row9 col5" >0.357484</td>
      <td id="T_6bf28_row9_col6" class="data row9 col6" >0.245956</td>
      <td id="T_6bf28_row9_col7" class="data row9 col7" >0.354309</td>
      <td id="T_6bf28_row9_col8" class="data row9 col8" >0.496119</td>
      <td id="T_6bf28_row9_col9" class="data row9 col9" >1.000000</td>
      <td id="T_6bf28_row9_col10" class="data row9 col10" >-0.103579</td>
      <td id="T_6bf28_row9_col11" class="data row9 col11" >-0.269058</td>
      <td id="T_6bf28_row9_col12" class="data row9 col12" >-0.053375</td>
      <td id="T_6bf28_row9_col13" class="data row9 col13" >-0.005121</td>
      <td id="T_6bf28_row9_col14" class="data row9 col14" >0.814108</td>
      <td id="T_6bf28_row9_col15" class="data row9 col15" >0.360140</td>
      <td id="T_6bf28_row9_col16" class="data row9 col16" >0.496143</td>
      <td id="T_6bf28_row9_col17" class="data row9 col17" >1.000000</td>
      <td id="T_6bf28_row9_col18" class="data row9 col18" >0.048466</td>
      <td id="T_6bf28_row9_col19" class="data row9 col19" >0.060023</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row10" class="row_heading level0 row10" >냉방온도</th>
      <td id="T_6bf28_row10_col0" class="data row10 col0" >0.028192</td>
      <td id="T_6bf28_row10_col1" class="data row10 col1" >-0.017607</td>
      <td id="T_6bf28_row10_col2" class="data row10 col2" >-0.040844</td>
      <td id="T_6bf28_row10_col3" class="data row10 col3" >0.002730</td>
      <td id="T_6bf28_row10_col4" class="data row10 col4" >-0.163204</td>
      <td id="T_6bf28_row10_col5" class="data row10 col5" >0.087924</td>
      <td id="T_6bf28_row10_col6" class="data row10 col6" >-0.136779</td>
      <td id="T_6bf28_row10_col7" class="data row10 col7" >-0.359168</td>
      <td id="T_6bf28_row10_col8" class="data row10 col8" >0.299690</td>
      <td id="T_6bf28_row10_col9" class="data row10 col9" >-0.103579</td>
      <td id="T_6bf28_row10_col10" class="data row10 col10" >1.000000</td>
      <td id="T_6bf28_row10_col11" class="data row10 col11" >-0.491072</td>
      <td id="T_6bf28_row10_col12" class="data row10 col12" >0.581471</td>
      <td id="T_6bf28_row10_col13" class="data row10 col13" >0.372114</td>
      <td id="T_6bf28_row10_col14" class="data row10 col14" >-0.187032</td>
      <td id="T_6bf28_row10_col15" class="data row10 col15" >-0.318933</td>
      <td id="T_6bf28_row10_col16" class="data row10 col16" >0.299638</td>
      <td id="T_6bf28_row10_col17" class="data row10 col17" >-0.103634</td>
      <td id="T_6bf28_row10_col18" class="data row10 col18" >0.095593</td>
      <td id="T_6bf28_row10_col19" class="data row10 col19" >0.101461</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row11" class="row_heading level0 row11" >냉방부하</th>
      <td id="T_6bf28_row11_col0" class="data row11 col0" >0.608341</td>
      <td id="T_6bf28_row11_col1" class="data row11 col1" >0.596009</td>
      <td id="T_6bf28_row11_col2" class="data row11 col2" >-0.185409</td>
      <td id="T_6bf28_row11_col3" class="data row11 col3" >-0.031856</td>
      <td id="T_6bf28_row11_col4" class="data row11 col4" >0.201229</td>
      <td id="T_6bf28_row11_col5" class="data row11 col5" >-0.026613</td>
      <td id="T_6bf28_row11_col6" class="data row11 col6" >-0.082902</td>
      <td id="T_6bf28_row11_col7" class="data row11 col7" >0.023614</td>
      <td id="T_6bf28_row11_col8" class="data row11 col8" >-0.131863</td>
      <td id="T_6bf28_row11_col9" class="data row11 col9" >-0.269058</td>
      <td id="T_6bf28_row11_col10" class="data row11 col10" >-0.491072</td>
      <td id="T_6bf28_row11_col11" class="data row11 col11" >1.000000</td>
      <td id="T_6bf28_row11_col12" class="data row11 col12" >-0.535039</td>
      <td id="T_6bf28_row11_col13" class="data row11 col13" >0.096505</td>
      <td id="T_6bf28_row11_col14" class="data row11 col14" >-0.173651</td>
      <td id="T_6bf28_row11_col15" class="data row11 col15" >-0.037412</td>
      <td id="T_6bf28_row11_col16" class="data row11 col16" >-0.131821</td>
      <td id="T_6bf28_row11_col17" class="data row11 col17" >-0.269003</td>
      <td id="T_6bf28_row11_col18" class="data row11 col18" >-0.042442</td>
      <td id="T_6bf28_row11_col19" class="data row11 col19" >-0.064333</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row12" class="row_heading level0 row12" >난방온도</th>
      <td id="T_6bf28_row12_col0" class="data row12 col0" >0.300661</td>
      <td id="T_6bf28_row12_col1" class="data row12 col1" >0.229053</td>
      <td id="T_6bf28_row12_col2" class="data row12 col2" >-0.344317</td>
      <td id="T_6bf28_row12_col3" class="data row12 col3" >-0.250606</td>
      <td id="T_6bf28_row12_col4" class="data row12 col4" >-0.116181</td>
      <td id="T_6bf28_row12_col5" class="data row12 col5" >0.256154</td>
      <td id="T_6bf28_row12_col6" class="data row12 col6" >-0.118560</td>
      <td id="T_6bf28_row12_col7" class="data row12 col7" >-0.003158</td>
      <td id="T_6bf28_row12_col8" class="data row12 col8" >-0.263039</td>
      <td id="T_6bf28_row12_col9" class="data row12 col9" >-0.053375</td>
      <td id="T_6bf28_row12_col10" class="data row12 col10" >0.581471</td>
      <td id="T_6bf28_row12_col11" class="data row12 col11" >-0.535039</td>
      <td id="T_6bf28_row12_col12" class="data row12 col12" >1.000000</td>
      <td id="T_6bf28_row12_col13" class="data row12 col13" >-0.374729</td>
      <td id="T_6bf28_row12_col14" class="data row12 col14" >-0.097308</td>
      <td id="T_6bf28_row12_col15" class="data row12 col15" >0.007400</td>
      <td id="T_6bf28_row12_col16" class="data row12 col16" >-0.263096</td>
      <td id="T_6bf28_row12_col17" class="data row12 col17" >-0.053433</td>
      <td id="T_6bf28_row12_col18" class="data row12 col18" >-0.033694</td>
      <td id="T_6bf28_row12_col19" class="data row12 col19" >-0.034794</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row13" class="row_heading level0 row13" >난방부하</th>
      <td id="T_6bf28_row13_col0" class="data row13 col0" >-0.250316</td>
      <td id="T_6bf28_row13_col1" class="data row13 col1" >-0.270891</td>
      <td id="T_6bf28_row13_col2" class="data row13 col2" >0.194068</td>
      <td id="T_6bf28_row13_col3" class="data row13 col3" >0.225217</td>
      <td id="T_6bf28_row13_col4" class="data row13 col4" >-0.109370</td>
      <td id="T_6bf28_row13_col5" class="data row13 col5" >-0.057690</td>
      <td id="T_6bf28_row13_col6" class="data row13 col6" >-0.130924</td>
      <td id="T_6bf28_row13_col7" class="data row13 col7" >-0.383132</td>
      <td id="T_6bf28_row13_col8" class="data row13 col8" >0.597813</td>
      <td id="T_6bf28_row13_col9" class="data row13 col9" >-0.005121</td>
      <td id="T_6bf28_row13_col10" class="data row13 col10" >0.372114</td>
      <td id="T_6bf28_row13_col11" class="data row13 col11" >0.096505</td>
      <td id="T_6bf28_row13_col12" class="data row13 col12" >-0.374729</td>
      <td id="T_6bf28_row13_col13" class="data row13 col13" >1.000000</td>
      <td id="T_6bf28_row13_col14" class="data row13 col14" >-0.088278</td>
      <td id="T_6bf28_row13_col15" class="data row13 col15" >-0.362116</td>
      <td id="T_6bf28_row13_col16" class="data row13 col16" >0.597797</td>
      <td id="T_6bf28_row13_col17" class="data row13 col17" >-0.005124</td>
      <td id="T_6bf28_row13_col18" class="data row13 col18" >0.200674</td>
      <td id="T_6bf28_row13_col19" class="data row13 col19" >0.206683</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row14" class="row_heading level0 row14" >총추정광량</th>
      <td id="T_6bf28_row14_col0" class="data row14 col0" >-0.341991</td>
      <td id="T_6bf28_row14_col1" class="data row14 col1" >-0.439107</td>
      <td id="T_6bf28_row14_col2" class="data row14 col2" >0.135976</td>
      <td id="T_6bf28_row14_col3" class="data row14 col3" >-0.102444</td>
      <td id="T_6bf28_row14_col4" class="data row14 col4" >0.140163</td>
      <td id="T_6bf28_row14_col5" class="data row14 col5" >0.349921</td>
      <td id="T_6bf28_row14_col6" class="data row14 col6" >0.283771</td>
      <td id="T_6bf28_row14_col7" class="data row14 col7" >0.745572</td>
      <td id="T_6bf28_row14_col8" class="data row14 col8" >0.269544</td>
      <td id="T_6bf28_row14_col9" class="data row14 col9" >0.814108</td>
      <td id="T_6bf28_row14_col10" class="data row14 col10" >-0.187032</td>
      <td id="T_6bf28_row14_col11" class="data row14 col11" >-0.173651</td>
      <td id="T_6bf28_row14_col12" class="data row14 col12" >-0.097308</td>
      <td id="T_6bf28_row14_col13" class="data row14 col13" >-0.088278</td>
      <td id="T_6bf28_row14_col14" class="data row14 col14" >1.000000</td>
      <td id="T_6bf28_row14_col15" class="data row14 col15" >0.796136</td>
      <td id="T_6bf28_row14_col16" class="data row14 col16" >0.269591</td>
      <td id="T_6bf28_row14_col17" class="data row14 col17" >0.814124</td>
      <td id="T_6bf28_row14_col18" class="data row14 col18" >-0.134164</td>
      <td id="T_6bf28_row14_col19" class="data row14 col19" >-0.133084</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row15" class="row_heading level0 row15" >백색광추정광량</th>
      <td id="T_6bf28_row15_col0" class="data row15 col0" >-0.089971</td>
      <td id="T_6bf28_row15_col1" class="data row15 col1" >-0.151600</td>
      <td id="T_6bf28_row15_col2" class="data row15 col2" >-0.031516</td>
      <td id="T_6bf28_row15_col3" class="data row15 col3" >-0.178893</td>
      <td id="T_6bf28_row15_col4" class="data row15 col4" >0.153977</td>
      <td id="T_6bf28_row15_col5" class="data row15 col5" >0.225883</td>
      <td id="T_6bf28_row15_col6" class="data row15 col6" >0.265799</td>
      <td id="T_6bf28_row15_col7" class="data row15 col7" >0.950771</td>
      <td id="T_6bf28_row15_col8" class="data row15 col8" >-0.311848</td>
      <td id="T_6bf28_row15_col9" class="data row15 col9" >0.360140</td>
      <td id="T_6bf28_row15_col10" class="data row15 col10" >-0.318933</td>
      <td id="T_6bf28_row15_col11" class="data row15 col11" >-0.037412</td>
      <td id="T_6bf28_row15_col12" class="data row15 col12" >0.007400</td>
      <td id="T_6bf28_row15_col13" class="data row15 col13" >-0.362116</td>
      <td id="T_6bf28_row15_col14" class="data row15 col14" >0.796136</td>
      <td id="T_6bf28_row15_col15" class="data row15 col15" >1.000000</td>
      <td id="T_6bf28_row15_col16" class="data row15 col16" >-0.311798</td>
      <td id="T_6bf28_row15_col17" class="data row15 col17" >0.360155</td>
      <td id="T_6bf28_row15_col18" class="data row15 col18" >-0.304854</td>
      <td id="T_6bf28_row15_col19" class="data row15 col19" >-0.318758</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row16" class="row_heading level0 row16" >적색광추정광량</th>
      <td id="T_6bf28_row16_col0" class="data row16 col0" >-0.366396</td>
      <td id="T_6bf28_row16_col1" class="data row16 col1" >-0.394598</td>
      <td id="T_6bf28_row16_col2" class="data row16 col2" >0.276777</td>
      <td id="T_6bf28_row16_col3" class="data row16 col3" >0.181610</td>
      <td id="T_6bf28_row16_col4" class="data row16 col4" >-0.019634</td>
      <td id="T_6bf28_row16_col5" class="data row16 col5" >0.131423</td>
      <td id="T_6bf28_row16_col6" class="data row16 col6" >-0.038245</td>
      <td id="T_6bf28_row16_col7" class="data row16 col7" >-0.347626</td>
      <td id="T_6bf28_row16_col8" class="data row16 col8" >1.000000</td>
      <td id="T_6bf28_row16_col9" class="data row16 col9" >0.496143</td>
      <td id="T_6bf28_row16_col10" class="data row16 col10" >0.299638</td>
      <td id="T_6bf28_row16_col11" class="data row16 col11" >-0.131821</td>
      <td id="T_6bf28_row16_col12" class="data row16 col12" >-0.263096</td>
      <td id="T_6bf28_row16_col13" class="data row16 col13" >0.597797</td>
      <td id="T_6bf28_row16_col14" class="data row16 col14" >0.269591</td>
      <td id="T_6bf28_row16_col15" class="data row16 col15" >-0.311798</td>
      <td id="T_6bf28_row16_col16" class="data row16 col16" >1.000000</td>
      <td id="T_6bf28_row16_col17" class="data row16 col17" >0.496156</td>
      <td id="T_6bf28_row16_col18" class="data row16 col18" >0.246135</td>
      <td id="T_6bf28_row16_col19" class="data row16 col19" >0.268809</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row17" class="row_heading level0 row17" >청색광추정광량</th>
      <td id="T_6bf28_row17_col0" class="data row17 col0" >-0.412047</td>
      <td id="T_6bf28_row17_col1" class="data row17 col1" >-0.519996</td>
      <td id="T_6bf28_row17_col2" class="data row17 col2" >0.189070</td>
      <td id="T_6bf28_row17_col3" class="data row17 col3" >-0.062680</td>
      <td id="T_6bf28_row17_col4" class="data row17 col4" >0.077936</td>
      <td id="T_6bf28_row17_col5" class="data row17 col5" >0.357497</td>
      <td id="T_6bf28_row17_col6" class="data row17 col6" >0.245923</td>
      <td id="T_6bf28_row17_col7" class="data row17 col7" >0.354327</td>
      <td id="T_6bf28_row17_col8" class="data row17 col8" >0.496131</td>
      <td id="T_6bf28_row17_col9" class="data row17 col9" >1.000000</td>
      <td id="T_6bf28_row17_col10" class="data row17 col10" >-0.103634</td>
      <td id="T_6bf28_row17_col11" class="data row17 col11" >-0.269003</td>
      <td id="T_6bf28_row17_col12" class="data row17 col12" >-0.053433</td>
      <td id="T_6bf28_row17_col13" class="data row17 col13" >-0.005124</td>
      <td id="T_6bf28_row17_col14" class="data row17 col14" >0.814124</td>
      <td id="T_6bf28_row17_col15" class="data row17 col15" >0.360155</td>
      <td id="T_6bf28_row17_col16" class="data row17 col16" >0.496156</td>
      <td id="T_6bf28_row17_col17" class="data row17 col17" >1.000000</td>
      <td id="T_6bf28_row17_col18" class="data row17 col18" >0.048483</td>
      <td id="T_6bf28_row17_col19" class="data row17 col19" >0.060043</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row18" class="row_heading level0 row18" >무게</th>
      <td id="T_6bf28_row18_col0" class="data row18 col0" >-0.096137</td>
      <td id="T_6bf28_row18_col1" class="data row18 col1" >-0.005515</td>
      <td id="T_6bf28_row18_col2" class="data row18 col2" >0.225323</td>
      <td id="T_6bf28_row18_col3" class="data row18 col3" >0.117638</td>
      <td id="T_6bf28_row18_col4" class="data row18 col4" >0.024076</td>
      <td id="T_6bf28_row18_col5" class="data row18 col5" >-0.153996</td>
      <td id="T_6bf28_row18_col6" class="data row18 col6" >-0.284305</td>
      <td id="T_6bf28_row18_col7" class="data row18 col7" >-0.343482</td>
      <td id="T_6bf28_row18_col8" class="data row18 col8" >0.246122</td>
      <td id="T_6bf28_row18_col9" class="data row18 col9" >0.048466</td>
      <td id="T_6bf28_row18_col10" class="data row18 col10" >0.095593</td>
      <td id="T_6bf28_row18_col11" class="data row18 col11" >-0.042442</td>
      <td id="T_6bf28_row18_col12" class="data row18 col12" >-0.033694</td>
      <td id="T_6bf28_row18_col13" class="data row18 col13" >0.200674</td>
      <td id="T_6bf28_row18_col14" class="data row18 col14" >-0.134164</td>
      <td id="T_6bf28_row18_col15" class="data row18 col15" >-0.304854</td>
      <td id="T_6bf28_row18_col16" class="data row18 col16" >0.246135</td>
      <td id="T_6bf28_row18_col17" class="data row18 col17" >0.048483</td>
      <td id="T_6bf28_row18_col18" class="data row18 col18" >1.000000</td>
      <td id="T_6bf28_row18_col19" class="data row18 col19" >0.992092</td>
    </tr>
    <tr>
      <th id="T_6bf28_level0_row19" class="row_heading level0 row19" >비율</th>
      <td id="T_6bf28_row19_col0" class="data row19 col0" >-0.122599</td>
      <td id="T_6bf28_row19_col1" class="data row19 col1" >-0.029887</td>
      <td id="T_6bf28_row19_col2" class="data row19 col2" >0.270407</td>
      <td id="T_6bf28_row19_col3" class="data row19 col3" >0.141438</td>
      <td id="T_6bf28_row19_col4" class="data row19 col4" >0.009779</td>
      <td id="T_6bf28_row19_col5" class="data row19 col5" >-0.171411</td>
      <td id="T_6bf28_row19_col6" class="data row19 col6" >-0.276562</td>
      <td id="T_6bf28_row19_col7" class="data row19 col7" >-0.360687</td>
      <td id="T_6bf28_row19_col8" class="data row19 col8" >0.268795</td>
      <td id="T_6bf28_row19_col9" class="data row19 col9" >0.060023</td>
      <td id="T_6bf28_row19_col10" class="data row19 col10" >0.101461</td>
      <td id="T_6bf28_row19_col11" class="data row19 col11" >-0.064333</td>
      <td id="T_6bf28_row19_col12" class="data row19 col12" >-0.034794</td>
      <td id="T_6bf28_row19_col13" class="data row19 col13" >0.206683</td>
      <td id="T_6bf28_row19_col14" class="data row19 col14" >-0.133084</td>
      <td id="T_6bf28_row19_col15" class="data row19 col15" >-0.318758</td>
      <td id="T_6bf28_row19_col16" class="data row19 col16" >0.268809</td>
      <td id="T_6bf28_row19_col17" class="data row19 col17" >0.060043</td>
      <td id="T_6bf28_row19_col18" class="data row19 col18" >0.992092</td>
      <td id="T_6bf28_row19_col19" class="data row19 col19" >1.000000</td>
    </tr>
  </tbody>
</table>





```python
sns.scatterplot(train_df_meta, x=train_df_meta["총추정광량"], y=train_df_meta["백색광추정광량"]+train_df_meta["적색광추정광량"]+train_df_meta["청색광추정광량"]);
```


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/optimizing_the_growth_env_files/optimizing_the_growth_env_36_0.png?raw=true)
    


### CatBoost


```python
features = ['내부온도관측치', '외부온도관측치', '내부습도관측치', '외부습도관측치', 'CO2관측치', 'EC관측치',
         '최근분무량', '냉방온도', '냉방부하',
         '난방온도', '난방부하', '백색광추정광량', '적색광추정광량', '청색광추정광량', '비율']

train_col = train_df_meta[features]

test_col = test_df[features]

train_target = train_df_meta["무게"]

train_x, val_x, train_y, val_y = train_test_split(train_col, train_target, test_size=0.2, random_state=32)
```


```python
CATmodel = CatBoostRegressor(verbose=50, n_estimators=10000, eval_metric="MAE", early_stopping_rounds=50)
CATmodel.fit(train_x, train_y, eval_set=[(val_x, val_y)], use_best_model=True)

val_pred = CATmodel.predict(val_x)
plt.figure(figsize=(20,10))
plt.plot(np.array(val_pred),label = "pred")
plt.plot(np.array(val_y),label = "true")
plt.legend()
plt.show()

train_score = CATmodel.score(train_x, train_y)
val_score = CATmodel.score(val_x, val_y)
```

    Learning rate set to 0.012542
    0:	learn: 81.6377352	test: 85.8903420	best: 85.8903420 (0)	total: 147ms	remaining: 24m 30s
    50:	learn: 47.6997804	test: 51.2987982	best: 51.2987982 (50)	total: 233ms	remaining: 45.5s
    ...
    6300:	learn: 0.4522590	test: 3.1227911	best: 3.1225241 (6295)	total: 12.2s	remaining: 7.17s
    6350:	learn: 0.4476624	test: 3.1219959	best: 3.1219880 (6349)	total: 12.3s	remaining: 7.07s
    6400:	learn: 0.4438297	test: 3.1213585	best: 3.1212632 (6383)	total: 12.4s	remaining: 6.96s
    Stopped by overfitting detector  (50 iterations wait)
    
    bestTest = 3.121263223
    bestIteration = 6383
    
    Shrink model to first 6384 iterations.
    


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/optimizing_the_growth_env_files/optimizing_the_growth_env_39_1.png?raw=true)
    



```python
CATresult = CATmodel.predict(test_col)

submission = pd.read_csv('./open/sample_submission.csv')
submission['leaf_weight'] = CATresult
submission.to_csv('./CATsubmit.csv', index=False)
```

### ANN
- sklearn으로 메타 데이터 스케일 조정


```python
def scale_datasets(x_train, x_test):
    # Z - Score
    standard_scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        standard_scaler.fit_transform(x_train),
        columns=x_train.columns
    )
    x_test_scaled = pd.DataFrame(
        standard_scaler.transform(x_test),
        columns = x_test.columns
    )
    return x_train_scaled, x_test_scaled

train_scaled, test_scaled = scale_datasets(train_col, test_col)

train_x_scale, val_x_scale, train_y_scale, val_y_scale = train_test_split(train_scaled, train_target, test_size=0.2, random_state=32)
```


```python
tf.random.set_seed(42)

def build_model_using_sequential():
    model = Sequential([
      Dense(100, kernel_initializer='normal', activation='relu'),
      Dense(50, kernel_initializer='normal', activation='relu'),
      Dense(25, kernel_initializer='normal', activation='relu'),
      Dense(1, kernel_initializer='normal', activation='linear')
    ])
    return model

ANNmodel = build_model_using_sequential()

# Loss Func.
mae = MeanAbsoluteError()
ANNmodel.compile(
    loss=mae, 
    optimizer=Adam(learning_rate=0.001), 
    metrics=[mae]
)

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=50,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# train the model
history = ANNmodel.fit(
    train_x_scale, 
    train_y_scale, 
    epochs=1000, 
    batch_size=32,
    validation_data=(val_x_scale, val_y_scale),
    callbacks=[early_stopping_monitor],
    verbose= 2
)
```

    Epoch 1/1000
    32/32 - 1s - loss: 79.6819 - mean_absolute_error: 79.6508 - val_loss: 80.7002 - val_mean_absolute_error: 80.7002 - 1s/epoch - 34ms/step
    Epoch 2/1000
    32/32 - 0s - loss: 78.0658 - mean_absolute_error: 78.0423 - val_loss: 77.3219 - val_mean_absolute_error: 77.3219 - 129ms/epoch - 4ms/step
    ...
    Epoch 301/1000
    Restoring model weights from the end of the best epoch: 251.
    32/32 - 0s - loss: 2.7592 - mean_absolute_error: 2.7637 - val_loss: 3.2672 - val_mean_absolute_error: 3.2672 - 171ms/epoch - 5ms/step
    Epoch 301: early stopping
    


```python
val_pred = ANNmodel.predict(val_x_scale)
plt.figure(figsize=(20,10))
plt.plot(np.array(val_pred),label = "pred")
plt.plot(np.array(val_y_scale),label = "true")
plt.legend()
plt.show()
```

    8/8 [==============================] - 0s 2ms/step
    


    
![png](https://github.com/nuyhc/github.io.archives/blob/main/optimizing_the_growth_env_files/optimizing_the_growth_env_44_1.png?raw=true)
    



```python
ANNresult = ANNmodel.predict(test_scaled)

submission = pd.read_csv('./open/sample_submission.csv')
submission['leaf_weight'] = ANNresult
submission.to_csv('./ANNsubmit.csv', index=False)
```

    15/15 [==============================] - 0s 2ms/step
    

### Ensemble


```python
CNN = pd.read_csv('./CNNsubmit.csv')
CAT = pd.read_csv('./CATsubmit.csv')
ANN = pd.read_csv('./ANNsubmit.csv')

submission_final = pd.read_csv('./open/sample_submission.csv')
submission_final['leaf_weight'] = (CNN['leaf_weight'] * 0.65 + CAT['leaf_weight'] * 0.25 + ANN['leaf_weight'] * 0.1)
submission_final.to_csv('ENSEMBLEsubmit.csv', index=False)
```

### 마무리
- 이미지 데이터와 메타 데이터가 함께 있는 경우 어떻게 전처리를 하고 사용하는지 의아했는데, 해당 필사를 통해 일부 궁금증을 해소할 수 있었음
