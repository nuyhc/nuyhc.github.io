---
title: PyCaret 맛보기
date: 2022-11-26T09:23:28.272Z

categories:
  - Programming
  - Machine Learning
tags:
  - PyCaret
  - Tutorial
---

# 피마 인디언 데이터셋 with PyCaret


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
```

### Data Load
피마 인디언 당뇨병 데이터 셋


```python
df_pima = pd.read_csv("http://bit.ly/data-diabetes-csv")
df_pima.shape
```




    (768, 9)



### PyCaret
당뇨병 여부 분류 문제 적용시


```python
from pycaret.classification import *
```

#### setup
Train data, Test data, Label, Target 등을 설정하는 부분이며, 데이터에 전처리 기법들을 적용 할 수 있음


```python
pycaret_models = setup(
    session_id=42, # 랜덤 시드
    data=df_pima, # Input Data
    target="Outcome", # Target
    normalize=True, # 정규화 여부
    normalize_method="minmax", # 정규화 방식
    transformation=True, # 데이터의 분포가 정규 분포에 더 가까워지도록 처리
    fold_strategy="stratifiedkfold",
    use_gpu=True
)
```


<style type="text/css">
#T_cf367_row18_col1, #T_cf367_row29_col1, #T_cf367_row31_col1, #T_cf367_row44_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_cf367">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cf367_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_cf367_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cf367_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cf367_row0_col0" class="data row0 col0" >session_id</td>
      <td id="T_cf367_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cf367_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_cf367_row1_col1" class="data row1 col1" >Outcome</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cf367_row2_col0" class="data row2 col0" >Target Type</td>
      <td id="T_cf367_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cf367_row3_col0" class="data row3 col0" >Label Encoded</td>
      <td id="T_cf367_row3_col1" class="data row3 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cf367_row4_col0" class="data row4 col0" >Original Data</td>
      <td id="T_cf367_row4_col1" class="data row4 col1" >(768, 9)</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cf367_row5_col0" class="data row5 col0" >Missing Values</td>
      <td id="T_cf367_row5_col1" class="data row5 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cf367_row6_col0" class="data row6 col0" >Numeric Features</td>
      <td id="T_cf367_row6_col1" class="data row6 col1" >7</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cf367_row7_col0" class="data row7 col0" >Categorical Features</td>
      <td id="T_cf367_row7_col1" class="data row7 col1" >1</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cf367_row8_col0" class="data row8 col0" >Ordinal Features</td>
      <td id="T_cf367_row8_col1" class="data row8 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cf367_row9_col0" class="data row9 col0" >High Cardinality Features</td>
      <td id="T_cf367_row9_col1" class="data row9 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_cf367_row10_col0" class="data row10 col0" >High Cardinality Method</td>
      <td id="T_cf367_row10_col1" class="data row10 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_cf367_row11_col0" class="data row11 col0" >Transformed Train Set</td>
      <td id="T_cf367_row11_col1" class="data row11 col1" >(537, 24)</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_cf367_row12_col0" class="data row12 col0" >Transformed Test Set</td>
      <td id="T_cf367_row12_col1" class="data row12 col1" >(231, 24)</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_cf367_row13_col0" class="data row13 col0" >Shuffle Train-Test</td>
      <td id="T_cf367_row13_col1" class="data row13 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_cf367_row14_col0" class="data row14 col0" >Stratify Train-Test</td>
      <td id="T_cf367_row14_col1" class="data row14 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_cf367_row15_col0" class="data row15 col0" >Fold Generator</td>
      <td id="T_cf367_row15_col1" class="data row15 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_cf367_row16_col0" class="data row16 col0" >Fold Number</td>
      <td id="T_cf367_row16_col1" class="data row16 col1" >10</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_cf367_row17_col0" class="data row17 col0" >CPU Jobs</td>
      <td id="T_cf367_row17_col1" class="data row17 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_cf367_row18_col0" class="data row18 col0" >Use GPU</td>
      <td id="T_cf367_row18_col1" class="data row18 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_cf367_row19_col0" class="data row19 col0" >Log Experiment</td>
      <td id="T_cf367_row19_col1" class="data row19 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_cf367_row20_col0" class="data row20 col0" >Experiment Name</td>
      <td id="T_cf367_row20_col1" class="data row20 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_cf367_row21_col0" class="data row21 col0" >USI</td>
      <td id="T_cf367_row21_col1" class="data row21 col1" >d7e1</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_cf367_row22_col0" class="data row22 col0" >Imputation Type</td>
      <td id="T_cf367_row22_col1" class="data row22 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_cf367_row23_col0" class="data row23 col0" >Iterative Imputation Iteration</td>
      <td id="T_cf367_row23_col1" class="data row23 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_cf367_row24_col0" class="data row24 col0" >Numeric Imputer</td>
      <td id="T_cf367_row24_col1" class="data row24 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_cf367_row25_col0" class="data row25 col0" >Iterative Imputation Numeric Model</td>
      <td id="T_cf367_row25_col1" class="data row25 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_cf367_row26_col0" class="data row26 col0" >Categorical Imputer</td>
      <td id="T_cf367_row26_col1" class="data row26 col1" >constant</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_cf367_row27_col0" class="data row27 col0" >Iterative Imputation Categorical Model</td>
      <td id="T_cf367_row27_col1" class="data row27 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_cf367_row28_col0" class="data row28 col0" >Unknown Categoricals Handling</td>
      <td id="T_cf367_row28_col1" class="data row28 col1" >least_frequent</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_cf367_row29_col0" class="data row29 col0" >Normalize</td>
      <td id="T_cf367_row29_col1" class="data row29 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_cf367_row30_col0" class="data row30 col0" >Normalize Method</td>
      <td id="T_cf367_row30_col1" class="data row30 col1" >minmax</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_cf367_row31_col0" class="data row31 col0" >Transformation</td>
      <td id="T_cf367_row31_col1" class="data row31 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_cf367_row32_col0" class="data row32 col0" >Transformation Method</td>
      <td id="T_cf367_row32_col1" class="data row32 col1" >yeo-johnson</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_cf367_row33_col0" class="data row33 col0" >PCA</td>
      <td id="T_cf367_row33_col1" class="data row33 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_cf367_row34_col0" class="data row34 col0" >PCA Method</td>
      <td id="T_cf367_row34_col1" class="data row34 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_cf367_row35_col0" class="data row35 col0" >PCA Components</td>
      <td id="T_cf367_row35_col1" class="data row35 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_cf367_row36_col0" class="data row36 col0" >Ignore Low Variance</td>
      <td id="T_cf367_row36_col1" class="data row36 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_cf367_row37_col0" class="data row37 col0" >Combine Rare Levels</td>
      <td id="T_cf367_row37_col1" class="data row37 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_cf367_row38_col0" class="data row38 col0" >Rare Level Threshold</td>
      <td id="T_cf367_row38_col1" class="data row38 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_cf367_row39_col0" class="data row39 col0" >Numeric Binning</td>
      <td id="T_cf367_row39_col1" class="data row39 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row40" class="row_heading level0 row40" >40</th>
      <td id="T_cf367_row40_col0" class="data row40 col0" >Remove Outliers</td>
      <td id="T_cf367_row40_col1" class="data row40 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row41" class="row_heading level0 row41" >41</th>
      <td id="T_cf367_row41_col0" class="data row41 col0" >Outliers Threshold</td>
      <td id="T_cf367_row41_col1" class="data row41 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row42" class="row_heading level0 row42" >42</th>
      <td id="T_cf367_row42_col0" class="data row42 col0" >Remove Multicollinearity</td>
      <td id="T_cf367_row42_col1" class="data row42 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row43" class="row_heading level0 row43" >43</th>
      <td id="T_cf367_row43_col0" class="data row43 col0" >Multicollinearity Threshold</td>
      <td id="T_cf367_row43_col1" class="data row43 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row44" class="row_heading level0 row44" >44</th>
      <td id="T_cf367_row44_col0" class="data row44 col0" >Remove Perfect Collinearity</td>
      <td id="T_cf367_row44_col1" class="data row44 col1" >True</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row45" class="row_heading level0 row45" >45</th>
      <td id="T_cf367_row45_col0" class="data row45 col0" >Clustering</td>
      <td id="T_cf367_row45_col1" class="data row45 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row46" class="row_heading level0 row46" >46</th>
      <td id="T_cf367_row46_col0" class="data row46 col0" >Clustering Iteration</td>
      <td id="T_cf367_row46_col1" class="data row46 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row47" class="row_heading level0 row47" >47</th>
      <td id="T_cf367_row47_col0" class="data row47 col0" >Polynomial Features</td>
      <td id="T_cf367_row47_col1" class="data row47 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row48" class="row_heading level0 row48" >48</th>
      <td id="T_cf367_row48_col0" class="data row48 col0" >Polynomial Degree</td>
      <td id="T_cf367_row48_col1" class="data row48 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row49" class="row_heading level0 row49" >49</th>
      <td id="T_cf367_row49_col0" class="data row49 col0" >Trignometry Features</td>
      <td id="T_cf367_row49_col1" class="data row49 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row50" class="row_heading level0 row50" >50</th>
      <td id="T_cf367_row50_col0" class="data row50 col0" >Polynomial Threshold</td>
      <td id="T_cf367_row50_col1" class="data row50 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row51" class="row_heading level0 row51" >51</th>
      <td id="T_cf367_row51_col0" class="data row51 col0" >Group Features</td>
      <td id="T_cf367_row51_col1" class="data row51 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row52" class="row_heading level0 row52" >52</th>
      <td id="T_cf367_row52_col0" class="data row52 col0" >Feature Selection</td>
      <td id="T_cf367_row52_col1" class="data row52 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row53" class="row_heading level0 row53" >53</th>
      <td id="T_cf367_row53_col0" class="data row53 col0" >Feature Selection Method</td>
      <td id="T_cf367_row53_col1" class="data row53 col1" >classic</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row54" class="row_heading level0 row54" >54</th>
      <td id="T_cf367_row54_col0" class="data row54 col0" >Features Selection Threshold</td>
      <td id="T_cf367_row54_col1" class="data row54 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row55" class="row_heading level0 row55" >55</th>
      <td id="T_cf367_row55_col0" class="data row55 col0" >Feature Interaction</td>
      <td id="T_cf367_row55_col1" class="data row55 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row56" class="row_heading level0 row56" >56</th>
      <td id="T_cf367_row56_col0" class="data row56 col0" >Feature Ratio</td>
      <td id="T_cf367_row56_col1" class="data row56 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row57" class="row_heading level0 row57" >57</th>
      <td id="T_cf367_row57_col0" class="data row57 col0" >Interaction Threshold</td>
      <td id="T_cf367_row57_col1" class="data row57 col1" >None</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row58" class="row_heading level0 row58" >58</th>
      <td id="T_cf367_row58_col0" class="data row58 col0" >Fix Imbalance</td>
      <td id="T_cf367_row58_col1" class="data row58 col1" >False</td>
    </tr>
    <tr>
      <th id="T_cf367_level0_row59" class="row_heading level0 row59" >59</th>
      <td id="T_cf367_row59_col0" class="data row59 col0" >Fix Imbalance Method</td>
      <td id="T_cf367_row59_col1" class="data row59 col1" >SMOTE</td>
    </tr>
  </tbody>
</table>



### models


```python
models_list = models()
```


```python
models_list
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
      <th>Name</th>
      <th>Reference</th>
      <th>Turbo</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lr</th>
      <td>Logistic Regression</td>
      <td>sklearn.linear_model._logistic.LogisticRegression</td>
      <td>True</td>
    </tr>
    <tr>
      <th>knn</th>
      <td>K Neighbors Classifier</td>
      <td>sklearn.neighbors._classification.KNeighborsCl...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>nb</th>
      <td>Naive Bayes</td>
      <td>sklearn.naive_bayes.GaussianNB</td>
      <td>True</td>
    </tr>
    <tr>
      <th>dt</th>
      <td>Decision Tree Classifier</td>
      <td>sklearn.tree._classes.DecisionTreeClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>svm</th>
      <td>SVM - Linear Kernel</td>
      <td>sklearn.linear_model._stochastic_gradient.SGDC...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>rbfsvm</th>
      <td>SVM - Radial Kernel</td>
      <td>sklearn.svm._classes.SVC</td>
      <td>False</td>
    </tr>
    <tr>
      <th>gpc</th>
      <td>Gaussian Process Classifier</td>
      <td>sklearn.gaussian_process._gpc.GaussianProcessC...</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mlp</th>
      <td>MLP Classifier</td>
      <td>sklearn.neural_network._multilayer_perceptron....</td>
      <td>False</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>Ridge Classifier</td>
      <td>sklearn.linear_model._ridge.RidgeClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>rf</th>
      <td>Random Forest Classifier</td>
      <td>sklearn.ensemble._forest.RandomForestClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>qda</th>
      <td>Quadratic Discriminant Analysis</td>
      <td>sklearn.discriminant_analysis.QuadraticDiscrim...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>ada</th>
      <td>Ada Boost Classifier</td>
      <td>sklearn.ensemble._weight_boosting.AdaBoostClas...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>gbc</th>
      <td>Gradient Boosting Classifier</td>
      <td>sklearn.ensemble._gb.GradientBoostingClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>lda</th>
      <td>Linear Discriminant Analysis</td>
      <td>sklearn.discriminant_analysis.LinearDiscrimina...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>et</th>
      <td>Extra Trees Classifier</td>
      <td>sklearn.ensemble._forest.ExtraTreesClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>lightgbm</th>
      <td>Light Gradient Boosting Machine</td>
      <td>lightgbm.sklearn.LGBMClassifier</td>
      <td>True</td>
    </tr>
    <tr>
      <th>dummy</th>
      <td>Dummy Classifier</td>
      <td>sklearn.dummy.DummyClassifier</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



`pycaret`에서 사용 가능한 모델 목록을 확인 할 수 있음

### compare_models


```python
pc_clf_models = compare_models(
    n_select=25, # 반환할 모델 개수
    include=models_list.index.tolist()
)
```


<style type="text/css">
#T_54b16 th {
  text-align: left;
}
#T_54b16_row0_col0, #T_54b16_row0_col2, #T_54b16_row0_col3, #T_54b16_row0_col5, #T_54b16_row0_col6, #T_54b16_row1_col0, #T_54b16_row1_col1, #T_54b16_row1_col2, #T_54b16_row1_col3, #T_54b16_row1_col4, #T_54b16_row1_col5, #T_54b16_row1_col7, #T_54b16_row2_col0, #T_54b16_row2_col1, #T_54b16_row2_col2, #T_54b16_row2_col3, #T_54b16_row2_col4, #T_54b16_row2_col5, #T_54b16_row2_col6, #T_54b16_row2_col7, #T_54b16_row3_col0, #T_54b16_row3_col1, #T_54b16_row3_col2, #T_54b16_row3_col3, #T_54b16_row3_col4, #T_54b16_row3_col5, #T_54b16_row3_col6, #T_54b16_row3_col7, #T_54b16_row4_col0, #T_54b16_row4_col1, #T_54b16_row4_col2, #T_54b16_row4_col3, #T_54b16_row4_col4, #T_54b16_row4_col5, #T_54b16_row4_col6, #T_54b16_row4_col7, #T_54b16_row5_col0, #T_54b16_row5_col1, #T_54b16_row5_col2, #T_54b16_row5_col4, #T_54b16_row5_col6, #T_54b16_row5_col7, #T_54b16_row6_col0, #T_54b16_row6_col1, #T_54b16_row6_col2, #T_54b16_row6_col3, #T_54b16_row6_col4, #T_54b16_row6_col5, #T_54b16_row6_col6, #T_54b16_row6_col7, #T_54b16_row7_col0, #T_54b16_row7_col1, #T_54b16_row7_col3, #T_54b16_row7_col4, #T_54b16_row7_col5, #T_54b16_row7_col6, #T_54b16_row7_col7, #T_54b16_row8_col0, #T_54b16_row8_col1, #T_54b16_row8_col2, #T_54b16_row8_col3, #T_54b16_row8_col4, #T_54b16_row8_col5, #T_54b16_row8_col6, #T_54b16_row8_col7, #T_54b16_row9_col0, #T_54b16_row9_col1, #T_54b16_row9_col2, #T_54b16_row9_col3, #T_54b16_row9_col4, #T_54b16_row9_col5, #T_54b16_row9_col6, #T_54b16_row9_col7, #T_54b16_row10_col0, #T_54b16_row10_col1, #T_54b16_row10_col2, #T_54b16_row10_col3, #T_54b16_row10_col4, #T_54b16_row10_col5, #T_54b16_row10_col6, #T_54b16_row10_col7, #T_54b16_row11_col0, #T_54b16_row11_col1, #T_54b16_row11_col2, #T_54b16_row11_col3, #T_54b16_row11_col4, #T_54b16_row11_col5, #T_54b16_row11_col6, #T_54b16_row11_col7, #T_54b16_row12_col0, #T_54b16_row12_col1, #T_54b16_row12_col2, #T_54b16_row12_col3, #T_54b16_row12_col4, #T_54b16_row12_col5, #T_54b16_row12_col6, #T_54b16_row12_col7, #T_54b16_row13_col0, #T_54b16_row13_col1, #T_54b16_row13_col2, #T_54b16_row13_col3, #T_54b16_row13_col4, #T_54b16_row13_col5, #T_54b16_row13_col6, #T_54b16_row13_col7, #T_54b16_row14_col0, #T_54b16_row14_col1, #T_54b16_row14_col2, #T_54b16_row14_col3, #T_54b16_row14_col4, #T_54b16_row14_col5, #T_54b16_row14_col6, #T_54b16_row14_col7, #T_54b16_row15_col0, #T_54b16_row15_col1, #T_54b16_row15_col2, #T_54b16_row15_col3, #T_54b16_row15_col4, #T_54b16_row15_col5, #T_54b16_row15_col6, #T_54b16_row15_col7, #T_54b16_row16_col0, #T_54b16_row16_col1, #T_54b16_row16_col2, #T_54b16_row16_col3, #T_54b16_row16_col4, #T_54b16_row16_col5, #T_54b16_row16_col6, #T_54b16_row16_col7 {
  text-align: left;
}
#T_54b16_row0_col1, #T_54b16_row0_col4, #T_54b16_row0_col7, #T_54b16_row1_col6, #T_54b16_row5_col3, #T_54b16_row5_col5, #T_54b16_row7_col2 {
  text-align: left;
  background-color: yellow;
}
#T_54b16_row0_col8, #T_54b16_row1_col8, #T_54b16_row2_col8, #T_54b16_row3_col8, #T_54b16_row4_col8, #T_54b16_row5_col8, #T_54b16_row6_col8, #T_54b16_row7_col8, #T_54b16_row8_col8, #T_54b16_row9_col8, #T_54b16_row10_col8, #T_54b16_row11_col8, #T_54b16_row12_col8, #T_54b16_row13_col8, #T_54b16_row14_col8, #T_54b16_row16_col8 {
  text-align: left;
  background-color: lightgrey;
}
#T_54b16_row15_col8 {
  text-align: left;
  background-color: yellow;
  background-color: lightgrey;
}
</style>
<table id="T_54b16">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_54b16_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_54b16_level0_col1" class="col_heading level0 col1" >Accuracy</th>
      <th id="T_54b16_level0_col2" class="col_heading level0 col2" >AUC</th>
      <th id="T_54b16_level0_col3" class="col_heading level0 col3" >Recall</th>
      <th id="T_54b16_level0_col4" class="col_heading level0 col4" >Prec.</th>
      <th id="T_54b16_level0_col5" class="col_heading level0 col5" >F1</th>
      <th id="T_54b16_level0_col6" class="col_heading level0 col6" >Kappa</th>
      <th id="T_54b16_level0_col7" class="col_heading level0 col7" >MCC</th>
      <th id="T_54b16_level0_col8" class="col_heading level0 col8" >TT (Sec)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_54b16_level0_row0" class="row_heading level0 row0" >gpc</th>
      <td id="T_54b16_row0_col0" class="data row0 col0" >Gaussian Process Classifier</td>
      <td id="T_54b16_row0_col1" class="data row0 col1" >0.7710</td>
      <td id="T_54b16_row0_col2" class="data row0 col2" >0.8098</td>
      <td id="T_54b16_row0_col3" class="data row0 col3" >0.5485</td>
      <td id="T_54b16_row0_col4" class="data row0 col4" >0.7305</td>
      <td id="T_54b16_row0_col5" class="data row0 col5" >0.6223</td>
      <td id="T_54b16_row0_col6" class="data row0 col6" >0.4645</td>
      <td id="T_54b16_row0_col7" class="data row0 col7" >0.4764</td>
      <td id="T_54b16_row0_col8" class="data row0 col8" >0.1060</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row1" class="row_heading level0 row1" >et</th>
      <td id="T_54b16_row1_col0" class="data row1 col0" >Extra Trees Classifier</td>
      <td id="T_54b16_row1_col1" class="data row1 col1" >0.7691</td>
      <td id="T_54b16_row1_col2" class="data row1 col2" >0.8185</td>
      <td id="T_54b16_row1_col3" class="data row1 col3" >0.5643</td>
      <td id="T_54b16_row1_col4" class="data row1 col4" >0.7206</td>
      <td id="T_54b16_row1_col5" class="data row1 col5" >0.6279</td>
      <td id="T_54b16_row1_col6" class="data row1 col6" >0.4654</td>
      <td id="T_54b16_row1_col7" class="data row1 col7" >0.4755</td>
      <td id="T_54b16_row1_col8" class="data row1 col8" >0.4720</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row2" class="row_heading level0 row2" >lr</th>
      <td id="T_54b16_row2_col0" class="data row2 col0" >Logistic Regression</td>
      <td id="T_54b16_row2_col1" class="data row2 col1" >0.7653</td>
      <td id="T_54b16_row2_col2" class="data row2 col2" >0.8368</td>
      <td id="T_54b16_row2_col3" class="data row2 col3" >0.5801</td>
      <td id="T_54b16_row2_col4" class="data row2 col4" >0.7055</td>
      <td id="T_54b16_row2_col5" class="data row2 col5" >0.6320</td>
      <td id="T_54b16_row2_col6" class="data row2 col6" >0.4632</td>
      <td id="T_54b16_row2_col7" class="data row2 col7" >0.4704</td>
      <td id="T_54b16_row2_col8" class="data row2 col8" >0.0260</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row3" class="row_heading level0 row3" >rf</th>
      <td id="T_54b16_row3_col0" class="data row3 col0" >Random Forest Classifier</td>
      <td id="T_54b16_row3_col1" class="data row3 col1" >0.7615</td>
      <td id="T_54b16_row3_col2" class="data row3 col2" >0.8406</td>
      <td id="T_54b16_row3_col3" class="data row3 col3" >0.5693</td>
      <td id="T_54b16_row3_col4" class="data row3 col4" >0.6962</td>
      <td id="T_54b16_row3_col5" class="data row3 col5" >0.6202</td>
      <td id="T_54b16_row3_col6" class="data row3 col6" >0.4511</td>
      <td id="T_54b16_row3_col7" class="data row3 col7" >0.4591</td>
      <td id="T_54b16_row3_col8" class="data row3 col8" >0.4840</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row4" class="row_heading level0 row4" >ada</th>
      <td id="T_54b16_row4_col0" class="data row4 col0" >Ada Boost Classifier</td>
      <td id="T_54b16_row4_col1" class="data row4 col1" >0.7597</td>
      <td id="T_54b16_row4_col2" class="data row4 col2" >0.8199</td>
      <td id="T_54b16_row4_col3" class="data row4 col3" >0.6061</td>
      <td id="T_54b16_row4_col4" class="data row4 col4" >0.6741</td>
      <td id="T_54b16_row4_col5" class="data row4 col5" >0.6360</td>
      <td id="T_54b16_row4_col6" class="data row4 col6" >0.4580</td>
      <td id="T_54b16_row4_col7" class="data row4 col7" >0.4609</td>
      <td id="T_54b16_row4_col8" class="data row4 col8" >0.0820</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row5" class="row_heading level0 row5" >lightgbm</th>
      <td id="T_54b16_row5_col0" class="data row5 col0" >Light Gradient Boosting Machine</td>
      <td id="T_54b16_row5_col1" class="data row5 col1" >0.7597</td>
      <td id="T_54b16_row5_col2" class="data row5 col2" >0.8174</td>
      <td id="T_54b16_row5_col3" class="data row5 col3" >0.6333</td>
      <td id="T_54b16_row5_col4" class="data row5 col4" >0.6677</td>
      <td id="T_54b16_row5_col5" class="data row5 col5" >0.6437</td>
      <td id="T_54b16_row5_col6" class="data row5 col6" >0.4643</td>
      <td id="T_54b16_row5_col7" class="data row5 col7" >0.4691</td>
      <td id="T_54b16_row5_col8" class="data row5 col8" >0.9150</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row6" class="row_heading level0 row6" >lda</th>
      <td id="T_54b16_row6_col0" class="data row6 col0" >Linear Discriminant Analysis</td>
      <td id="T_54b16_row6_col1" class="data row6 col1" >0.7596</td>
      <td id="T_54b16_row6_col2" class="data row6 col2" >0.8319</td>
      <td id="T_54b16_row6_col3" class="data row6 col3" >0.5798</td>
      <td id="T_54b16_row6_col4" class="data row6 col4" >0.6866</td>
      <td id="T_54b16_row6_col5" class="data row6 col5" >0.6223</td>
      <td id="T_54b16_row6_col6" class="data row6 col6" >0.4501</td>
      <td id="T_54b16_row6_col7" class="data row6 col7" >0.4565</td>
      <td id="T_54b16_row6_col8" class="data row6 col8" >0.0140</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row7" class="row_heading level0 row7" >rbfsvm</th>
      <td id="T_54b16_row7_col0" class="data row7 col0" >SVM - Radial Kernel</td>
      <td id="T_54b16_row7_col1" class="data row7 col1" >0.7577</td>
      <td id="T_54b16_row7_col2" class="data row7 col2" >0.8418</td>
      <td id="T_54b16_row7_col3" class="data row7 col3" >0.5263</td>
      <td id="T_54b16_row7_col4" class="data row7 col4" >0.7080</td>
      <td id="T_54b16_row7_col5" class="data row7 col5" >0.6004</td>
      <td id="T_54b16_row7_col6" class="data row7 col6" >0.4331</td>
      <td id="T_54b16_row7_col7" class="data row7 col7" >0.4445</td>
      <td id="T_54b16_row7_col8" class="data row7 col8" >0.0300</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row8" class="row_heading level0 row8" >ridge</th>
      <td id="T_54b16_row8_col0" class="data row8 col0" >Ridge Classifier</td>
      <td id="T_54b16_row8_col1" class="data row8 col1" >0.7559</td>
      <td id="T_54b16_row8_col2" class="data row8 col2" >0.0000</td>
      <td id="T_54b16_row8_col3" class="data row8 col3" >0.5693</td>
      <td id="T_54b16_row8_col4" class="data row8 col4" >0.6790</td>
      <td id="T_54b16_row8_col5" class="data row8 col5" >0.6151</td>
      <td id="T_54b16_row8_col6" class="data row8 col6" >0.4405</td>
      <td id="T_54b16_row8_col7" class="data row8 col7" >0.4457</td>
      <td id="T_54b16_row8_col8" class="data row8 col8" >0.0090</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row9" class="row_heading level0 row9" >gbc</th>
      <td id="T_54b16_row9_col0" class="data row9 col0" >Gradient Boosting Classifier</td>
      <td id="T_54b16_row9_col1" class="data row9 col1" >0.7541</td>
      <td id="T_54b16_row9_col2" class="data row9 col2" >0.8396</td>
      <td id="T_54b16_row9_col3" class="data row9 col3" >0.6225</td>
      <td id="T_54b16_row9_col4" class="data row9 col4" >0.6566</td>
      <td id="T_54b16_row9_col5" class="data row9 col5" >0.6332</td>
      <td id="T_54b16_row9_col6" class="data row9 col6" >0.4502</td>
      <td id="T_54b16_row9_col7" class="data row9 col7" >0.4544</td>
      <td id="T_54b16_row9_col8" class="data row9 col8" >0.1010</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row10" class="row_heading level0 row10" >knn</th>
      <td id="T_54b16_row10_col0" class="data row10 col0" >K Neighbors Classifier</td>
      <td id="T_54b16_row10_col1" class="data row10 col1" >0.7466</td>
      <td id="T_54b16_row10_col2" class="data row10 col2" >0.7800</td>
      <td id="T_54b16_row10_col3" class="data row10 col3" >0.5424</td>
      <td id="T_54b16_row10_col4" class="data row10 col4" >0.6789</td>
      <td id="T_54b16_row10_col5" class="data row10 col5" >0.6000</td>
      <td id="T_54b16_row10_col6" class="data row10 col6" >0.4183</td>
      <td id="T_54b16_row10_col7" class="data row10 col7" >0.4262</td>
      <td id="T_54b16_row10_col8" class="data row10 col8" >0.3870</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row11" class="row_heading level0 row11" >mlp</th>
      <td id="T_54b16_row11_col0" class="data row11 col0" >MLP Classifier</td>
      <td id="T_54b16_row11_col1" class="data row11 col1" >0.7411</td>
      <td id="T_54b16_row11_col2" class="data row11 col2" >0.8044</td>
      <td id="T_54b16_row11_col3" class="data row11 col3" >0.5860</td>
      <td id="T_54b16_row11_col4" class="data row11 col4" >0.6483</td>
      <td id="T_54b16_row11_col5" class="data row11 col5" >0.6105</td>
      <td id="T_54b16_row11_col6" class="data row11 col6" >0.4186</td>
      <td id="T_54b16_row11_col7" class="data row11 col7" >0.4230</td>
      <td id="T_54b16_row11_col8" class="data row11 col8" >1.6220</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row12" class="row_heading level0 row12" >svm</th>
      <td id="T_54b16_row12_col0" class="data row12 col0" >SVM - Linear Kernel</td>
      <td id="T_54b16_row12_col1" class="data row12 col1" >0.7299</td>
      <td id="T_54b16_row12_col2" class="data row12 col2" >0.0000</td>
      <td id="T_54b16_row12_col3" class="data row12 col3" >0.6284</td>
      <td id="T_54b16_row12_col4" class="data row12 col4" >0.6121</td>
      <td id="T_54b16_row12_col5" class="data row12 col5" >0.6119</td>
      <td id="T_54b16_row12_col6" class="data row12 col6" >0.4073</td>
      <td id="T_54b16_row12_col7" class="data row12 col7" >0.4127</td>
      <td id="T_54b16_row12_col8" class="data row12 col8" >0.0090</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row13" class="row_heading level0 row13" >dt</th>
      <td id="T_54b16_row13_col0" class="data row13 col0" >Decision Tree Classifier</td>
      <td id="T_54b16_row13_col1" class="data row13 col1" >0.7187</td>
      <td id="T_54b16_row13_col2" class="data row13 col2" >0.6909</td>
      <td id="T_54b16_row13_col3" class="data row13 col3" >0.5965</td>
      <td id="T_54b16_row13_col4" class="data row13 col4" >0.6013</td>
      <td id="T_54b16_row13_col5" class="data row13 col5" >0.5919</td>
      <td id="T_54b16_row13_col6" class="data row13 col6" >0.3799</td>
      <td id="T_54b16_row13_col7" class="data row13 col7" >0.3848</td>
      <td id="T_54b16_row13_col8" class="data row13 col8" >0.0100</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row14" class="row_heading level0 row14" >nb</th>
      <td id="T_54b16_row14_col0" class="data row14 col0" >Naive Bayes</td>
      <td id="T_54b16_row14_col1" class="data row14 col1" >0.6610</td>
      <td id="T_54b16_row14_col2" class="data row14 col2" >0.7666</td>
      <td id="T_54b16_row14_col3" class="data row14 col3" >0.1123</td>
      <td id="T_54b16_row14_col4" class="data row14 col4" >0.4499</td>
      <td id="T_54b16_row14_col5" class="data row14 col5" >0.1719</td>
      <td id="T_54b16_row14_col6" class="data row14 col6" >0.0824</td>
      <td id="T_54b16_row14_col7" class="data row14 col7" >0.1127</td>
      <td id="T_54b16_row14_col8" class="data row14 col8" >0.0090</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row15" class="row_heading level0 row15" >dummy</th>
      <td id="T_54b16_row15_col0" class="data row15 col0" >Dummy Classifier</td>
      <td id="T_54b16_row15_col1" class="data row15 col1" >0.6499</td>
      <td id="T_54b16_row15_col2" class="data row15 col2" >0.5000</td>
      <td id="T_54b16_row15_col3" class="data row15 col3" >0.0000</td>
      <td id="T_54b16_row15_col4" class="data row15 col4" >0.0000</td>
      <td id="T_54b16_row15_col5" class="data row15 col5" >0.0000</td>
      <td id="T_54b16_row15_col6" class="data row15 col6" >0.0000</td>
      <td id="T_54b16_row15_col7" class="data row15 col7" >0.0000</td>
      <td id="T_54b16_row15_col8" class="data row15 col8" >0.0060</td>
    </tr>
    <tr>
      <th id="T_54b16_level0_row16" class="row_heading level0 row16" >qda</th>
      <td id="T_54b16_row16_col0" class="data row16 col0" >Quadratic Discriminant Analysis</td>
      <td id="T_54b16_row16_col1" class="data row16 col1" >0.5529</td>
      <td id="T_54b16_row16_col2" class="data row16 col2" >0.5573</td>
      <td id="T_54b16_row16_col3" class="data row16 col3" >0.5865</td>
      <td id="T_54b16_row16_col4" class="data row16 col4" >0.4949</td>
      <td id="T_54b16_row16_col5" class="data row16 col5" >0.4345</td>
      <td id="T_54b16_row16_col6" class="data row16 col6" >0.1167</td>
      <td id="T_54b16_row16_col7" class="data row16 col7" >0.1724</td>
      <td id="T_54b16_row16_col8" class="data row16 col8" >0.0090</td>
    </tr>
  </tbody>
</table>



### create_model
여러 모델이 아닌 하나의 모델에 대해서 `setup` 설정으로 학습 및 결과 확인


```python
clf_lgbm = create_model("lightgbm")
```


<style type="text/css">
#T_cd79a_row10_col0, #T_cd79a_row10_col1, #T_cd79a_row10_col2, #T_cd79a_row10_col3, #T_cd79a_row10_col4, #T_cd79a_row10_col5, #T_cd79a_row10_col6 {
  background: yellow;
}
</style>
<table id="T_cd79a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_cd79a_level0_col0" class="col_heading level0 col0" >Accuracy</th>
      <th id="T_cd79a_level0_col1" class="col_heading level0 col1" >AUC</th>
      <th id="T_cd79a_level0_col2" class="col_heading level0 col2" >Recall</th>
      <th id="T_cd79a_level0_col3" class="col_heading level0 col3" >Prec.</th>
      <th id="T_cd79a_level0_col4" class="col_heading level0 col4" >F1</th>
      <th id="T_cd79a_level0_col5" class="col_heading level0 col5" >Kappa</th>
      <th id="T_cd79a_level0_col6" class="col_heading level0 col6" >MCC</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_cd79a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_cd79a_row0_col0" class="data row0 col0" >0.8148</td>
      <td id="T_cd79a_row0_col1" class="data row0 col1" >0.8932</td>
      <td id="T_cd79a_row0_col2" class="data row0 col2" >0.7895</td>
      <td id="T_cd79a_row0_col3" class="data row0 col3" >0.7143</td>
      <td id="T_cd79a_row0_col4" class="data row0 col4" >0.7500</td>
      <td id="T_cd79a_row0_col5" class="data row0 col5" >0.6035</td>
      <td id="T_cd79a_row0_col6" class="data row0 col6" >0.6054</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_cd79a_row1_col0" class="data row1 col0" >0.7593</td>
      <td id="T_cd79a_row1_col1" class="data row1 col1" >0.8045</td>
      <td id="T_cd79a_row1_col2" class="data row1 col2" >0.4737</td>
      <td id="T_cd79a_row1_col3" class="data row1 col3" >0.7500</td>
      <td id="T_cd79a_row1_col4" class="data row1 col4" >0.5806</td>
      <td id="T_cd79a_row1_col5" class="data row1 col5" >0.4236</td>
      <td id="T_cd79a_row1_col6" class="data row1 col6" >0.4456</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_cd79a_row2_col0" class="data row2 col0" >0.7222</td>
      <td id="T_cd79a_row2_col1" class="data row2 col1" >0.8466</td>
      <td id="T_cd79a_row2_col2" class="data row2 col2" >0.6316</td>
      <td id="T_cd79a_row2_col3" class="data row2 col3" >0.6000</td>
      <td id="T_cd79a_row2_col4" class="data row2 col4" >0.6154</td>
      <td id="T_cd79a_row2_col5" class="data row2 col5" >0.3982</td>
      <td id="T_cd79a_row2_col6" class="data row2 col6" >0.3985</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_cd79a_row3_col0" class="data row3 col0" >0.6852</td>
      <td id="T_cd79a_row3_col1" class="data row3 col1" >0.7278</td>
      <td id="T_cd79a_row3_col2" class="data row3 col2" >0.6316</td>
      <td id="T_cd79a_row3_col3" class="data row3 col3" >0.5455</td>
      <td id="T_cd79a_row3_col4" class="data row3 col4" >0.5854</td>
      <td id="T_cd79a_row3_col5" class="data row3 col5" >0.3338</td>
      <td id="T_cd79a_row3_col6" class="data row3 col6" >0.3361</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_cd79a_row4_col0" class="data row4 col0" >0.7778</td>
      <td id="T_cd79a_row4_col1" class="data row4 col1" >0.8451</td>
      <td id="T_cd79a_row4_col2" class="data row4 col2" >0.7368</td>
      <td id="T_cd79a_row4_col3" class="data row4 col3" >0.6667</td>
      <td id="T_cd79a_row4_col4" class="data row4 col4" >0.7000</td>
      <td id="T_cd79a_row4_col5" class="data row4 col5" >0.5242</td>
      <td id="T_cd79a_row4_col6" class="data row4 col6" >0.5259</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_cd79a_row5_col0" class="data row5 col0" >0.8519</td>
      <td id="T_cd79a_row5_col1" class="data row5 col1" >0.9023</td>
      <td id="T_cd79a_row5_col2" class="data row5 col2" >0.7895</td>
      <td id="T_cd79a_row5_col3" class="data row5 col3" >0.7895</td>
      <td id="T_cd79a_row5_col4" class="data row5 col4" >0.7895</td>
      <td id="T_cd79a_row5_col5" class="data row5 col5" >0.6752</td>
      <td id="T_cd79a_row5_col6" class="data row5 col6" >0.6752</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_cd79a_row6_col0" class="data row6 col0" >0.7222</td>
      <td id="T_cd79a_row6_col1" class="data row6 col1" >0.7158</td>
      <td id="T_cd79a_row6_col2" class="data row6 col2" >0.4737</td>
      <td id="T_cd79a_row6_col3" class="data row6 col3" >0.6429</td>
      <td id="T_cd79a_row6_col4" class="data row6 col4" >0.5455</td>
      <td id="T_cd79a_row6_col5" class="data row6 col5" >0.3520</td>
      <td id="T_cd79a_row6_col6" class="data row6 col6" >0.3605</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_cd79a_row7_col0" class="data row7 col0" >0.7358</td>
      <td id="T_cd79a_row7_col1" class="data row7 col1" >0.8079</td>
      <td id="T_cd79a_row7_col2" class="data row7 col2" >0.5556</td>
      <td id="T_cd79a_row7_col3" class="data row7 col3" >0.6250</td>
      <td id="T_cd79a_row7_col4" class="data row7 col4" >0.5882</td>
      <td id="T_cd79a_row7_col5" class="data row7 col5" >0.3948</td>
      <td id="T_cd79a_row7_col6" class="data row7 col6" >0.3963</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_cd79a_row8_col0" class="data row8 col0" >0.8113</td>
      <td id="T_cd79a_row8_col1" class="data row8 col1" >0.8492</td>
      <td id="T_cd79a_row8_col2" class="data row8 col2" >0.7778</td>
      <td id="T_cd79a_row8_col3" class="data row8 col3" >0.7000</td>
      <td id="T_cd79a_row8_col4" class="data row8 col4" >0.7368</td>
      <td id="T_cd79a_row8_col5" class="data row8 col5" >0.5904</td>
      <td id="T_cd79a_row8_col6" class="data row8 col6" >0.5924</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_cd79a_row9_col0" class="data row9 col0" >0.7170</td>
      <td id="T_cd79a_row9_col1" class="data row9 col1" >0.7786</td>
      <td id="T_cd79a_row9_col2" class="data row9 col2" >0.4737</td>
      <td id="T_cd79a_row9_col3" class="data row9 col3" >0.6429</td>
      <td id="T_cd79a_row9_col4" class="data row9 col4" >0.5455</td>
      <td id="T_cd79a_row9_col5" class="data row9 col5" >0.3468</td>
      <td id="T_cd79a_row9_col6" class="data row9 col6" >0.3553</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_cd79a_row10_col0" class="data row10 col0" >0.7597</td>
      <td id="T_cd79a_row10_col1" class="data row10 col1" >0.8171</td>
      <td id="T_cd79a_row10_col2" class="data row10 col2" >0.6333</td>
      <td id="T_cd79a_row10_col3" class="data row10 col3" >0.6677</td>
      <td id="T_cd79a_row10_col4" class="data row10 col4" >0.6437</td>
      <td id="T_cd79a_row10_col5" class="data row10 col5" >0.4643</td>
      <td id="T_cd79a_row10_col6" class="data row10 col6" >0.4691</td>
    </tr>
    <tr>
      <th id="T_cd79a_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_cd79a_row11_col0" class="data row11 col0" >0.0503</td>
      <td id="T_cd79a_row11_col1" class="data row11 col1" >0.0597</td>
      <td id="T_cd79a_row11_col2" class="data row11 col2" >0.1276</td>
      <td id="T_cd79a_row11_col3" class="data row11 col3" >0.0688</td>
      <td id="T_cd79a_row11_col4" class="data row11 col4" >0.0866</td>
      <td id="T_cd79a_row11_col5" class="data row11 col5" >0.1173</td>
      <td id="T_cd79a_row11_col6" class="data row11 col6" >0.1152</td>
    </tr>
  </tbody>
</table>



### tune_model
하이퍼파라미터 튜닝을 도와주는 메서드


```python
tuned_clf_lgbm = tune_model(clf_lgbm, n_iter=10, optimize="Accuracy")
```


<style type="text/css">
#T_3ac77_row10_col0, #T_3ac77_row10_col1, #T_3ac77_row10_col2, #T_3ac77_row10_col3, #T_3ac77_row10_col4, #T_3ac77_row10_col5, #T_3ac77_row10_col6 {
  background: yellow;
}
</style>
<table id="T_3ac77">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3ac77_level0_col0" class="col_heading level0 col0" >Accuracy</th>
      <th id="T_3ac77_level0_col1" class="col_heading level0 col1" >AUC</th>
      <th id="T_3ac77_level0_col2" class="col_heading level0 col2" >Recall</th>
      <th id="T_3ac77_level0_col3" class="col_heading level0 col3" >Prec.</th>
      <th id="T_3ac77_level0_col4" class="col_heading level0 col4" >F1</th>
      <th id="T_3ac77_level0_col5" class="col_heading level0 col5" >Kappa</th>
      <th id="T_3ac77_level0_col6" class="col_heading level0 col6" >MCC</th>
    </tr>
    <tr>
      <th class="index_name level0" >Fold</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3ac77_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3ac77_row0_col0" class="data row0 col0" >0.8333</td>
      <td id="T_3ac77_row0_col1" class="data row0 col1" >0.9308</td>
      <td id="T_3ac77_row0_col2" class="data row0 col2" >0.8421</td>
      <td id="T_3ac77_row0_col3" class="data row0 col3" >0.7273</td>
      <td id="T_3ac77_row0_col4" class="data row0 col4" >0.7805</td>
      <td id="T_3ac77_row0_col5" class="data row0 col5" >0.6473</td>
      <td id="T_3ac77_row0_col6" class="data row0 col6" >0.6518</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3ac77_row1_col0" class="data row1 col0" >0.8148</td>
      <td id="T_3ac77_row1_col1" class="data row1 col1" >0.8782</td>
      <td id="T_3ac77_row1_col2" class="data row1 col2" >0.6316</td>
      <td id="T_3ac77_row1_col3" class="data row1 col3" >0.8000</td>
      <td id="T_3ac77_row1_col4" class="data row1 col4" >0.7059</td>
      <td id="T_3ac77_row1_col5" class="data row1 col5" >0.5735</td>
      <td id="T_3ac77_row1_col6" class="data row1 col6" >0.5820</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3ac77_row2_col0" class="data row2 col0" >0.8148</td>
      <td id="T_3ac77_row2_col1" class="data row2 col1" >0.8496</td>
      <td id="T_3ac77_row2_col2" class="data row2 col2" >0.7368</td>
      <td id="T_3ac77_row2_col3" class="data row2 col3" >0.7368</td>
      <td id="T_3ac77_row2_col4" class="data row2 col4" >0.7368</td>
      <td id="T_3ac77_row2_col5" class="data row2 col5" >0.5940</td>
      <td id="T_3ac77_row2_col6" class="data row2 col6" >0.5940</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3ac77_row3_col0" class="data row3 col0" >0.6852</td>
      <td id="T_3ac77_row3_col1" class="data row3 col1" >0.7353</td>
      <td id="T_3ac77_row3_col2" class="data row3 col2" >0.4211</td>
      <td id="T_3ac77_row3_col3" class="data row3 col3" >0.5714</td>
      <td id="T_3ac77_row3_col4" class="data row3 col4" >0.4848</td>
      <td id="T_3ac77_row3_col5" class="data row3 col5" >0.2656</td>
      <td id="T_3ac77_row3_col6" class="data row3 col6" >0.2720</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3ac77_row4_col0" class="data row4 col0" >0.7222</td>
      <td id="T_3ac77_row4_col1" class="data row4 col1" >0.8226</td>
      <td id="T_3ac77_row4_col2" class="data row4 col2" >0.6316</td>
      <td id="T_3ac77_row4_col3" class="data row4 col3" >0.6000</td>
      <td id="T_3ac77_row4_col4" class="data row4 col4" >0.6154</td>
      <td id="T_3ac77_row4_col5" class="data row4 col5" >0.3982</td>
      <td id="T_3ac77_row4_col6" class="data row4 col6" >0.3985</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_3ac77_row5_col0" class="data row5 col0" >0.8333</td>
      <td id="T_3ac77_row5_col1" class="data row5 col1" >0.8977</td>
      <td id="T_3ac77_row5_col2" class="data row5 col2" >0.6842</td>
      <td id="T_3ac77_row5_col3" class="data row5 col3" >0.8125</td>
      <td id="T_3ac77_row5_col4" class="data row5 col4" >0.7429</td>
      <td id="T_3ac77_row5_col5" class="data row5 col5" >0.6209</td>
      <td id="T_3ac77_row5_col6" class="data row5 col6" >0.6259</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_3ac77_row6_col0" class="data row6 col0" >0.7593</td>
      <td id="T_3ac77_row6_col1" class="data row6 col1" >0.7805</td>
      <td id="T_3ac77_row6_col2" class="data row6 col2" >0.5263</td>
      <td id="T_3ac77_row6_col3" class="data row6 col3" >0.7143</td>
      <td id="T_3ac77_row6_col4" class="data row6 col4" >0.6061</td>
      <td id="T_3ac77_row6_col5" class="data row6 col5" >0.4384</td>
      <td id="T_3ac77_row6_col6" class="data row6 col6" >0.4490</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_3ac77_row7_col0" class="data row7 col0" >0.7170</td>
      <td id="T_3ac77_row7_col1" class="data row7 col1" >0.8500</td>
      <td id="T_3ac77_row7_col2" class="data row7 col2" >0.5556</td>
      <td id="T_3ac77_row7_col3" class="data row7 col3" >0.5882</td>
      <td id="T_3ac77_row7_col4" class="data row7 col4" >0.5714</td>
      <td id="T_3ac77_row7_col5" class="data row7 col5" >0.3604</td>
      <td id="T_3ac77_row7_col6" class="data row7 col6" >0.3607</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_3ac77_row8_col0" class="data row8 col0" >0.7547</td>
      <td id="T_3ac77_row8_col1" class="data row8 col1" >0.8492</td>
      <td id="T_3ac77_row8_col2" class="data row8 col2" >0.5556</td>
      <td id="T_3ac77_row8_col3" class="data row8 col3" >0.6667</td>
      <td id="T_3ac77_row8_col4" class="data row8 col4" >0.6061</td>
      <td id="T_3ac77_row8_col5" class="data row8 col5" >0.4301</td>
      <td id="T_3ac77_row8_col6" class="data row8 col6" >0.4339</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_3ac77_row9_col0" class="data row9 col0" >0.7170</td>
      <td id="T_3ac77_row9_col1" class="data row9 col1" >0.7724</td>
      <td id="T_3ac77_row9_col2" class="data row9 col2" >0.5263</td>
      <td id="T_3ac77_row9_col3" class="data row9 col3" >0.6250</td>
      <td id="T_3ac77_row9_col4" class="data row9 col4" >0.5714</td>
      <td id="T_3ac77_row9_col5" class="data row9 col5" >0.3625</td>
      <td id="T_3ac77_row9_col6" class="data row9 col6" >0.3655</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row10" class="row_heading level0 row10" >Mean</th>
      <td id="T_3ac77_row10_col0" class="data row10 col0" >0.7652</td>
      <td id="T_3ac77_row10_col1" class="data row10 col1" >0.8366</td>
      <td id="T_3ac77_row10_col2" class="data row10 col2" >0.6111</td>
      <td id="T_3ac77_row10_col3" class="data row10 col3" >0.6842</td>
      <td id="T_3ac77_row10_col4" class="data row10 col4" >0.6421</td>
      <td id="T_3ac77_row10_col5" class="data row10 col5" >0.4691</td>
      <td id="T_3ac77_row10_col6" class="data row10 col6" >0.4733</td>
    </tr>
    <tr>
      <th id="T_3ac77_level0_row11" class="row_heading level0 row11" >Std</th>
      <td id="T_3ac77_row11_col0" class="data row11 col0" >0.0522</td>
      <td id="T_3ac77_row11_col1" class="data row11 col1" >0.0571</td>
      <td id="T_3ac77_row11_col2" class="data row11 col2" >0.1149</td>
      <td id="T_3ac77_row11_col3" class="data row11 col3" >0.0826</td>
      <td id="T_3ac77_row11_col4" class="data row11 col4" >0.0897</td>
      <td id="T_3ac77_row11_col5" class="data row11 col5" >0.1238</td>
      <td id="T_3ac77_row11_col6" class="data row11 col6" >0.1241</td>
    </tr>
  </tbody>
</table>



### save_model
학습한 모델을 저장


```python
save_model(tuned_clf_lgbm, "./tuned_clf_lgbm")
```

    Transformation Pipeline and Model Successfully Saved
    




    (Pipeline(memory=None,
              steps=[('dtypes',
                      DataTypes_Auto_infer(categorical_features=[],
                                           display_types=True, features_todrop=[],
                                           id_columns=[],
                                           ml_usecase='classification',
                                           numerical_features=[], target='Outcome',
                                           time_features=[])),
                     ('imputer',
                      Simple_Imputer(categorical_strategy='not_available',
                                     fill_value_categorical=None,
                                     fill_value_numerical=None,
                                     numeric_stra...
                                     colsample_bytree=1.0, device='gpu',
                                     feature_fraction=1.0, importance_type='split',
                                     learning_rate=0.1, max_depth=-1,
                                     min_child_samples=71, min_child_weight=0.001,
                                     min_split_gain=0.6, n_estimators=130, n_jobs=-1,
                                     num_leaves=4, objective=None, random_state=42,
                                     reg_alpha=0.3, reg_lambda=4, silent='warn',
                                     subsample=1.0, subsample_for_bin=200000,
                                     subsample_freq=0)]],
              verbose=False),
     './tuned_clf_lgbm.pkl')



### load_model


```python
clf_lgbm = load_model("./tuned_clf_lgbm")
```

    Transformation Pipeline and Model Successfully Loaded
    


```python
clf_lgbm["trained_model"]
```




    LGBMClassifier(bagging_fraction=0.6, bagging_freq=5, boosting_type='gbdt',
                   class_weight=None, colsample_bytree=1.0, device='gpu',
                   feature_fraction=1.0, importance_type='split', learning_rate=0.1,
                   max_depth=-1, min_child_samples=71, min_child_weight=0.001,
                   min_split_gain=0.6, n_estimators=130, n_jobs=-1, num_leaves=4,
                   objective=None, random_state=42, reg_alpha=0.3, reg_lambda=4,
                   silent='warn', subsample=1.0, subsample_for_bin=200000,
                   subsample_freq=0)



위와 같이 하이퍼파라미터 튜닝 목록을 확인할 수 있음
