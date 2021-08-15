---
layout: single
title:  "[글또] Feature selection"
excerpt: "적절한 변수 고르기"

categories:
  - machine_learning
tags:
  - [machine_learning, AI, regression, feature, feature_selection, wrapper, rfe, sfs, selectfrommodel]
header:
  teaser: https://user-images.githubusercontent.com/66911578/129455808-54ecb625-ec6f-47ee-aed5-b0fadabd0e05.jpg
toc: true
toc_sticky: true
 
date: 2021-08-15
last_modified_at: 2021-08-15
---
# 개요

분석해야 하는 데이터를 탐색하고 머신러닝 알고리즘을 이용해 회귀나 분류 문제를 모델링할 때 중요한 것 중 하나가 바로 변수의 취사선택입니다. 레이블과 어떠한 형태로든 관련 있는 혹은 더 관련 있는 변수를 추리고 그 변수로 재구성된 데이터를 모델에 입력해 결과를 냅니다. 변수 선택으로 우리가 얻을 수 있는 이익은 아래 세 가지 정도입니다.

1. 기준(VIF 5 미만, t검정 통계량 1 미만 등)에 부적합한 features를 제거한 후 모델링하면 평가지표 상승을 기대할 수 있습니다. 모델을 단순화하여 과적합을 줄여줍니다.  
2. AI 모델링에 요구되는 정보의 사이즈를 줄여 데이터 수집, 관리에 드는 비용을 절감할 수 있습니다. 예측이란 항상 불확실성을 동반하고 이를 완화하려면 초기 탐색 단계에서 데이터는 다다익선입니다. 그러나 다양한 소스에서 새로운 데이터가 계속 축적됩니다. train & inference를 위해 많은 features가 필요하다면 이를 수집, 처리, 관리하는 것은 또 다른 부담입니다. 특정 feature의 양상, 경향이 바뀔 수도 있고 이상치 여부를 판단해야 하며 뜻하지 않은 이슈로 대량의 결측이 발생할 가능성도 염두에 두어야 합니다. 데이터 관리 및 모델링의 안정성 측면에서도 적절한 양의, 꼭 필요한, 최적화된 변수 집합 탐색을 반드시 고려해야 합니다.  
3. 데이터를 분석하여 "유의미한 변수"를 추출해 제시하는 것만으로도 고객의 신뢰를 얻습니다. 고객은 해당 분야의 현직자로서 독자적인 도메인 지식과 경험을 갖추고 있습니다. 그들의 입장에서 납득 가능한, 그들도 중요시하는 변수를 통계와 머신러닝 기법만으로 골라 보여준다면, 이는 데이터 사이언티스트 나아가 인공지능에 대한 의구심을 해소하고 그 결과물을 수용하게 하는 단초가 됩니다.

저는 보통 1. EDA 부분에서 시각화, 통계적 기법을 활용해 features와 label의 관계를 살피고 2. 머신러닝 모델링 시에 여러 변수 선택 기법을 활용합니다. 기법은 크게 세 가지로 구분됩니다.
1. 필터(filter): 카이 제곱 검정, 상관계수 등 통계학 이용
2. 래퍼(wrapper): 가장 적합한 변수 부분 집합 탐색. 전진선택, 후진제거, stepwise 등
3. 임베디드(embedded): LASSO, Ridge 등 모델 자체에 변수 선택을 위한 조건이 포함된 기법

이번 시간에는 래퍼 기법 중 recursive feature elimination, sequential feature selection과 임베디드 기법인 select from model을 이해하고 파이썬 코드로 구현하며 그 결과를 scikit-learn 라이브러리의 대응되는 함수로 산출한 값과 비교해보겠습니다.
<br>

<br />
# 필요한 라이브러리와 데이터 불러오기

Numpy, Pandas 그리고 scikit-learn에서 샘플 데이터(보스턴 주택 가격), 회귀 모델, 변수 선택 기법, cross validation 등을 불러옵니다.


```python
import numpy as np
import pandas as pd
```


```python
from sklearn.base import clone
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
```


```python
X, y, feature_names, _, _ = load_boston().values()
```


```python
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X, columns = feature_names), pd.Series(y, name='label'), test_size = .2, random_state=42)
```


```python
X_train.head()
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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>477</th>
      <td>15.02340</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.6140</td>
      <td>5.304</td>
      <td>97.3</td>
      <td>2.1007</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>349.48</td>
      <td>24.91</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.62739</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.5380</td>
      <td>5.834</td>
      <td>56.5</td>
      <td>4.4986</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>395.62</td>
      <td>8.47</td>
    </tr>
    <tr>
      <th>332</th>
      <td>0.03466</td>
      <td>35.0</td>
      <td>6.06</td>
      <td>0.0</td>
      <td>0.4379</td>
      <td>6.031</td>
      <td>23.3</td>
      <td>6.6407</td>
      <td>1.0</td>
      <td>304.0</td>
      <td>16.9</td>
      <td>362.25</td>
      <td>7.83</td>
    </tr>
    <tr>
      <th>423</th>
      <td>7.05042</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.6140</td>
      <td>6.103</td>
      <td>85.1</td>
      <td>2.0218</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>2.52</td>
      <td>23.29</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.72580</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.5380</td>
      <td>5.727</td>
      <td>69.5</td>
      <td>3.7965</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>390.95</td>
      <td>11.28</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

가장 기본적인 회귀 모델인 Linear Regression을 사용하겠습니다.


```python
model_lr = LinearRegression()
```
<br>

<br />
# Feature selection 구현

## Recursive Feature Elimination
먼저 모델에 전체 데이터를 학습시킵니다. 가장 덜 중요한(트리 기반 모델이면 feature importance, 선형 모델 혹은 SVM이면 coefficient의 절댓값이 가장 작은) 변수를 제외합니다. 모델은 이 변수가 제거된 데이터를 학습하고 역시 중요도가 가장 떨어지는 변수 하나를 선택하여 제외합니다. 이 과정을 반복하고 남아 있는 변수의 수가 사용자가 초기에 설정한 값과 같아지면 멈춥니다.

이 기법을 처음 접했을 때는 모델이 전체 데이터를 학습한 후 (반복 없이) 중요도 상위 N개의 변수만 고르는 기법으로 오해했습니다. 하지만 중요도 평균 이상, 중앙값 이상 등 threshold를 정해두고 그 이상인 변수만 고르는 기법은 Select from model이었습니다. 아래에서 살펴볼 sequential feature selection(backward)과도 다릅니다. 

넘파이 코드로 구현하면 아래와 같습니다. 사용자가 몇 개의 변수만 고를지 지정하지 않으면 전체 변수 개수의 절반만 고르도록 했습니다.


```python
class rfe:
    def __init__(self, estimator, n_select = None):
        self.estimator  = clone(estimator)
        self.n_select = n_select
        
    def fit(self, X, y):
        self.n_features_ori = X.shape[1]
        result = np.ones(self.n_features_ori, dtype=int)
        
        if self.n_select is None:
            self.n_features = self.n_features_ori//2
        else:
            self.n_features = self.n_select
        
        n = 0
        while n < self.n_features_ori - self.n_features:
            self.estimator.fit(X * np.array(result==1, dtype = int), y)
            try:
                rk = np.argsort(self.estimator.feature_importances_)
            except:
                rk = np.argsort(np.abs(self.estimator.coef_.flatten()))
            
            min_idx = rk[:n+1]
            result[min_idx] += 1 
            n += 1
        
        self.ranking = result
        self.support = result == 1
        
        return None
    
    def transform(self, X):
        return X[:, self.support]
        
    def fit_transform(self, X, y):
        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X)
```

중요한 순서대로 랭킹이 매겨집니다. 먼저 축출될수록 순위가 뒤로 밀립니다. 살아남은 N개의 순위는 1로 모두 동일하고 나머지는 늦게 제외된 순서이며 `dense_rank` 방식입니다.

scikit-learn 함수의 결과와 비교했을 때 동일함을 확인할 수 있습니다.


```python
rfe_sk = RFE(estimator=model_lr, n_features_to_select=5)
rfe_sk.fit(X_train, y_train)

rfe_mine = rfe(estimator=model_lr, n_select=5)
rfe_mine.fit(X_train, y_train)

print('Ranking:', rfe_sk.ranking_)
print('sklearn == implementation?:', np.alltrue(rfe_sk.ranking_ == rfe_mine.ranking))
```

    Ranking: [5 7 8 6 1 1 9 1 3 2 1 4 1]
    sklearn == implementation?: True
    

## Sequencial Feature Selection

- **forward**
  1. 어떤 변수도 고르지 않은, 공집합에서 시작합니다. 모든 변수를 하나씩 추출하여 모델에 cross-validation으로 학습시킵니다. 스코어(default: 회귀 r2 / 분류 accuracy) 평균이 가장 높은 변수를 선택해 원소가 1개인 부분집합을 만듭니다.  
  2. '부분집합 + 뽑히지 않은 변수 1개'의 모든 경우의 수에 대해 동일한 작업을 실시합니다. 역시 스코어 평균이 가장 높았던 조합을 선택하고 변수 부분집합에 해당 변수를 추가합니다.
  3. 변수가 사용자가 지정한 N개만큼 뽑힐 때까지 반복합니다.
- **backward**
  1. 모든 변수를 포함한 상태에서 시작합니다. 전체에서 변수 하나씩 제거하여 학습시킵니다. 스코어 평균이 가장 높은 부분집합을 선택하고 해당 변수를 제거합니다.
  2. 변수가 사용자가 지정한 N개만큼 남을 때까지 반복합니다.
- 선택할 N개를 미리 정하지 않고 알고리즘에 따라 선택/제거를 반복하다가 전 단계에 비해 스코어 평균이 감소하면 반복을 중단하고 전 단계의 변수 부분집합을 return하는 형태도 구현할 수 있습니다. 다만 아래에 따로 만들진 않았습니다.

역시 넘파이로 구현했습니다. 사용자가 몇 개의 변수만 고를지 지정하지 않으면 전체 변수 개수의 절반만 고르도록 했습니다.


```python
class sfs:
    def __init__(self, estimator, scoring, n_select = None, cv = 5, direction = 'forward'):
        self.estimator = clone(estimator)
        self.n_select = n_select
        self.cv = cv
        self.direction = direction
        self.scoring = scoring
    
    def fit(self, X, y):
        self.n_features_ori = X.shape[1]
        if self.n_select is None:
            self.n_features = self.n_features_ori//2
        else:
            self.n_features = self.n_select
        if self.direction == 'backward':
            self.n_features = self.n_features_ori - self.n_features
        
        self.support = np.zeros(self.n_features_ori, dtype = bool)
        
        for _ in range(self.n_features):
            non_selected = np.flatnonzero(~self.support)
            dict_score = dict()
            for f in non_selected:
                candidates = self.support.copy()
                candidates[f] = True
                if self.direction == 'backward':
                    candidates = ~candidates
                cvs = cross_val_score(estimator=self.estimator, X = X[:, candidates], y= y, cv=self.cv, scoring=self.scoring).mean()
                dict_score[f] = cvs
            selected = max(dict_score, key=lambda x: dict_score[x])
            self.support[selected] = True
        
        if self.direction == 'backward':
            self.support = ~self.support
            
        return None
    
    def transform(self, X):
        return X[:, self.support]
    
    def fit_transform(self, X, y):
        if y is None:
            return self.fit(X).transform(X)
        else:
            return self.fit(X, y).transform(X)
```

scikit-learn 함수의 결과와 비교했을 때 동일함을 확인할 수 있습니다.

- forward


```python
sfs_sk = SequentialFeatureSelector(estimator=model_lr, cv = 3, direction='forward', scoring='r2')
sfs_sk.fit(X_train, y_train)

sfs_mine = sfs(estimator=model_lr, cv = 3, direction='forward', scoring='r2')
sfs_mine.fit(X_train, y_train)

print('Selected:', sfs_sk.support_)
print('sklearn == implementation?:', np.alltrue(sfs_sk.support_ == sfs_mine.support))
```

    Selected: [False False False False  True  True False  True False False  True  True
      True]
    sklearn == implementation?: True
    

- backward


```python
sfs_sk = SequentialFeatureSelector(estimator=model_lr, cv = 3, direction='backward', scoring='r2')
sfs_sk.fit(X_train, y_train)

sfs_mine = sfs(estimator=model_lr, cv = 3, direction='backward', scoring='r2')
sfs_mine.fit(X_train, y_train)

print('Selected:', sfs_sk.support_)
print('sklearn == implementation?:', np.alltrue(sfs_sk.support_ == sfs_mine.support))
```

    Selected: [False False False False  True  True False  True False False  True  True
      True]
    sklearn == implementation?: True
    

## Select from model

위 두 방법에 비해 알고리즘과 코드를 이해하기 수월합니다. 
1. 모델이 전체 데이터를 학습하게 합니다.
2. 평균, 중앙값 등 중요도의 threshold를 정해두고 threshold를 넘으면 선택하고 넘지 못하면 버립니다.


```python
class SFM:
    def __init__(self, estimator, strategy='mean'):
        self.estimator = clone(estimator)
        self.strategy = strategy
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        
        try:
            self.importance = self.estimator.feature_importances_
        except:
            self.importance = np.abs(self.estimator.coef_.flatten())
            
        if self.strategy == 'mean':
            self.threshold = self.importance.mean()
        elif self.strategy == 'median':
            self.threshold = np.median(self.importance)
        else:
            self.threshold = self.strategy.copy()
            
        self.support = self.importance >= self.threshold
        return None
    
    def transform(self, X):
        return X[:, self.support]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
```

scikit-learn 함수의 결과와 비교했을 때 동일함을 확인할 수 있습니다.


```python
sfm_sk = SelectFromModel(estimator=model_lr, threshold= 'mean')
sfm_sk.fit(X_train, y_train)

sfm_mine = SFM(estimator=model_lr, strategy = 'mean')
sfm_mine.fit(X_train, y_train)

print('- SelectFromModel / threshold = mean')
print('Selected:', sfm_sk.get_support())
print('sklearn == implementation?:', np.alltrue(sfm_sk.get_support() == sfm_mine.support))
```

    - SelectFromModel / threshold = mean
    Selected: [False False False False  True  True False  True  True  True  True False
      True]
    sklearn == implementation?: True
    


```python
sfm_sk = SelectFromModel(estimator=model_lr, threshold= 'median')
sfm_sk.fit(X_train, y_train)

sfm_mine = SFM(estimator=model_lr, strategy = 'median')
sfm_mine.fit(X_train, y_train)

print('- SelectFromModel / threshold = median')
print('Selected:', sfm_sk.get_support())
print('sklearn == implementation?:', np.alltrue(sfm_sk.get_support() == sfm_mine.support))
```

    - SelectFromModel / threshold = median
    Selected: [False False False False  True  True False  True  True  True  True False
      True]
    sklearn == implementation?: True
    
<br>

<br />
# 나가며
이상으로 변수 선택 기법 중 일부에 대해 자세히 알아보았습니다. 알고리즘을 이해한 후 직접 구현하였고 그 결과를 기존 라이브러리의 계산치와 비교하는 방식으로 검증했습니다. 하이퍼파라미터로는 '선택할 변수의 개수'가 있습니다. 이를 튜닝하려면 평가지표(AIC, R2, Accuracy 등)의 증감 추이에 주목해야 한다는 것도 파악했습니다.

중요도는 모델에 따라 feature importance 혹은 coefficient로 사용됩니다. coefficient는 다항식의 '계수'라는 점에서 비교적 직관적이지만 초심자에게 feature importance는 낯선 개념입니다. 다음 시간에는 feature importance란 무엇인지, 어떤 모델에서 쓰이는지, 어떻게 계산하는지 살펴보겠습니다.

[source of teaser](https://unsplash.com/photos/VstBbOwxZSI)
<br>

<br />
[Scroll to Top](#){: .btn .btn--primary .btn-small .align-center}