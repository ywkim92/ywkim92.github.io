---
layout: single
title:  "[글또] Decision tree"
excerpt: "최적 분기(the best split) 탐색과 트리 구조 구현"

categories:
  - machine_learning
tags:
  - [machine_learning, AI, regression, classification, feature, threshold, decision_tree, tree_structure, impurity, mse, mae, variance,node, ]
header:
  teaser: https://user-images.githubusercontent.com/66911578/132971233-d3f8c972-08e3-48b4-ad2a-466685143642.jpg
toc: true
toc_sticky: true
use_math: true
date: 2021-09-12
last_modified_at: 2021-09-12
---
# 개요

물리학을 전공한 제가 ML을 배우기 시작했을 때 선형 회귀나 로지스틱 회귀보다 생소했던 것은 tree 구조와 의사결정나무였습니다. 언뜻 보기에 컨셉은 간단했습니다. 각 노드마다 기준에 따라 샘플을 나누고 분기(split)된 자식 노드에 할당된 샘플에 특정한 종속변수(label) 값을 부여한다는 것이죠. 하지만 어떤 기준으로 나누는지, 그렇게 나누는 목적은 무엇인지, 노드에 부여되는 label 값은 어떻게 계산되는지 등 방법론을 구체적으로, 한눈에 파악하기 어려웠습니다.

다행히도 `plot_tree` 함수와 `tree_` attribute를 접하면서 이해의 물꼬를 틀 수 있었습니다. 그림을 통해 생성된 의사결정나무 모델의 전체 구조를 직관적으로 볼 수 있었고 `tree_`의 low level attributes를 뜯어보며 분기 방법, 노드, 불순도 등 개념을 잡았습니다.

오늘은 1. 의사결정나무의 분기 방법과 기준, 분기 후 얻은 자식 노드에 속하는 샘플에 어떤 값을 부여하는지 살피고 2. `tree_`의 attributes를 받아와 직접 의사결정나무를 구현함으로써 전체 구조를 조망해보겠습니다.

# 필요한 라이브러리와 데이터 불러오기

Numpy, Pandas, matplotlib 그리고 scikit-learn에서 샘플 데이터(보스턴 주택 가격), decision tree, random forest regression model 등을 불러옵니다.


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from treelib import Tree

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.tree import plot_tree, DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
```

학습 및 검증을 위해 데이터를 분할합니다.


```python
data = load_boston()

X = data['data']
y = data['target']

feature_names = data['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
```

# 의사결정나무

## 분기 방법

![decision_tree](https://user-images.githubusercontent.com/66911578/131226624-13bb270b-855c-429c-a9df-4e44ad2ca159.png){: .align-center}{: width="60%" height="60%"}

`plot_tree` 함수로 트리를 그려보면 각 노드의 상태가 표기되어 있습니다. 세부 사항은 [지난 포스트](https://ywkim92.github.io/machine_learning/feature_importance/#%EC%8B%9C%EA%B0%81%ED%99%94)에서 설명하였습니다. 다른 값들에 비해 유독 두 번째 줄 즉 feature와 threshold가 어째서 그렇게 결정되는지 이해하기 어려웠습니다. `random_state`가 바뀌어도 분기 기준은 바뀌지 않았으므로 무작위는 아니었습니다. 

먼저 분기 때마다 feature가 어떻게 정해지는지 찾아봤습니다. 검색 능력이 신통치 않아 적당한 레퍼런스를 발견하지 못했습니다. 다음엔 가설을 세우고 검증을 진행했습니다. 학습 데이터의 변수 중 분산이 가장 큰 (독립)변수? unique value 개수가 가장 많은 변수? (regression tree의 경우) 레이블과의 피어슨 상관계수가 가장 높은 변수? 모두 아니었습니다.

그러다 Feature importance를 공부하던 중 무릎을 쳤습니다. 핵심은 **불순도 감소량(impurity decrease)** 이었습니다. scikit-learn의 의사결정나무 모델에서 parameter `splitter = 'best'`(default)로 설정했을 경우, 불순도를 가장 크게 감소시키는 feature와 threshold를 탐색해 분기 기준을 결정합니다.

간단한 회귀 트리 모델을 생성하고 위 주장을 검증해보겠습니다. 앞서 분할한 train data를 학습합니다. 최상위 root node의 분기 기준은 아래와 같습니다.


```python
model_dt = DecisionTreeRegressor(max_leaf_nodes=10, random_state=42, )
model_dt.fit(X_train, y_train)

print('Feature for splitting root node:', feature_names[model_dt.tree_.feature[0]])
print('Threshold for splitting root node:', model_dt.tree_.threshold[0])
```

    Feature for splitting root node: RM
    Threshold for splitting root node: 6.940999984741211
    

Best splitter를 찾기 위해 brute force 방법을 사용했습니다.
1. 독립변수 A를 선택한다.  
2. 학습 데이터에서 해당 독립변수의 unique values를 추출하고 내림차순으로 정렬한다.  
3. 모든 인접한 두 unique values 사이의 평균을 계산한다. 예를 들어, uniuqe values가 `[4, 3, 2, 1]`이라면, 계산 결과는 `[3.5, 2.5, 1.5]`이다.  
4. 3번에서 얻은 리스트의 각 원소를 threshold value로 두고 split했을 때 불순도 감소량을 관찰한다. 불순도를 가장 크게 감소시키는 원소를 독립변수 A의 threshold로 정한다.(불순도 감소량 산출식은 [지난 포스트](https://ywkim92.github.io/machine_learning/feature_importance/#feature-importance) 참조)  
5. 모든 독립변수에 대해 1~4번을 반복한다.  
6. 불순도 감소량이 가장 큰 독립변수와 그 threshold로 해당 노드의 샘플을 split한다.

위 방법에 따라 코딩하고 root node의 분기 기준을 도출합니다.


```python
def best_splitter_reg(node_samples, feature_names, criterion):
    '''Find the best splitter for a decision tree regressor.
    
    node_samples: train data(pandas DataFrame) whose the last column is label 
	and its column name is 'label'
    '''
    n = node_samples.shape[0] # the number of samples for training
    dic= dict()
    for col in feature_names:
        tr = node_samples[[col, 'label']]
        nt = tr.shape[0]
        it = tr['label'].var()
        values = np.unique(tr[col])
        if values.size==1:
            continue
        thresholds = np.array([values[i:i+2].mean() for i in range(values.size-1)])[::-1]

        if values.size <=10:
            iter_lim = thresholds.size
        else:
            iter_lim = thresholds.size//2

        i_max = 0
        th_tag = 0
        lim = 0
        for th in thresholds:
            # right node
            trr = tr[(tr[col]>th)]
            ntr = trr.shape[0]
            # calculate impurity for regression tree; 'mae' or 'mse'(default)
            if criterion == 'mae':
                ir = np.abs(trr['label'] - np.median(trr['label'])).mean()
            else:
                ir = trr['label'].var()

            # left node
            trl = tr[tr[col]<=th]
            ntl = trl.shape[0]
            if criterion == 'mae':
                il = np.abs(trl['label'] - np.median(trl['label'])).mean()
            else:
                il = trl['label'].var()

            # calculate weighted impurity decrease
            i = (nt / n) * ( it - (ntl / nt) * il - (ntr / nt)* ir )

            if i > i_max:
                i_max = i
                th_tag = th

            lim+=1
            if lim==iter_lim:
                break

        dic[col] = (i_max, th_tag)
    
    best_feature_ = max(dic, key = lambda x: dic[x][0])
    best_threshold_ = dic[best_feature_][1]
    return best_feature_, best_threshold_
```

모든 경우의 수에 대해 불순도 감소량을 산출하다 보니 scikit-learn 모델에 비해 시간이 너무 오래 걸렸습니다. unique value의 중앙값 부근에서는 불순도가 정체되는 양상이 확인되므로 unique values의 상위 50%에 대해서만 계산하게끔 처리했습니다. `iter_lim`을 정의하고 그 값을 넘으면 for문을 종료했습니다. best splitter를 찾는 알고리즘에 대해서는 추가 리서치할 계획입니다.

정의한 `best_splitter_reg` 함수로 도출한 분기 기준은 scikit-learn의 결과와 동일합니다.


```python
train = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1,1))), 
		columns=feature_names.tolist()+['label'])

print('The best splitter of the root node(feature, threshold):',
		best_splitter_reg(train, feature_names, model_dt.criterion))
```

    The best splitter of the root node(feature, threshold): ('RM', 6.941)
    

root node에서 분기된 두 노드에 대해 검증하여도 scikit-learn과 같은 결과를 얻습니다.


```python
print('* Using scikit-learn attributes')
print('Feature for splitting the 1st node:', feature_names[model_dt.tree_.feature[1]])
print('Threshold for splitting the 1st node:', model_dt.tree_.threshold[1],)

print('\n* Using implementation')
train_node1 = train[train['RM']<=6.941]
print('The best splitter of the 1st node(feature, threshold):', 
		best_splitter_reg(train_node1, feature_names, model_dt.criterion))
```

    * Using scikit-learn attributes
    Feature for splitting the 1st node: LSTAT
    Threshold for splitting the 1st node: 14.400000095367432
    
    * Using implementation
    The best splitter of the 1st node(feature, threshold): ('LSTAT', 14.399999999999999)
    


```python
print('* Using scikit-learn attributes')
print('Feature for splitting the 2nd node:', feature_names[model_dt.tree_.feature[2]])
print('Threshold for splitting the 2nd node:', model_dt.tree_.threshold[2],)

print('\n* Using implementation')
train_node2 = train[train['RM']>6.941]
print('The best splitter of the 1st node(feature, threshold):', 
		best_splitter_reg(train_node2, feature_names, model_dt.criterion))
```

    * Using scikit-learn attributes
    Feature for splitting the 2nd node: RM
    Threshold for splitting the 2nd node: 7.437000036239624
    
    * Using implementation
    The best splitter of the 1st node(feature, threshold): ('RM', 7.436999999999999)
    

`criterion = 'mae'`으로 설정했을 때도 마찬가지입니다.


```python
model_dt1 = DecisionTreeRegressor(max_leaf_nodes=10, random_state=42, criterion='mae')
model_dt1.fit(X_train, y_train)

print('when criterion = "mae"')
print('* Using scikit-learn attributes')
print('Feature for splitting root node:', feature_names[model_dt1.tree_.feature[0]])
print('Threshold for splitting root node:', model_dt1.tree_.threshold[0])

print('\n* Using implementation')
print('The best splitter of the root node(feature, threshold):',
		best_splitter_reg(train, feature_names, model_dt1.criterion))
```

    when criterion = "mae"
    * Using scikit-learn attributes
    Feature for splitting root node: RM
    Threshold for splitting root node: 6.797000169754028
    
    * Using implementation
    The best splitter of the root node(feature, threshold): ('RM', 6.797)
    

## 노드에 속하는 샘플에 부여되는 label 값

* 회귀: 해당 노드에 속하는 샘플들의 평균입니다. 각 노드에 부여되는 label 값을 산출하는 코드를 짜서 검증해봅니다.


```python
def tree_value_reg(X_train, y_train, tree_regressor):
    result = []
    n_node = tree_regressor.tree_.node_count
    for i in range(n_node):
        value = y_train[tree_regressor.decision_path(X_train).toarray()[:, i]==1].mean()
        result.append(value)
    return np.array(result)
```


```python
tree_values_sklearn = model_dt.tree_.value.flatten()
tree_values_imp = tree_value_reg(X_train, y_train, model_dt)
print('Sklean == my implementation:',np.allclose(tree_values_sklearn, tree_values_imp))
```

    Sklean == my implementation: True
    

* 분류: 각 class를 index로 하며 해당 노드에 속하는 class별 샘플의 개수를 원소로 하는 list입니다. 이는 predict 단계에서 class probabilities를 계산할 때 사용됩니다. 예를 들어 이진 분류 모델에서 leaf node N의 value가 `[2, 8]`이라면, leaf node N에 최종 도달한 샘플이 class 0일 확률은 0.2, class 1일 확률은 0.8로 계산됩니다. threshold가 0.5라면 이 샘플은 class 1로 분류되겠지요.  
코드로 구현한 후 그 결과를 scikit-learn 결과와 비교해봅니다. 검증 결과 동일합니다.


```python
def tree_value_clf(X_train, y_train, tree_clf):
    n_class = tree_clf.n_classes_
    result = np.array([]).reshape(-1, n_class)
    n_node = tree_clf.tree_.node_count
    
    for i in range(n_node):
        uniq, value = np.unique(y_train[tree_clf.decision_path(X_train).toarray()[:, i]==1], 
			return_counts=True)
        if value.size == n_class:
            result = np.vstack((result, value))
        else:
            value_ = np.zeros(n_class)
            value_[np.array(uniq, dtype=int)] = value
            result = np.vstack((result, value_))
        
    return np.array(result)
```

검증에 사용할 이진 분류 데이터 샘플(유방암 데이터)과 의사결정나무 분류 모델을 불러옵니다. 검증 결과 동일합니다.


```python
data_clf = load_breast_cancer()
X_clf = data_clf['data']
y_clf = data_clf['target']

model_clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=42)
model_clf.fit(X_clf, y_clf)

tree_values_sklearn = model_clf.tree_.value[:, 0, :]
tree_values_imp = tree_value_clf(X_clf, y_clf, model_clf)
print('Sklean == my implementation:',np.allclose(tree_values_sklearn, tree_values_imp))
```

    Sklean == my implementation: True
    

# Tree 그리기

[지난 포스트](https://ywkim92.github.io/machine_learning/feature_importance/)와 이번 포스트에서 의사결정나무 알고리즘의 train 과정을 대략적으로 살펴보았습니다. 다룬 내용을 바탕으로 모델의 트리 구조를 구현(시각화)하는 코드를 작성 및 검증하며 마무리하겠습니다.

* `_parent`: 각 노드를 index로 하며 각 노드의 부모 노드를 원소로 가지는 array를 리턴하는 함수. root node의 부모 노드는 자기 자신  
* `_depth`: 각 노드를 index로 하며 각 노드의 깊이를 원소로 가지는 array를 리턴하는 함수. root node의 깊이는 0  
* `print_tree_vertical`: 위에서 아래로 내려가는 방향  
* `print_tree`: 왼쪽에서 오른쪽으로 뻗어 감. treelib 라이브러리 이용.  


```python
class represent_tree:
    def __init__(self, fitted_model):
        self.model = fitted_model
        self.n_nodes = self.model.tree_.node_count
        self.left = self.model.tree_.children_left.tolist()
        self.right = self.model.tree_.children_right.tolist()
    
    def _depth(self):
        depth_list = [0] * self.n_nodes
        for n in range(1, self.n_nodes):
            if n%2==0:
                parent = self.right.index(n)
                depth_list[n] = depth_list[parent]+1
            else:
                parent = self.left.index(n)
                depth_list[n] = depth_list[parent]+1
        return np.array(depth_list)

    def _parent(self):
        parent_list = [0] * self.n_nodes

        for n in range(1, self.n_nodes):
            if n%2==0:
                parent = self.right.index(n)
                parent_list[n] = parent
            else:
                parent = self.left.index(n)
                parent_list[n] = parent
        return np.array(parent_list)
    
    def _isfloat(self, num, d):
        if (type(num) == float) | (type(num) == np.float64):
            return round(num, d)
        else:
            return num
    
    def print_tree_vertical(self, round_digit = 3, feat_names=None):
        max_depth = self.model.tree_.max_depth
        parent_node = self._parent()
        node_depth = self._depth()
        n_feats = self.model.n_features_
        if feat_names is None:
            feat_names = list('F{}'.format(str(i).zfill(len(str(n_feats)))) for i in range(n_feats))
        feat_in_node = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        impurity = self.model.tree_.impurity
        samples = self.model.tree_.n_node_samples
        if 'Reg' in str(self.model):
            values = self.model.tree_.value.flatten()
        else:
            values = self.model.tree_.value[:, 0, :]

        for dep in range(max_depth+1):
            print('Depth', dep)
            nodes = np.where(node_depth==dep)[0]
            child_parent = sorted((zip(nodes, [parent_node[i] for i in nodes])), key = lambda x: (x[1],x[0]) )
            #print(child_parent)
            nodes1 = [i[0] for i in child_parent]
            for node in nodes1:
                if feat_in_node[node] == -2:
                    print('node #{} *leaf*'.format(node), model_dt.criterion, '=', round(impurity[node], round_digit), 
                      'samples =', samples[node], 'value =', self._isfloat(values[node], round_digit), 'parent_node =', parent_node[node], end=' || ')
                else:
                    print('node #{}'.format(node), feat_names[feat_in_node[node]], '<=', round(threshold[node], round_digit), model_dt.criterion, '=', round(impurity[node], round_digit), 
                          'samples =', samples[node], 'value =', self._isfloat(values[node], round_digit), 'parent_node =', parent_node[node], end=' || ')
            print('\n')
            
    def print_tree(self, round_digit = 3, feat_names=None):
        max_depth = self.model.tree_.max_depth
        parent_node = self._parent()
        node_depth = self._depth()
        n_feats = self.model.n_features_
        if feat_names is None:
            feat_names = list('F{}'.format(str(i).zfill(len(str(n_feats)))) for i in range(n_feats))
        feat_in_node = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        impurity = self.model.tree_.impurity
        samples = self.model.tree_.n_node_samples
        if 'Reg' in str(self.model):
            values = self.model.tree_.value.flatten()
        else:
            values = self.model.tree_.value[:, 0, :]
        
        tree = Tree()
        
        for node in range(self.n_nodes):
            
            if node == 0:
                node_list = ['node #{}'.format(node), feat_names[feat_in_node[node]], '<=', round(threshold[node], round_digit), model_dt.criterion, '=', 
                             round(impurity[node], round_digit), 'samples =', samples[node], 'value =', self._isfloat(values[node], round_digit),]
                tree.create_node(' '.join([str(i) for i in node_list]), node)
            elif feat_in_node[node] == -2:
                node_list = ['node #{}'.format(node), model_dt.criterion, '=', round(impurity[node], round_digit), 
                      'samples =', samples[node], 'value =', self._isfloat(values[node], round_digit), '*leaf*']
                tree.create_node(' '.join([str(i) for i in node_list]), node, parent = parent_node[node])
            else:
                node_list = ['node #{}'.format(node), feat_names[feat_in_node[node]], '<=', round(threshold[node], round_digit), model_dt.criterion, '=', 
                             round(impurity[node], round_digit), 'samples =', samples[node], 'value =', self._isfloat(values[node], round_digit),]
                tree.create_node(' '.join([str(i) for i in node_list]), node, parent = parent_node[node])
        tree.show()
            
    def print_nodes(self, feat_names=None):
        max_depth = self.model.tree_.max_depth
        parent_node = self._parent()
        node_depth = self._depth()
        feat_in_node = self.model.tree_.feature

        for dep in range(max_depth+1):
            print('Depth', dep)
            nodes = np.where(node_depth==dep)[0]
            child_parent = sorted((zip(nodes, [parent_node[i] for i in nodes])), key = lambda x: (x[1],x[0]) )
            nodes1 = [i[0] for i in child_parent]
            for node in nodes1:
                if feat_in_node[node] == -2:
                    print('n #{} *leaf*'.format(node),  'parent =', parent_node[node], end=' || ')
                else:
                    print('n #{}'.format(node),  'parent =', parent_node[node], end=' || ')
            print('\n')  
```

## 회귀 트리

scikit-learn `plot_tree`의 결과와 동일합니다. 물론 예쁘진 않지만요&#128514;


```python
rt = represent_tree(model_dt)
rt.print_tree(round_digit=3, feat_names=data['feature_names'])
```

    node #0 RM <= 6.941 mse = 86.873 samples = 404 value = 22.797
    ├── node #1 LSTAT <= 14.4 mse = 40.321 samples = 337 value = 19.947
    │   ├── node #3 DIS <= 1.385 mse = 25.693 samples = 203 value = 23.325
    │   │   ├── node #7 CRIM <= 10.109 mse = 91.577 samples = 4 value = 44.475
    │   │   │   ├── node #17 mse = 0.0 samples = 3 value = 50.0 *leaf*
    │   │   │   └── node #18 mse = -0.0 samples = 1 value = 27.9 *leaf*
    │   │   └── node #8 RM <= 6.543 mse = 15.197 samples = 199 value = 22.899
    │   │       ├── node #10 mse = 12.03 samples = 43 value = 27.498 *leaf*
    │   │       └── node #9 mse = 8.635 samples = 156 value = 21.632 *leaf*
    │   └── node #4 CRIM <= 6.926 mse = 19.005 samples = 134 value = 14.829
    │       ├── node #11 mse = 10.401 samples = 76 value = 17.062 *leaf*
    │       └── node #12 mse = 15.188 samples = 58 value = 11.903 *leaf*
    └── node #2 RM <= 7.437 mse = 74.684 samples = 67 value = 37.131
        ├── node #5 LSTAT <= 15.765 mse = 38.306 samples = 41 value = 32.363
        │   ├── node #13 mse = 21.586 samples = 39 value = 33.3 *leaf*
        │   └── node #14 mse = 13.69 samples = 2 value = 14.1 *leaf*
        └── node #6 PTRATIO <= 19.65 mse = 39.671 samples = 26 value = 44.65
            ├── node #15 mse = 19.727 samples = 25 value = 45.56 *leaf*
            └── node #16 mse = -0.0 samples = 1 value = 21.9 *leaf*
    
    


```python
plt.figure(figsize=(20,10))
plot_tree(model_dt, node_ids=True, fontsize=14, feature_names=data['feature_names'], filled=True)
plt.show()
```


    
![output_35_0](https://user-images.githubusercontent.com/66911578/132970658-a428d7f3-92b8-48bf-baa0-9dde8ed40be5.png)
    


## 분류 트리

역시 scikit-learn `plot_tree`의 결과와 동일합니다.


```python
rt_clf = represent_tree(model_clf)
rt_clf.print_tree(round_digit=3, feat_names=data_clf['feature_names'])
```

    node #0 worst radius <= 16.795 mse = 0.468 samples = 569 value = [212. 357.]
    ├── node #1 worst concave points <= 0.136 mse = 0.159 samples = 379 value = [ 33. 346.]
    │   ├── node #3 mse = 0.03 samples = 333 value = [  5. 328.] *leaf*
    │   └── node #4 worst texture <= 25.67 mse = 0.476 samples = 46 value = [28. 18.]
    │       ├── node #5 worst area <= 810.3 mse = 0.332 samples = 19 value = [ 4. 15.]
    │       │   ├── node #11 mse = 0.124 samples = 15 value = [ 1. 14.] *leaf*
    │       │   └── node #12 mse = 0.375 samples = 4 value = [3. 1.] *leaf*
    │       └── node #6 mean concavity <= 0.097 mse = 0.198 samples = 27 value = [24.  3.]
    │           ├── node #13 mean texture <= 19.435 mse = 0.5 samples = 6 value = [3. 3.]
    │           │   ├── node #15 mse = 0.0 samples = 3 value = [0. 3.] *leaf*
    │           │   └── node #16 mse = 0.0 samples = 3 value = [3. 0.] *leaf*
    │           └── node #14 mse = 0.0 samples = 21 value = [21.  0.] *leaf*
    └── node #2 mean texture <= 16.11 mse = 0.109 samples = 190 value = [179.  11.]
        ├── node #7 worst concave points <= 0.145 mse = 0.498 samples = 17 value = [8. 9.]
        │   ├── node #10 mse = 0.0 samples = 8 value = [8. 0.] *leaf*
        │   └── node #9 mse = 0.0 samples = 9 value = [0. 9.] *leaf*
        └── node #8 worst smoothness <= 0.088 mse = 0.023 samples = 173 value = [171.   2.]
            ├── node #17 mse = 0.0 samples = 1 value = [0. 1.] *leaf*
            └── node #18 mse = 0.012 samples = 172 value = [171.   1.] *leaf*
    
    


```python
plt.figure(figsize=(20,10))
plot_tree(model_clf, node_ids=True, fontsize=14, feature_names=data_clf['feature_names'], filled=True)
plt.show()
```


    
![output_39_0](https://user-images.githubusercontent.com/66911578/132970664-eb5876c2-c5f4-4709-8dd2-add2743a47cf.png)
    
[source of teaser](https://unsplash.com/photos/C7B-ExXpOIE?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
<br>

<br />
[Scroll to Top](#){: .btn .btn--primary .btn-small .align-center}
