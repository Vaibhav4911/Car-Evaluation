#                     Machine Learning  
# Lab : 8   
                                      

Submitted by:

Name : Vaibhav Agarwal

Rollno.: 20103155

Submitted to : Dr Jagdeep Kaur


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df=pd.read_csv("C:\\Users\\USER\\Downloads\\car_evaluation.csv")
df.head()
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
      <th>vhigh</th>
      <th>vhigh.1</th>
      <th>2</th>
      <th>2.1</th>
      <th>small</th>
      <th>low</th>
      <th>unacc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (1727, 7)




```python
col_names = ['purchase', 'Abbreviation', 'doors', 'persons', 'lugBoot', 'safety', 'target']


df.columns = col_names

col_names

```




    ['purchase', 'Abbreviation', 'doors', 'persons', 'lugBoot', 'safety', 'target']




```python
df.head()
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
      <th>purchase</th>
      <th>Abbreviation</th>
      <th>doors</th>
      <th>persons</th>
      <th>lugBoot</th>
      <th>safety</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
      <td>unacc</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>high</td>
      <td>unacc</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1727 entries, 0 to 1726
    Data columns (total 7 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   purchase      1727 non-null   object
     1   Abbreviation  1727 non-null   object
     2   doors         1727 non-null   object
     3   persons       1727 non-null   object
     4   lugBoot       1727 non-null   object
     5   safety        1727 non-null   object
     6   target        1727 non-null   object
    dtypes: object(7)
    memory usage: 94.6+ KB
    


```python
col_names = ['purchase', 'Abbreviation', 'doors', 'persons', 'lugBoot', 'safety', 'target']


for col in col_names:
    
    print(df[col].value_counts()) 
```

    high     432
    med      432
    low      432
    vhigh    431
    Name: purchase, dtype: int64
    high     432
    med      432
    low      432
    vhigh    431
    Name: Abbreviation, dtype: int64
    3        432
    4        432
    5more    432
    2        431
    Name: doors, dtype: int64
    4       576
    more    576
    2       575
    Name: persons, dtype: int64
    med      576
    big      576
    small    575
    Name: lugBoot, dtype: int64
    med     576
    high    576
    low     575
    Name: safety, dtype: int64
    unacc    1209
    acc       384
    good       69
    vgood      65
    Name: target, dtype: int64
    


```python
df['target'].value_counts()
```




    unacc    1209
    acc       384
    good       69
    vgood      65
    Name: target, dtype: int64




```python
df.isnull().sum()
```




    purchase        0
    Abbreviation    0
    doors           0
    persons         0
    lugBoot         0
    safety          0
    target          0
    dtype: int64




```python
X = df.drop(['target'], axis=1)

y = df['target']
```


```python
X.head()
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
      <th>purchase</th>
      <th>Abbreviation</th>
      <th>doors</th>
      <th>persons</th>
      <th>lugBoot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>high</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

```


```python
X_train.shape, X_test.shape
```




    ((1157, 6), (570, 6))




```python
X_train.dtypes
```




    purchase        object
    Abbreviation    object
    doors           object
    persons         object
    lugBoot         object
    safety          object
    dtype: object




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
      <th>purchase</th>
      <th>Abbreviation</th>
      <th>doors</th>
      <th>persons</th>
      <th>lugBoot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>5more</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
    </tr>
    <tr>
      <th>48</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>3</td>
      <td>more</td>
      <td>med</td>
      <td>med</td>
    </tr>
    <tr>
      <th>468</th>
      <td>high</td>
      <td>vhigh</td>
      <td>3</td>
      <td>4</td>
      <td>small</td>
      <td>med</td>
    </tr>
    <tr>
      <th>155</th>
      <td>vhigh</td>
      <td>high</td>
      <td>3</td>
      <td>more</td>
      <td>med</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>med</td>
      <td>high</td>
      <td>4</td>
      <td>more</td>
      <td>small</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>



We can see that all the variables are ordinal categorical data type.

Now, I will encode the categorical variables.


```python
pip install --upgrade category_encoders
```

    Requirement already satisfied: category_encoders in c:\programdata\anaconda3\lib\site-packages (2.4.0)
    Requirement already satisfied: statsmodels>=0.9.0 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (0.12.2)
    Requirement already satisfied: numpy>=1.14.0 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (1.20.3)
    Requirement already satisfied: scipy>=1.0.0 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (1.7.1)
    Requirement already satisfied: scikit-learn>=0.20.0 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (0.24.2)
    Requirement already satisfied: pandas>=0.21.1 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (1.3.4)
    Requirement already satisfied: patsy>=0.5.1 in c:\programdata\anaconda3\lib\site-packages (from category_encoders) (0.5.2)
    Requirement already satisfied: pytz>=2017.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.21.1->category_encoders) (2021.3)
    Requirement already satisfied: python-dateutil>=2.7.3 in c:\programdata\anaconda3\lib\site-packages (from pandas>=0.21.1->category_encoders) (2.8.2)
    Requirement already satisfied: six in c:\programdata\anaconda3\lib\site-packages (from patsy>=0.5.1->category_encoders) (1.16.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn>=0.20.0->category_encoders) (2.2.0)
    Requirement already satisfied: joblib>=0.11 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn>=0.20.0->category_encoders) (1.1.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import category_encoders as ce
```


```python
encoder = ce.OrdinalEncoder(cols=['purchase', 'Abbreviation', 'doors', 'persons', 'lugBoot', 'safety'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

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
      <th>purchase</th>
      <th>Abbreviation</th>
      <th>doors</th>
      <th>persons</th>
      <th>lugBoot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>468</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>155</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.head()
```




    83      unacc
    48      unacc
    468     unacc
    155     unacc
    1043    unacc
    Name: target, dtype: object




```python
y_test.head()
```




    599     unacc
    932     unacc
    628     unacc
    1497      acc
    1262    unacc
    Name: target, dtype: object




```python
X_test.head()
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
      <th>purchase</th>
      <th>Abbreviation</th>
      <th>doors</th>
      <th>persons</th>
      <th>lugBoot</th>
      <th>safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>599</th>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>932</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>628</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1262</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We now have training and test set ready for model building.

# Decision Tree Classifier with criterion gini index


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

clf_gini.fit(X_train, y_train)
```




    DecisionTreeClassifier(max_depth=3, random_state=0)




```python
y_pred_gini = clf_gini.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
```

    Model accuracy score with criterion gini index: 0.8053
    


```python
y_pred_train_gini = clf_gini.predict(X_train)

y_pred_train_gini
```




    array(['unacc', 'unacc', 'unacc', ..., 'unacc', 'unacc', 'acc'],
          dtype=object)




```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
```

    Training-set accuracy score: 0.7848
    


```python
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
```

    Training set score: 0.7848
    Test set score: 0.8053
    

Here, the training-set accuracy score is 0.7865 while the test-set accuracy to be 0.8021. These two values are quite comparable. So, there is no sign of overfitting.


```python
# instantiate the DecisionTreeClassifier model with criterion entropy

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)
```




    DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)



Predict the Test set results with criterion entropy


```python
y_pred_en = clf_en.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))
```

    Model accuracy score with criterion entropy: 0.8053
    

Compare the train-set and test-set accuracy


Now, I will compare the train-set and test-set accuracy to check for overfitting.


```python
y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en
```




    array(['unacc', 'unacc', 'unacc', ..., 'unacc', 'unacc', 'acc'],
          dtype=object)




```python
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))
```

    Training-set accuracy score: 0.7848
    

Check for overfitting and underfitting


```python
print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))
```

    Training set score: 0.7848
    Test set score: 0.8053
    


```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)
```

    Confusion matrix
    
     [[ 71   0  56   0]
     [ 18   0   0   0]
     [ 11   0 388   0]
     [ 26   0   0   0]]
    


```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_en))
```

                  precision    recall  f1-score   support
    
             acc       0.56      0.56      0.56       127
            good       0.00      0.00      0.00        18
           unacc       0.87      0.97      0.92       399
           vgood       0.00      0.00      0.00        26
    
        accuracy                           0.81       570
       macro avg       0.36      0.38      0.37       570
    weighted avg       0.74      0.81      0.77       570
    
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\metrics\_classification.py:1248: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    
