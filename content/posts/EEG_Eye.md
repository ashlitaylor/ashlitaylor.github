---
title: Eye State Classification
subtitle: Python, Scikit-learn, Random Forests, SVC, KNN, Feature Selection, PCA, Temporal Analysis
excerpt: An assortment of classification models that evaluates the future potential of predicting whether or not a person's eyes are open or closed using EEG sensor data. 
date: 999
thumb_img_path: images\EEgEye.png
template: page
---

<header> 
    <h1 align="center"> Classification of Eye State via Supervised Learning using the <br> Python Scikit Learn Library </h1>
    <h6 align="center"> Data Exploration, Data Cleaning, Hyperparameter Tuning and Feature Selection </h6>
</header>

In this project I evaluated three classifier models to accurately predict when an individual's eyes are open or closed using electroencephalography (EEG) measurements. I performed principal component analysis and recursive feature elimination to test if the models and experiments could be simplified via dimmensionality reduction. I chose the best model based on various classification metrics, and used it to perform temporal analysis on the data to further improve the model accuracy.

The dataset for this project originates from the UCI [EEG Eye State dataset](http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#), and was collected in 2013 by Roesler et al to understand how EEG measurements could be used to perform classification of a person's eye state. The goal of my project was to evaluate the results of the original experiment and explore some of the proposed future work.

<header> 
    <h3> <u><br>Background</u> </h3>
</header>
Eye state detection is the task of predicting the state of whether the eye is open or closed, and is useful for cognitive state classification. The variety of applications for predicting the eye state has a wide range, and includes emotion detection, autonomous vehicle driver drowsiness, and computer games.

Roesler et al proposed that eye state could be predicted by brain waves using electroencephalography (EEG) measurements and conducted an experiment to test this hypothesis in 2013. In their trial, they used a test subject who was instructed to open and close their eyes for varied intervals while wearing an Emotiv EEG Neuroheadset. The resulting dataset was constructed from one continuous 117 second long electroencephalography (EEG) measurement. Their experiments on the collected data involved 42 classification models from the Weka toolskit. The results of their expriments ultimately proved that it was possible to predict eye state using EEG measurements, and they favored an instance based model, KStar, that produced an accuracy rate of 97%.

<header> 
    <h3> <u>Motivation</u> </h3>
</header>
My original motivation to explore this classification task was a project that involved evaulating the EEG Eye State Dataset and comparing a selection of classification models. I developed an interest in the research material after I read Roesler's <a href="http://suendermann.com/su/pdf/aihls2013.pdf" target="_blank">paper</a> about the results of his team's experiments, and I created models to explore some of the future work proposed by Roesler et al by evaluating:

1. If the number of sensors can be decreased while still maintaining a low rate of classification error <br>
2. If a temporal relationship can be established and used to improve eye state predictions.

I created and compared [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), and [K-Nearest-Neighbor](https://scikit-learn.org/stable/modules/neighbors.html) classification models using the Scikit-Learn machine learning library, and performed principal component analysis, feature selection and temporal analysis.

<header> 
    <h3> <u>Library Imports</u> </h3>
</header>

I imported various libraries for this project.

- Numpy, since Scikit-Learn is built on numpy
- The sklearn neighbors, svm and ensemble machine learning libraries to build the respective KNN, SVC and Random Forest classification models.
- Pandas, matplotlib and seaborn to faciliate data exploration and visualization.
- Various classification performance measures.
- Preprocessing library to standardize the data.
- PCA and Feature Selection to perform feature analysis and selection.
- Time to measure how long each model takes to fit the data.
- Warnings to suppress unnecessary deprecation warnings

```python
#To facilitate data exploration
import numpy as np
from numpy import array
import pandas as pd
#Importing the libraries for KNN, SVC and Random Forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#For plots and visualizations
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#To evaluate classifcation performance
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
#Processing the data
from sklearn.preprocessing import StandardScaler, normalize
#Principal Component Analysis
from sklearn.decomposition import PCA
#Feature Selection
from sklearn.feature_selection import SelectFromModel, RFECV
#Printing model parameter lists
from pprint import pprint
#To time the models
import time
#Supress unnecessary deprecation warnings
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
```

<header> 
    <h3> <u>The Data</u> </h3>
</header>

The [EEG Eye State dataset](http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#) contains 14980 instances and consists of 14 EEG values and a value indicating the eye state. '1' indicates the eye-closed and '0' the eye-open state. They are stored in chronological order of collection.

#### Data Exploration

I read the CSV that the file is stored in into a pandas dataframe.

```python
# Reading the data
data = pd.read_csv('eeg_dataset.csv')
#print('Features in the data file: ',data.columns.values)
print("Number of instances in the  data file: ", len(data))
values = data.values
data.head()
```

    Number of instances in the  data file:  14980

<style>
    .table_wrapper{
        display: block;
        overflow-x:auto;
        white-space:nowrap;
    }
</style>

<div class = "table_wrapper">
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>X10</th>
      <th>X11</th>
      <th>X12</th>
      <th>X13</th>
      <th>X14</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4329.23</td>
      <td>4009.23</td>
      <td>4289.23</td>
      <td>4148.21</td>
      <td>4350.26</td>
      <td>4586.15</td>
      <td>4096.92</td>
      <td>4641.03</td>
      <td>4222.05</td>
      <td>4238.46</td>
      <td>4211.28</td>
      <td>4280.51</td>
      <td>4635.90</td>
      <td>4393.85</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4324.62</td>
      <td>4004.62</td>
      <td>4293.85</td>
      <td>4148.72</td>
      <td>4342.05</td>
      <td>4586.67</td>
      <td>4097.44</td>
      <td>4638.97</td>
      <td>4210.77</td>
      <td>4226.67</td>
      <td>4207.69</td>
      <td>4279.49</td>
      <td>4632.82</td>
      <td>4384.10</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4327.69</td>
      <td>4006.67</td>
      <td>4295.38</td>
      <td>4156.41</td>
      <td>4336.92</td>
      <td>4583.59</td>
      <td>4096.92</td>
      <td>4630.26</td>
      <td>4207.69</td>
      <td>4222.05</td>
      <td>4206.67</td>
      <td>4282.05</td>
      <td>4628.72</td>
      <td>4389.23</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4328.72</td>
      <td>4011.79</td>
      <td>4296.41</td>
      <td>4155.90</td>
      <td>4343.59</td>
      <td>4582.56</td>
      <td>4097.44</td>
      <td>4630.77</td>
      <td>4217.44</td>
      <td>4235.38</td>
      <td>4210.77</td>
      <td>4287.69</td>
      <td>4632.31</td>
      <td>4396.41</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4326.15</td>
      <td>4011.79</td>
      <td>4292.31</td>
      <td>4151.28</td>
      <td>4347.69</td>
      <td>4586.67</td>
      <td>4095.90</td>
      <td>4627.69</td>
      <td>4210.77</td>
      <td>4244.10</td>
      <td>4212.82</td>
      <td>4288.21</td>
      <td>4632.82</td>
      <td>4398.46</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

The figures below show the temporal plots of the features and output. There are four instances of transmission errors that are cited in the original paper that are outliers. These instances are easily identifiable from the large spikes in the plots.

```python
# create a temporal subplot for each time series
for i in range(values.shape[1]):
    fig, ax = plt.subplots(figsize = (10,3))
    ax.plot(values[:, i])
    ax.set_title(str(data.columns[i]) + ' temporal trace')
plt.show()
```

<style>
    .image_wrapper{
        display: block;
        overflow-y:auto;
        height:250px;

    }
</style>

<div class = "image_wrapper">

![png](images/eeg_images/output_7_0.png)
![png](images/eeg_images/output_7_1.png)
![png](images/eeg_images/output_7_2.png)
![png](images/eeg_images/output_7_3.png)
![png](images/eeg_images/output_7_4.png)
![png](images/eeg_images/output_7_5.png)
![png](images/eeg_images/output_7_6.png)
![png](images/eeg_images/output_7_7.png)
![png](images/eeg_images/output_7_8.png)
![png](images/eeg_images/output_7_9.png)
![png](images/eeg_images/output_7_10.png)
![png](images/eeg_images/output_7_11.png)
![png](images/eeg_images/output_7_12.png)
![png](images/eeg_images/output_7_13.png)
![png](images/eeg_images/output_7_14.png)

</div>

Box and whisker plots of the measurements further confirm the presence of these anomalous measurements.

```python
x_data = data.loc[:, data.columns != "y"]
#y_data = data.loc[:, "y"]
#Box-Whisker
red_square = dict(markerfacecolor='r', marker='s')
for column in x_data:
    fig, ax = plt.subplots()
    ax.set_title('Plot of feature '+ column)
    ax.boxplot(x_data[column], flierprops=red_square)
```

<div class = "image_wrapper">

![png](images/eeg_images/output_9_0.png)
![png](images/eeg_images/output_9_1.png)
![png](images/eeg_images/output_9_2.png)
![png](images/eeg_images/output_9_3.png)
![png](images/eeg_images/output_9_4.png)
![png](images/eeg_images/output_9_5.png)
![png](images/eeg_images/output_9_6.png)
![png](images/eeg_images/output_9_7.png)
![png](images/eeg_images/output_9_8.png)
![png](images/eeg_images/output_9_9.png)
![png](images/eeg_images/output_9_10.png)
![png](images/eeg_images/output_9_11.png)
![png](images/eeg_images/output_9_12.png)
![png](images/eeg_images/output_9_13.png)

</div>

#### Cleaning

I identified and removed the four anomalous measurements by removing the instances with values that are ten standard deviations or more from the median. I saved the values to a new CSV file, and read it into a pandas dataframe for use in the models. I separated the columns for the predictor variables (x_data) and the dependent variable (y_data).

```python
from numpy import mean
from numpy import median
from numpy import std
from numpy import delete
from numpy import savetxt
# step over each EEG column
for i in range(values.shape[1] - 1):
    # calculate column median and standard deviation
    data_median, data_std = median(values[:,i]), std(values[:,i])
    # define outlier bounds
    cut_off = data_std * 10
    lower, upper = data_median - cut_off, data_median + cut_off
    # remove small values
    too_small = [j for j in range(values.shape[0]) if values[j,i] < lower]
    values = delete(values, too_small, 0)
    # remove large values
    too_large = [j for j in range(values.shape[0]) if values[j,i] > upper]
    values = delete(values, too_large, 0)
# save the results to a new file
savetxt('EEG_Eye_State_no_outliers.csv', values, delimiter=',', header = str(list(data.columns.values)))

#Read in the cleaned data
data_sans_outliers = pd.read_csv('EEG_Eye_State_no_outliers.csv')
for i in range(0,15):
    #print(i)
    #print(data.columns[i])
    data_sans_outliers.columns = data.columns
    #print(data_sans_outliers.columns[i])
print('Number of instances in the cleaned data set: ',len(data_sans_outliers))
x_data = data_sans_outliers.iloc[:,0:14]
y_data = data_sans_outliers.iloc[:, 14]

values = data_sans_outliers.values
```

    Number of instances in the cleaned data set:  14976

The temporal plots of the features of the cleaned dataset are shown below. Now that the erroneous values have been removed, the traces of the EEG measurements are much easier to discern.

```python
# create a temporal subplot for each time series
for i in range(values.shape[1]):
    fig, ax = plt.subplots(figsize = (10,3))
    ax.plot(values[:, i])
    ax.set_title(str(data_sans_outliers.columns[i]) + ' temporal trace')
plt.show()
```
<div class = "image_wrapper">

![png](images/eeg_images/output_13_0.png)
![png](images/eeg_images/output_13_1.png)
![png](images/eeg_images/output_13_2.png)
![png](images/eeg_images/output_13_3.png)
![png](images/eeg_images/output_13_4.png)
![png](images/eeg_images/output_13_5.png)
![png](images/eeg_images/output_13_6.png)
![png](images/eeg_images/output_13_7.png)
![png](images/eeg_images/output_13_8.png)
![png](images/eeg_images/output_13_9.png)
![png](images/eeg_images/output_13_10.png)
![png](images/eeg_images/output_13_11.png)
![png](images/eeg_images/output_13_12.png)
![png](images/eeg_images/output_13_13.png)
![png](images/eeg_images/output_13_14.png)

</div>

I created histograms and box-whisker plots of the features of the cleaned dataset below.

```python
#Histograms
fig = plt.figure()
for i in x_data:
    fig,ax =plt.subplots()
    ax.set_title('Histogram of feature ' + i)
    ax.hist(x_data[i], bins = 20)
    #plt.hist(x_data[i], bins = 20)
```

    <Figure size 432x288 with 0 Axes>
<div class = "image_wrapper">

![png](images/eeg_images/output_15_1.png)
![png](images/eeg_images/output_15_2.png)
![png](images/eeg_images/output_15_3.png)
![png](images/eeg_images/output_15_4.png)
![png](images/eeg_images/output_15_5.png)
![png](images/eeg_images/output_15_6.png)
![png](images/eeg_images/output_15_7.png)
![png](images/eeg_images/output_15_8.png)
![png](images/eeg_images/output_15_9.png)
![png](images/eeg_images/output_15_10.png)
![png](images/eeg_images/output_15_11.png)
![png](images/eeg_images/output_15_12.png)
![png](images/eeg_images/output_15_13.png)
![png](images/eeg_images/output_15_14.png)

</div>

```python
#Box-Whisker
red_square = dict(markerfacecolor='r', marker='s')
for column in x_data:
    fig, ax = plt.subplots()
    ax.set_title('Plot of feature '+ column)
    ax.boxplot(x_data[column], flierprops=red_square)

```

<div class = "image_wrapper">

![png](images/eeg_images/output_16_0.png)
![png](images/eeg_images/output_16_1.png)
![png](images/eeg_images/output_16_2.png)
![png](images/eeg_images/output_16_3.png)
![png](images/eeg_images/output_16_4.png)
![png](images/eeg_images/output_16_5.png)
![png](images/eeg_images/output_16_6.png)
![png](images/eeg_images/output_16_7.png)
![png](images/eeg_images/output_16_8.png)
![png](images/eeg_images/output_16_9.png)
![png](images/eeg_images/output_16_10.png)
![png](images/eeg_images/output_16_11.png)
![png](images/eeg_images/output_16_12.png)
![png](images/eeg_images/output_16_13.png)

</div>

#### Getting the data ready - Split Train and Test

I split the data using a 70:30 ratio into a training set, and a test set. I specified the random_state seed to ensure that the results are reproducible. At this stage, I was not evaluating any temporal dependencies, so I set the shuffle parameter to True.

```python
random_state = 100
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.7, random_state = random_state, shuffle = True)
```

<header> 
    <h3> <u>The Models</u> </h3>
</header>

### Random Forest

The first model I built was a Random Forest Classification model using the default scikit-learn hyperparameters.

```python
randFor = RandomForestClassifier() #default n_estimators = 10
randModel = randFor.fit(x_train, y_train)

print("Default Random Forest parmeters in use:\n")
pprint(randFor.get_params())
```

    Default Random Forest parmeters in use:

    {'bootstrap': True,
     'class_weight': None,
     'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'max_leaf_nodes': None,
     'min_impurity_decrease': 0.0,
     'min_impurity_split': None,
     'min_samples_leaf': 1,
     'min_samples_split': 2,
     'min_weight_fraction_leaf': 0.0,
     'n_estimators': 10,
     'n_jobs': None,
     'oob_score': False,
     'random_state': None,
     'verbose': 0,
     'warm_start': False}

With these default parameters, I was able to achieve a test accuracy > 89%, which matches the results reported in the research paper.

```python
#Testing the Random Forest accuracy on the testing set using the accuracy_score method
rfPreds_test = randModel.predict(x_test)
rfTestAccuracy = accuracy_score(y_test, rfPreds_test)
print("Random Forest test accuracy without hyperparameter tuning: ",round(rfTestAccuracy,3))
```

    Random Forest test accuracy without hyperparameter tuning:  0.89

#### Tuning the hyperparameters

###### Random Hyperparameter Grid

I first used the RandomizedSearchGridDV by creating a parameter grid to sample from during fitting. With randomized search, I am able to cast a wide net of model parameters to test without testing each combination individually, which would be more computationally taxing. Instead, I used the result from the randomized search to narrow down the range of model parameters so that I could later perform a more thorough search via GridSearch.

```python
#Number of trees in the Random Forest. Ranges from 10 to 500
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 5)]
#Number of features to consider at each split. Either 'auto' or 'sqrt'
max_features = ['auto', 'sqrt']
#Maximum number of levels in the tree. Ranges from 10 to 110 in increments of 10
max_depth = [int(x) for x in np.linspace(20, 200, num = 10)]
#Minimum number of samples required at each leaf node for split
min_samples_split = [2,5,10]
#Minimum number fo samples required at each leaf node
min_samples_leaf = [1,2,4]
#Selection metho for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```

Instantiating the random search and fitting it to the data:

```python
randSearchTimeStart = time.clock()
rand_search_RF = RandomizedSearchCV(randFor, random_grid, n_iter = 15, cv = 5)
rand_search_RF.fit(x_train, y_train)
randSearchTimed = time.clock() - randSearchTimeStart
hours = randSearchTimed // 3600
randSearchTimed %= 3600
minutes = randSearchTimed // 60
randSearchTimed %=60
seconds = randSearchTimed
print("Time to perform Randomized Search : ", round(hours,0) , "hours,", round(minutes,0) , "minutes,", round(seconds,2), " seconds")
```

    Time to perform Randomized Search :  0.0 hours, 4.0 minutes, 21.89  seconds

Viewing the best parameters from fitting random search

```python
print("Best Random Forest parmeters from Randomized Search:\n")
pprint(rand_search_RF.best_params_)

```

    Best Random Forest parmeters from Randomized Search:

    {'bootstrap': False,
     'max_depth': 20,
     'max_features': 'auto',
     'min_samples_leaf': 2,
     'min_samples_split': 2,
     'n_estimators': 100}

##### Performing gridsearch

After I gained a better idea of what the best parameters for the Random Forest might be, I used GridSearch to perform a more thorough test of model parameters. The result was a Random Forest classifier that had over 92% accuracy. This was an improvement on the 89% accuracy achieved using default parameters.

```python
param_grid_RF ={'bootstrap':[False],
            'n_estimators':[55,70,85],
            'max_depth':[160],
            'max_features':['sqrt'],
            'min_samples_leaf':[2,3,4],
            'min_samples_split':[2,4,6]}
gridSearchTimeStart = time.clock()
grid_search_RF = GridSearchCV(randFor, param_grid = param_grid_RF, cv = 10)
grid_search_RF.fit(x_train, y_train)
gridSearchTimed = time.clock() - gridSearchTimeStart
hours = gridSearchTimed // 3600
gridSearchTimed %= 3600
minutes = gridSearchTimed // 60
gridSearchTimed %=60
seconds = gridSearchTimed
print("Time to perform Grid Search : ", round(hours,0) , "hours,", round(minutes,0) , "minutes,", round(seconds,2), " seconds")
```

    Time to perform Grid Search :  0.0 hours, 29.0 minutes, 15.94  seconds

```python
print("Best parameters: ", grid_search_RF.best_params_, "\nBest Score: ", round(grid_search_RF.best_score_,3))
grid_search_RF.best_estimator_
```

    Best parameters:  {'bootstrap': False, 'max_depth': 160, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 85}
    Best Score:  0.928





    RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                           max_depth=160, max_features='sqrt', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=2, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=85,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)

Creating a Random Forest using the parameters from GridSearch:

```python
#randForGrid = RandomForestClassifier(n_estimators = grid_search_RF.best_params_['n_estimators'], max_depth = grid_search_RF.best_params_['max_depth'])
randForGrid = grid_search_RF.best_estimator_
randModelGrid = randForGrid.fit(x_train, y_train)
gridTrainPreds_RF = randModelGrid.predict(x_train)
gridTestPreds_RF = randModelGrid.predict(x_test)
gridTrainAccuracy_RF = accuracy_score(y_train, gridTrainPreds_RF)
gridTestAccuracy_RF = accuracy_score(y_test, gridTestPreds_RF)
print("RandomForest test accuracy without hyperparameter tuning: ",round(rfTestAccuracy,3))
print("RandomForest test accuracy with hyperparameter tuning: ",round(gridTestAccuracy_RF,3))
mean_test_score = grid_search_RF.cv_results_['mean_test_score'][grid_search_RF.best_index_]
mean_fit_time = grid_search_RF.cv_results_['mean_fit_time'][grid_search_RF.best_index_]
print("Mean test score for best combination: ",round(mean_test_score,3))
print("Mean fit time for best combination: ", round(mean_fit_time,3), " seconds")
```

    RandomForest test accuracy without hyperparameter tuning:  0.89
    RandomForest test accuracy with hyperparameter tuning:  0.929
    Mean test score for best combination:  0.928
    Mean fit time for best combination:  8.623  seconds

### Support Vector Machine

The next model I fit to the data was a Support Vector Classifier. I first preprocessed the data by standardizing it. This ensures that the estimated weights will update similarly rather than at different rates during the build process, which helps reduce the training time.

```python
scaler = StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

```python
#SVM Default parameters
svcClass = SVC()
svcModel = svcClass.fit(x_train_scaled, y_train)
print("Default SVC parmeters in use:\n")
pprint(svcModel.get_params())
```

    Default SVC parmeters in use:

    {'C': 1.0,
     'cache_size': 200,
     'class_weight': None,
     'coef0': 0.0,
     'decision_function_shape': 'ovr',
     'degree': 3,
     'gamma': 'auto_deprecated',
     'kernel': 'rbf',
     'max_iter': -1,
     'probability': False,
     'random_state': None,
     'shrinking': True,
     'tol': 0.001,
     'verbose': False}

Using the default parameters, I was able to achieve an 89% accuracy rate that matched the default random forest model. The testing accuracy (89%) is close to the training value (89.7%) which suggests that the model created using default parameters generalizes well.

```python
svcTrainPreds = svcModel.predict(x_train_scaled)
svcTrainAccuracy =accuracy_score(y_train, svcTrainPreds)
print("SVC train accuracy without hyperparameter tuning: ",round(svcTrainAccuracy,3))
```

    SVC train accuracy without hyperparameter tuning:  0.897

```python
svcTestPreds = svcModel.predict(x_test_scaled)
svcTestAccuracy =accuracy_score(y_test, svcTestPreds)
print("SVC test accuracy without hyperparameter tuning: ",round(svcTestAccuracy,3))
```

    SVC test accuracy without hyperparameter tuning:  0.89

I performed a gridsearch to identify the best parameters for the SVC model, and obtained an accuracy rate of over 97%.

```python
param_grid_SVC ={'C':[0.1,1,10,100],
            'kernel':['rbf'],
             'gamma': [0.001, 0.01, 0.1, 1, 10]}
gridSearchTimeStart = time.clock()
grid_search_SVC = GridSearchCV(svcClass, param_grid = param_grid_SVC, cv = 10)
grid_search_SVC.fit(x_train_scaled, y_train)
#timing how long it takes
gridSearchTimed = time.clock() - gridSearchTimeStart
hours = gridSearchTimed // 3600
gridSearchTimed %= 3600
minutes = gridSearchTimed // 60
gridSearchTimed %=60
seconds = gridSearchTimed
```

```python
print("Time to perform Grid Search : ", round(hours,0) , "hour(s),", round(minutes,0) , "minutes,", round(seconds,2), " seconds")
```

    Time to perform Grid Search :  1.0 hour(s), 12.0 minutes, 34.43  seconds

```python
#grid_search_SVC.best_estimator_
print("Best parameters: ", grid_search_SVC.best_params_, "\nBest Score: ", round(grid_search_SVC.best_score_,3))
grid_search_SVC.best_estimator_
```

    Best parameters:  {'C': 100, 'gamma': 1, 'kernel': 'rbf'}
    Best Score:  0.976





    SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
        probability=False, random_state=None, shrinking=True, tol=0.001,
        verbose=False)

```python
#grid_search_SVC.cv_results_
svcGrid = grid_search_SVC.best_estimator_
#finding the best mean score and fit time
mean_test_score = grid_search_SVC.cv_results_['mean_test_score'][grid_search_SVC.best_index_]
mean_fit_time = grid_search_SVC.cv_results_['mean_fit_time'][grid_search_SVC.best_index_]
print("Mean test score for best combination: ",round(mean_test_score,3))
print("Mean fit time for best combination: ", round(mean_fit_time,3), " seconds")
#Fitting SVC with selected parameters
svcModelGrid = svcGrid.fit(x_train_scaled, y_train)
#Testing accuracy
gridTestPreds_SVC = svcModelGrid.predict(x_test_scaled)
gridTestAccuracy_SVC = accuracy_score(y_test, gridTestPreds_SVC)
print("SVC test accuracy with hyperparameter tuning: ",round(gridTestAccuracy_SVC,3))

```

    Mean test score for best combination:  0.976
    Mean fit time for best combination:  24.918  seconds
    SVC test accuracy with hyperparameter tuning:  0.974

### K- Nearest Neighbors

The last model I fit to the data was the KNN model using the default scikit learn parameters. This instance based modeling method was favored by the reserachers, and was able to achieve 96% accuracy even without parameter tuning.

```python
kNN = KNeighborsClassifier() #default n_estimators = 10
knnModel = kNN.fit(x_train, y_train)
from pprint import pprint
print("Default K-NN parmeters in use:\n")
pprint(knnModel.get_params())
```

    Default K-NN parmeters in use:

    {'algorithm': 'auto',
     'leaf_size': 30,
     'metric': 'minkowski',
     'metric_params': None,
     'n_jobs': None,
     'n_neighbors': 5,
     'p': 2,
     'weights': 'uniform'}

```python
knnTestPreds = knnModel.predict(x_test)
knnTestAccuracy =accuracy_score(y_test, knnTestPreds)
print("KNN test accuracy without hyperparameter tuning: ",round(knnTestAccuracy,3))
```

    KNN test accuracy without hyperparameter tuning:  0.96

Hyperparameter tuning improved the accuracy to over 97%.

```python
param_grid_KNN = {'n_neighbors': np.arange(1, 25)}
gridSearchTimeStart = time.clock()
grid_search_KNN = GridSearchCV(knnModel, param_grid = param_grid_KNN, cv = 10)
grid_search_KNN.fit(x_train, y_train)
#timing how long it takes
gridSearchTimed = time.clock() - gridSearchTimeStart
hours = gridSearchTimed // 3600
gridSearchTimed %= 3600
minutes = gridSearchTimed // 60
gridSearchTimed %=60
seconds = gridSearchTimed
```

```python
print("Time to perform Grid Search : ", round(hours,0) , "hours,", round(minutes,0) , "minutes,", round(seconds,2), " seconds")
```

    Time to perform Grid Search :  0.0 hours, 3.0 minutes, 36.73  seconds

```python
#grid_search_KNN.best_estimator_
print("Best parameters and score:")
pprint(grid_search_KNN.best_params_)#'C': 100, 'kernel': 'rbf'
print(round(grid_search_KNN.best_score_, 3))#0.8194735838260537
```

    Best parameters and score:
    {'n_neighbors': 1}
    0.974

```python
knnGrid = grid_search_KNN.best_estimator_
#Fitting KNN with selected parameters
knnModelGrid = knnGrid.fit(x_train, y_train)
#finding the best mean score and fit time
mean_test_score = grid_search_KNN.cv_results_['mean_test_score'][grid_search_KNN.best_index_]
mean_fit_time = grid_search_KNN.cv_results_['mean_fit_time'][grid_search_KNN.best_index_]
print("Mean test score for best combination: ",round(mean_test_score,3))
print("Mean fit time for best combination: ", round(mean_fit_time,3))
#Testing accuracy
gridTestPreds_KNN = knnModelGrid.predict(x_test)
gridTestAccuracy_KNN = accuracy_score(y_test, gridTestPreds_KNN)
print("KNN test accuracy with hyperparameter tuning: ",round(gridTestAccuracy_KNN,3))
```

    Mean test score for best combination:  0.974
    Mean fit time for best combination:  0.058
    KNN test accuracy with hyperparameter tuning:  0.971

After training and tuning all three models, I proceeded to compare them.

<header> 
    <h3> <u>Comparing Model Performance</u> </h3>
</header>

I compared the model performances using confusion matrices, precision, accuracy and fit time.

#### Accuracy vs Precision

The original research authers selected the instance based model because of its high accuracy. I chose to also evaluate the models based on accuracy, precision and speed. In this case, Accuracy represents the ability of the model to detect when the eye state is open, and precision is a measure of the ability of the model to correctly detect when the eye state is closed. Different application may warrant different classification performances and thresholds for both measures. For example, in Autonomous Vehicles, it would be more important to have a detection system that can identify when the driver's eyes are closed and therfore want higher model precision, whereas mobile gaming systems would need to detect when the user's eyes are open and need higher model accuracy.

#### Fit Time

Speed is another important factor for consideration. Again, in a case where this technology is implemented in an autonomous vehicle, it would be critically important to detect when the user's eye state is closed expediently.

```python
modelGridPreds = {'RF': gridTestPreds_RF, 'KNN': gridTestPreds_KNN, 'SVC' : gridTestPreds_SVC}
cm = {}
cm_per = {}

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 6))
for i in modelGridPreds:
    cm[i] = confusion_matrix(y_test, modelGridPreds[i])
    cm_per[i] = cm[i].astype('float') / cm[i].sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_per[i], annot = True, annot_kws = {"size":20},cmap = 'Blues',
                square = True, fmt = '.3f', ax = ax[list(modelGridPreds.keys()).index(i)])

    ax[list(modelGridPreds.keys()).index(i)].set_title("Confusion Matrix for " + str(i) + "\n", fontsize=15,  fontweight='black')
    ax[list(modelGridPreds.keys()).index(i)].set_ylabel('True Label', fontsize = 15)
    ax[list(modelGridPreds.keys()).index(i)].set_xlabel('Predicted Label', fontsize = 15)
    ax[list(modelGridPreds.keys()).index(i)].set_ylim(2,0)
plt.show()
```

![png](images/eeg_images/output_56_0.png)

```python
data = [[gridTestAccuracy_RF, [cm_per['RF'][1][1]]],
         [gridTestAccuracy_KNN, cm_per['KNN'][1][1]],
         [gridTestAccuracy_SVC],cm_per['SVC'][1][1]]

X = np.arange(2)
fig = plt.figure(figsize = (12, 6))
ax = fig.add_axes([0, 0, 1, 1])
plt.xticks([0.2,1.2], ['Accuracy', 'Precision'], fontsize = 15)
ax.bar(X + 0.00, data[0], color = 'navy', width = 0.2)#, alpha=0.6)
ax.bar(X + 0.2, data[1], color = 'green', width = 0.2)#, alpha=0.6)
ax.bar(X + 0.40, data[2], color = 'purple', width = 0.2)#, alpha=0.6)
ax.legend(labels=['RF', 'KNN', 'SVC'], loc = 'upper right', fontsize = 12)
ax.set_ylim(0.88, 1)
ax.yaxis.grid(linestyle = '--')
ax.set_axisbelow(True)
ax.set_title("Comparing Model Accuracies and Precision", fontsize=15, fontweight='black')
plt.show()

#Plotting Time to Fit models
models = ['KNN', 'RF', 'SVC' ]
modeltimefit = [grid_search_KNN.cv_results_['mean_fit_time'][grid_search_KNN.best_index_], grid_search_RF.cv_results_['mean_fit_time'][grid_search_RF.best_index_], grid_search_SVC.cv_results_['mean_fit_time'][grid_search_SVC.best_index_]]
my_range=list(range(1,len(models)+1))
fig, ax = plt.subplots(figsize=(10,2))
ax.tick_params(axis='both', which='major', labelsize=13)
plt.yticks(my_range, models)
plt.hlines(y=my_range, xmin=0, xmax=modeltimefit, color=['green', 'navy', 'purple'], alpha=0.4, linewidth=8)
plt.ylim([0.5,3.5])
plt.xlim([-0.2,27])
plt.plot(modeltimefit, my_range, "o", markersize=7, color='black', alpha=0.8)
plt.title('Comparing Model Test Data Fit Time', fontsize=12, fontweight='black')
ax.grid(which = 'both', linestyle = '--')
ax.grid(which = 'minor', alpha = 0.5)
ax.grid(which = 'major', alpha = 0.2)
ax.set_xlabel('Time(s)', fontsize = 15)
fig.text(-0.23, 0.96, ' ', fontsize=6, fontweight='black', color = '#333F4B')
for i, v, in enumerate(modeltimefit):
    ax.text(v + 0.25, i+0.9, str(round(v, 2)),
           color = 'black', fontweight = 'bold')
plt.show()
```

![png](images/eeg_images/output_57_0.png)

![png](images/eeg_images/output_57_1.png)

Based on accuracy and precision, the Support Vector Classifier slightly edges out the KNN Classifier with the best model performance. In applications where speed is not a critical issue, this classifier is the best. However, in applications where speed is critical factor, the KNN classifier outperforms both the Random Forest and Support Vector Machine, since it is a 'lazy' learner that fit the test the most expediently out of all three with a fit time of 0.05s.

<header> 
    <h3> <u>Feature Selection</u> </h3>
</header>

One of the items that the authors of the research paper cited for potential future work is ascertaining whether or not all 14 EEG sensors were needed to maintain high model accuracy. I explored this by performing recursive feature elimination with cross valiadation (RFECV) using the scikit learn library using my Random Forest model. I used this model because the SVC with RBF kernel and K-NN Classifiers do not leverage feature importance to when measuring similarity.

I first determined the feature importance to get a sense of their relative importance. The sensors do not appear to be equal in their predictive power in the model. The X7 sensor, for example, accounts for approximately three times as much variance in the data as the X9 sensor.

```python
#Retained Variance, Number of Components, Time to fit, Accuracy

sortedFeatureIndices = -np.argsort(randForGrid.feature_importances_)
orderedFeatures = list()
for i in sortedFeatureIndices:
    orderedFeatures.append(x_data.columns.values[i])

# Plot the name and gini importance of each feature
#for feature in zip(orderedFeatures, sorted(randModel.feature_importances_, reverse = True)):
#    print(feature)

y_pos = np.arange(len(orderedFeatures))
gini_index = sorted(randModel.feature_importances_, reverse = True)
plt.bar(y_pos, gini_index, align = 'center')
plt.xticks(y_pos, orderedFeatures)
plt.ylim(0.03, 0.14)
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.title('Feature Importance')
plt.show()
```

![png](images/eeg_images/output_60_0.png)

```python
selRFECVTimeStart = time.clock()
selRFECV_RF = RFECV(randForGrid, step=1, cv=10, scoring='accuracy')

selRF = selRFECV_RF.fit(x_train, y_train)
selRF.ranking_
```

    array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1])

```python
selRF.transform(x_train)
rfecvTimed = time.clock() - selRFECVTimeStart
hours = rfecvTimed // 3600
rfecvTimed %= 3600
minutes = rfecvTimed // 60
rfecvTimed %=60
seconds = rfecvTimed
selRF.n_features_
```

    13

The recursive feature elimination algorithm ranked all sensors as being equally important to the predictive accuracy of the random forest model, except the sensor with the lowest feature importance ranking, sensor X9. The cross validation acuracy score of the model is directly proportional to the number of sensors, however removing sensor X9 would not cause a decline in predictive accuracy.

```python
plt.figure()
plt.title('Random Forest CV score vs Number of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation accuracy score")
plt.plot(range(1, len(selRF.grid_scores_) + 1), selRF.grid_scores_)
plt.show()
```

![png](images/eeg_images/output_64_0.png)

```python
print("Time to perform RFE w/ CV : ", round(hours,0) , "hours,", round(minutes,0) , "minutes,", round(seconds,2), " seconds")
```

    Time to perform RFE w/ CV :  0.0 hours, 18.0 minutes, 11.53  seconds

```python
#Testing the accuracies
selRFpred = selRF.predict(x_test)
selTestAccuracy_RF = accuracy_score(y_test, selRFpred)
print("Random forest selected features test accuracy: ",round(selTestAccuracy_RF,3))
```

    Random forest selected features test accuracy:  0.933

The design of the headset is not condudive to this kind of design that excludes sensor X9. This suggests that other methods (different type of sensor, more accurate sensors, etc) would have to be tested to simplify the experiments in the future. In the interim, the dimmentionality of the data can be reduced using methods such as Principal Component Analysis.

#### Principal Component Analysis

An added benefit of dimmensionality redution using Principal Componet Analysis(PCA) is that it can also speed up the training time and fit time for Machine Learning algorithms. I used the [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) module in scikit-learn for this step. Since PCA is affected by scale, I applied PCA using the same scaled feature set that I used for the SVC model.

```python
variance_test_list = [0.99, 0.95, 0.9, 0.85, 0.8]
no_components = list()
fit_time = list()
model_accuracy = list()

for variance in variance_test_list:
    pca = PCA(variance)
    pca.fit(x_train_scaled)
    x_train_pca = pca.transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    #Start time
    svcPCATimeStart = time.clock()
    svcModelPCA = svcGrid.fit(x_train_pca, y_train)
    svcPCATimed = time.clock() - svcPCATimeStart
    #Testing accuracy
    pcaTestPreds_SVC = svcModelPCA.predict(x_test_pca)
    pcaTestAccuracy = accuracy_score(y_test, pcaTestPreds_SVC)
    #Updating lists
    fit_time.append(round(svcPCATimed,2))
    no_components.append(pca.n_components_)
    model_accuracy.append(round(pcaTestAccuracy,3))
```

I applied PCA to the SVC model since it had the longest training time out of the three models and could benefit the most from dimmensionality reduction. I varied the percent of retained variance to observe how the model accuracy and fit time changed. The results are tabulated below.

The result show that this model does not benefit from PCA.

```python
pca_report = pd.DataFrame(list(zip(variance_test_list, no_components, model_accuracy, fit_time)), columns = ['Retained Variance', 'No. Components', 'Accuracy', 'Fit Time (s)'])
pca_report
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
      <th>Retained Variance</th>
      <th>No. Components</th>
      <th>Accuracy</th>
      <th>Fit Time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.99</td>
      <td>12</td>
      <td>0.964</td>
      <td>19.01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.95</td>
      <td>8</td>
      <td>0.896</td>
      <td>32.99</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.90</td>
      <td>5</td>
      <td>0.771</td>
      <td>75.48</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.85</td>
      <td>3</td>
      <td>0.680</td>
      <td>93.68</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.80</td>
      <td>3</td>
      <td>0.680</td>
      <td>85.38</td>
    </tr>
  </tbody>
</table>
</div>

<header> 
    <h3> <u>Temporal Analysis</u> </h3>
</header>

Time Series analysis mandates that the temporal integrity of the instances is maintained. When performing the splits for training and testing, I set the 'shuffle' parameter to False to maintain the temporal order and use the first 70% of observations from the experiment for training, and the last 30% of observations from the experiment for testing.

The research authors indicated their preference for the instance based learning model (KStar) and a desire to track eye state in real-time. Real-time tracking would require expedient run-time behaviour of the predictive model. For this reason, I use the KNN Classifier that I obtained from my GridSearch to test temporal dependence since it is a 'lazy' that fit the data the most expediently with high accuracy. Out of the three models that I tested, it seemed to be the best suited for a degree of real-world application.

```python
# split the dataset
random_state = 100

x_data = data_sans_outliers.iloc[:,0:14]
y_data = data_sans_outliers.iloc[:, 14]

x_temp_train, x_temp_test, y_temp_train, y_temp_test = train_test_split(x_data, y_data, train_size = 0.7, random_state = random_state, shuffle = False)
```

I looped through the test set, and used the previous 10 instances in the time series to make the next prediction. I stored the predictions so that I could plot the predicted eye states against the actual eye states to better understand the model performance, and obtain an accuracy score.

```python
#xPrevious, yPrevious = [x for x in x_temp_train], [x for x in y_temp_train]
xPrevious = x_temp_train.copy()
yPrevious = y_temp_train.copy()
preds = []
for i in range(len(y_temp_test)):
    # define model
    tempKNN = knnGrid
    # fit model on a small subset of the train set
    tmpy = yPrevious.iloc[-10:]
    tmpX = xPrevious.iloc[-10:]

    tempKNN.fit(tmpX, tmpy)
    # forecast the next time step
    y_pred = tempKNN.predict(x_temp_test.iloc[i:i+1, :])
    # store prediction
    preds.append(float(y_pred))
    # add real observation to history
    xPrevious = xPrevious.append(x_temp_test.iloc[i:i+1, :])
    yPrevious = yPrevious.append(y_temp_test.iloc[i:i+1])

# evaluate predictions
predictions = pd.Series(preds, index = y_temp_test.index)
temp_accuracy = accuracy_score(y_temp_test, predictions)
print("The accuracy of the KNN model with temporal dependency is: ", round(temp_accuracy,3))
```

    The accuracy of the KNN model with temporal dependency is:  0.996

The new accuracy for this model is over 99%, which appears to be excellent, however further observation of the trace of the observed and predicted Eye States shows that the model assumes that the current state is most likely the past state. The predicted eye state trace (red) lags behind the observed eye state trace(green). This could be due to the fact that the frequency of open and closing the eye is very slow. For future work, in order to build a better model that fits the data quickly and can predict the Eye State in real time, an experiment that involves a higher frequency of eye state changes is needed.

```python
#y_temp_test = y_temp_test.reset_index(drop=True)
fig, ax = plt.subplots(figsize = (18,5))
ax.set_title('Temporal trace of predicted and observed Eye State')
plt.plot(predictions, marker = '', color = 'red', linewidth = 1)
plt.plot(y_temp_test, marker = '', color = 'green', linewidth = 1)

plt.show()
```

![png](images/eeg_images/output_77_0.png)

<header> 
    <h3> <u>Concluding remarks</u> </h3>
</header>
