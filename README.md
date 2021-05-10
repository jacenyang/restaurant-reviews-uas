# Restaurant Reviews - UAS

## Introduction
The purpose of this project is to build a prediction model to predict whether a review on a restaurant is positive or negative. To do so, we will work on Restaurant Review dataset, we will load it into predicitve algorithms Multinomial Naive Bayes, Bernoulli Naive Bayes and Logistic Regression.

## Explanation
To build a model to predict whether a review is positive or negative, following steps are performed.
- Importing Dataset
- Preprocessing Dataset
- Vectorization
- Training and Classification
- Analysis Conclusion

## How to Use
- Import a CSV file
- Enter test size in percent (e.g. 30 or 30%)
- Choose an algorithm to train and test the dataset
- Click Train and Test

## Dataset
Thanks to [Harshit Joshi on Kaggle](https://www.kaggle.com/hj5992/restaurantreviews)

## Requirements
- Check [requirements.txt](https://github.com/jacenyang/restaurant-reviews-uas/blob/master/requirements.txt) and make sure those packages are installed.
- Download stopwords from ntlk by running this script.
```sh
import nltk
nltk.download('stopwords')
```

## Algorithms
- Multinomial Naive Bayes
```sh
Accuracy score is 77.67%
Precision score is 0.78
Recall score is 0.77
F1 score is 0.77
```
- Bernoulli Naive Bayes
```sh
Accuracy score is 77.0%
Precision score is 0.76
Recall score is 0.78
F1 score is 0.77
```
- Logistic Regression
```sh
Accuracy score is 76.67%
Precision score is 0.8
Recall score is 0.71
F1 score is 0.75
```

## Screenshot
![screenshot](screenshot.png)
