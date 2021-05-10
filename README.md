# Restaurant Reviews - UTS

## Dataset
Thanks to [Harshit Joshi on Kaggle](https://www.kaggle.com/hj5992/restaurantreviews)

## Requirements
- Check [requirements.txt](https://github.com/jacenyang/restaurant-reviews-uts/blob/master/requirements.txt) and make sure those packages are installed.
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
