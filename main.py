# Importing the library
import pandas as pd
import tkinter as tk
from tkinter import filedialog


# Importing CSV function
def import_csv():
    global dataset
    import_file_path = filedialog.askopenfilename()
    dataset = pd.read_csv(import_file_path, delimiter='\t', quoting=3)


# Training and testing imported dataset
def train_and_test():

    # Preprocessing the dataset
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    # Creating the Bag of Words model using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    test_size_input = int(test_size_entry.get().strip('%'))/100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_input, random_state=42)

    if radio_state.get() == 1:

        # Multinomial Naive Bayes
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(X_train, y_train)

    elif radio_state.get() == 2:

        # Bernoulli Naive Bayes
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB(alpha=0.8)
        classifier.fit(X_train, y_train)

    elif radio_state.get() == 3:

        # Logistic Regression
        # Fitting Logistic Regression to the Training set
        from sklearn import linear_model
        classifier = linear_model.LogisticRegression(C=1.5)
        classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    # Accuracy, Precision, Recall and F1
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    score1 = accuracy_score(y_test, y_pred)
    score2 = precision_score(y_test, y_pred)
    score3 = recall_score(y_test, y_pred)
    score4 = f1_score(y_test, y_pred)
    as_value = round(score1 * 100, 2)
    ps_value = round(score2, 2)
    rs_value = round(score3, 2)
    f1s_value = round(score4, 2)

    label.config(text=f"Confusion Matrix:\n{cm}\nAccuracy score: {as_value}%\nPrecision score: {ps_value}\nRecall score: {rs_value}\nF1 score: {f1s_value}")


# Setting up GUI
window = tk.Tk()
window.title("Review Predictor")
window.config(padx=20, pady=20)
window.resizable(width=False, height=False)

# Import button
import_button = tk.Button(text="Import CSV File", width=18, command=import_csv)
import_button.pack()

# Test size entry
test_size_entry = tk.Entry(width=20, justify="center")
test_size_entry.insert(0, "Enter test size (%)")
test_size_entry.pack()

# Algorithm radio button
radio_state = tk.IntVar()
radiobutton1 = tk.Radiobutton(text="Multinomial Naive Bayes", value=1, variable=radio_state)
radiobutton2 = tk.Radiobutton(text="Bernoulli Naive Bayes", value=2, variable=radio_state)
radiobutton3 = tk.Radiobutton(text="Logistic Regression", value=3, variable=radio_state)
radiobutton1.pack()
radiobutton2.pack()
radiobutton3.pack()

# Train and test button
train_and_test_button = tk.Button(text="Train and Test", width=18, command=train_and_test)
train_and_test_button.pack()

# Result label
label = tk.Label()
label.config(pady=10)
label.pack()

window.mainloop()
