import re

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import spacy
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES


def load_files():
    reviews_train = []
    for line in open('data/full_train.txt', 'r', encoding='utf-8'):
        reviews_train.append(line.strip())

    reviews_test = []
    for line in open('data/full_test.txt', 'r', encoding='utf-8'):
        reviews_test.append(line.strip())

    return (reviews_train, reviews_test)


def preprocess_reviews_simple(reviews):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews


def remove_stopwords(text, nlp):
    nlp_doc = nlp(text)
    s = [token.text for token in nlp_doc if not token.is_stop]

    return ' '.join(s)


def preprocess_reviews_standard(reviews):
    simple = preprocess_reviews_simple(reviews)

    nlp = spacy.load('en_core_web_lg')

    standard = []
    for doc in simple:
        standard.append(remove_stopwords(doc, nlp))

    return standard


def lemmatize(text, nlp):
    lemmatized_line = [token.lemma_ for token in nlp(text)]
    lemmatized_line = list(filter(lambda a: a != '-PRON-', lemmatized_line))

    return ' '.join(lemmatized_line)


def preprocess_reviews_advanced(reviews):
    standard = preprocess_reviews_standard(reviews)

    nlp = spacy.load('en_core_web_lg')

    advanced = []
    for doc in standard:
        advanced.append(lemmatize(doc, nlp))

    return advanced


def get_best_param_logistic(X_train, y_train, X_val, y_val):

    scores = []

    for c in [0.01, 0.05, 0.25, 0.5, 1]:

        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        scores.append((c, accuracy_score(y_val, lr.predict(X_val))))

    sorted_scores = sorted(scores, reverse=True, key=lambda a: a[1])
    return (sorted_scores[0], scores)


def train_logistic_reg(X, target, X_test, c):
    final_model = LogisticRegression(C=c)
    final_model.fit(X, target)
    print ("Final Accuracy: {}".format(accuracy_score(target, final_model.predict(X_test))))

    return final_model


def get_best_param_svm(X_train, y_train, X_val, y_val):

    scores = []

    for c in [0.05, 0.25, 0.5, 1]:
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

            lr = SVC(C=c, kernel=kernel, gamma='scale')
            lr.fit(X_train, y_train)
            scores.append(((c, kernel), accuracy_score(y_val, lr.predict(X_val))))

    sorted_scores = sorted(scores, reverse=True, key=lambda a: a[1])
    return (sorted_scores[0], scores)


def train_svm(X, target, X_test, C, kernel):
    final_model = SVC(C=C, kernel=kernel, gamma='scale')
    final_model.fit(X, target)
    print ("Final Accuracy: {}".format(accuracy_score(target, final_model.predict(X_test))))

    return final_model


def get_feat_names(cv, final_model):
    feature_to_coef = {
        word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])
    }

    return feature_to_coef
