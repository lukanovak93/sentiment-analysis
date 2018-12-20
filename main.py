import argparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from util import *


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', choices=['logistic', 'svm'], required=True, help='Specify which model to use')
    ap.add_argument('-v','--verbose', required=False, action='store_true',
        help='Verbose output')
    args = vars(ap.parse_args())

    reviews_train, reviews_test = load_files()
    if args['verbose']:
        print('\nRaw text:\n\n', reviews_train[0])

    reviews_train_clean = preprocess_reviews(reviews_train)
    reviews_test_clean = preprocess_reviews(reviews_test)
    if args['verbose']:
        print('\nProcessed text:\n\n', reviews_train_clean[0])

    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    if args['verbose']:
        print('\nX_train shape: {}, X_test.shape: {}'.format(X.shape[0], X_test.shape[0]))

    target = [1 if i < 12500 else 0 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

    if args['verbose']:
        print('\nX_train: {0}, y_train: {2}, X_val: {1}, y_val: {3}'.format(X_train.shape[0], X_val.shape[0], len(y_train), len(y_val)))

    if args['model'] == 'logistic':
        best, scores = get_best_param_logistic(X_train, y_train, X_val, y_val)

        if args['verbose']:
            for c, s in scores:
                print("\nAccuracy for C={}: {}".format(c, s))

        best_param, best_score = best

        if args['verbose']:
            print('\nThe best accuracy of {} is for parameter c = {}'.format(best_score, best_param))

    if args['model'] == 'svm':
        best, scores = get_best_param_svm(X_train, y_train, X_val, y_val)

        if args['verbose']:
            for c, s in scores:
                print("\nAccuracy for C={}, kernel={}: {}".format(c[0], c[1], s))

        best_param, best_score = best

        if args['verbose']:
            print('\nThe best accuracy of {} is for parameter c = {} and kernel = {}'.format(best_score, best_param[0], best_param[1]))


    print('\nTraining the final model...\n')
    if args['model'] == 'logistic':
        model = train_logistic_reg(X, target, X_test, best_param)
    if args['model'] == 'svm':
        model = train_svm(X, target, X_test, C=best_param[0], kernel=best_param[1])

    if args['model'] == 'logistic':
        feature_to_coef = get_feat_names(cv, model)

        if args['verbose']:
            print('\nBest positive words:')
            for best_positive in sorted(feature_to_coef.items(), reverse=True, key=lambda x: x[1])[:5]:
                print (best_positive)

            print('\nBest negative words:')
            for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1])[:5]:
                print (best_negative)
