import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.feature_selection import f_classif

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_data(path: str):
    data = pd.read_csv(path, index_col=0)
    data.drop('player', axis=1, inplace=True)
    data.drop('rating', axis=1, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = pd.get_dummies(data, prefix=['map', 'agent'], columns=['map', 'agent'])
    X = data.drop('won', axis=1)
    y = data['won']
    features = X.columns
    X, y = X.values.astype(float), y
    return X, y, features


def evaluate(model, X, y, name='model', set_name='dev'):
    print(f'{name}:')
    print(f'  {set_name} accuracy:', accuracy_score(y, model.predict(X)))
    print(f'  {set_name} f1:', f1_score(y, model.predict(X)))
    print(f'  {set_name} auroc:', roc_auc_score(y, model.predict(X)))


if __name__ == '__main__':
    # Load data
    X, y, features = load_data('data.csv')

    # Split into train/dev/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.2)
    print(f'train: {X_train.shape}, {y_train.shape}')
    print(f'dev: {X_dev.shape}, {y_dev.shape}')
    print(f'test: {X_test.shape}, {y_test.shape}')

    # Scale data to zero-mean, unit-variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # Test out suite of foundational machine learning models

    logreg = LogisticRegression()
    logreg.fit(X_train_scaled, y_train)
    evaluate(logreg, X_dev_scaled, y_dev, name='logistic regression')

    svm = SVC()
    svm.fit(X_train_scaled, y_train)
    evaluate(svm, X_dev_scaled, y_dev, name='support vector machine')

    adaboost = AdaBoostClassifier(algorithm='SAMME')
    adaboost.fit(X_train_scaled, y_train)
    evaluate(adaboost, X_dev_scaled, y_dev, name='adaboost ensemble')

    # Compute ANOVA F-statistics for features

    f_statistics, p_values = f_classif(np.r_[X_train, X_dev, X_test], np.r_[y_train, y_dev, y_test])
    for (feature, f_statistic, p_value) in zip(features, f_statistics, p_values):
        print(feature, '\n\tf:', f_statistic, '\n\tp:', p_value)

    # takeaways:
    # - maps have no indication towards winning or losing, since VALORANT is a zero-sum game
    #   -> on each map, there are equal winners/losers
    # - kills have high impact on the outcome of the match, but deaths are more important
    #   -> staying alive is more valuable than getting kills in many (but not all) cases
    # - KAST is the most important feature
    #   -> this is a metric for smart decision-making in a game, so is a good predictor of match outcome
