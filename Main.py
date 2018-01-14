import subprocess as sp
import pandas as pd
import numpy as np
import os

from sklearn.externals import joblib
from random import uniform

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, RandomizedSearchCV
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def project_setup(saved_models_path='data/', saved_predictions_path='predictions/'):
    global paths
    paths = {
        'saved_models_path': saved_models_path,
        'saved_predictions_path': saved_predictions_path,
    }

    for _, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)


def split_data(data, test_size=0.2):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_index, test_index in split.split(data, data["Pclass"]):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
    return train_data, test_data


class DataFrameSelector(TransformerMixin, BaseEstimator):
    def __init__(self, atribs):
        self.atribs = atribs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.atribs].values


def transform_data(data, num_atribs, cat_atribs, no_fit=False):
    # Feature Engenering
    pd.set_option('chained_assignment', None)
    data.loc[:, ("Fare_Pclass")] = data["Fare"] / (np.power(data["Pclass"], 2))
    pd.set_option('chained_assignment', 'warn')

    # Data Transformation
    cat_data = DataFrameSelector(cat_atribs).fit_transform(data)
    one_hot_cat_data = pd.get_dummies(pd.DataFrame(cat_data), prefix='', prefix_sep='')

    saved_passengerId = data['PassengerId']
    data = data.drop(['PassengerId'], axis=1)

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_atribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    if no_fit:
        num_pipeline = joblib.load('{}num_pipe'.format(paths.get('saved_models_path')))
    else:
        num_pipeline.fit(data)
        joblib.dump(num_pipeline, '{}num_pipe'.format(paths.get('saved_models_path')))

    transformed_data = pd.concat([pd.DataFrame(num_pipeline.transform(data), columns=num_atribs), one_hot_cat_data],
                                 axis=1)

    transformed_data.loc[:, ('PassengerId')] = saved_passengerId.values

    return pd.DataFrame(transformed_data)

def drop_passenger_iD(X):
    saved_passengerId = X['PassengerId']
    X = X.drop(['PassengerId'], axis=1)
    return X, saved_passengerId

def run_model(model, train_data, train_data_lables, prediction_data, no_fit=False):
    print('Model: {}'.format(type(model).__name__))

    train_data, saved_pasid_train = drop_passenger_iD(train_data)
    prediction_data, saved_pasid_pred = drop_passenger_iD(prediction_data)

    # Crossvalidation of model Classifier
    scores = cross_val_score(model, train_data, train_data_lables, cv=4)
    print('Cross-validation score average: {}%'.format(truncate(sum(scores) / len(scores) * 100, 2)))

    # Make predictions
    if no_fit:
        model = joblib.load('{}Trained_{}_Data'.format(paths.get('saved_models_path'), type(model).__name__))
    else:
        model.fit(train_data, train_data_lables)
        joblib.dump(model, '{}Trained_{}_Data'.format(paths.get('saved_models_path'), type(model).__name__))

    pred_on_train_data = model.predict(train_data) == train_data_lables
    print('Acuracy on training data {}%\n'.format(
        truncate(len([p for p in pred_on_train_data if p == True]) / len(pred_on_train_data) * 100, 2)))

    df = pd.DataFrame()
    df['PassengerId'] = saved_pasid_pred
    df['Survived'] = model.predict(prediction_data)

    return df


def run():
    data = pd.read_csv('train.csv')
    submission_data = pd.read_csv('test.csv')
    train_data, test_data = split_data(data, test_size=0.01)

    num_atribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_atribs = ['Sex', 'Embarked']

    transformed_train_data = transform_data(train_data, num_atribs=num_atribs, cat_atribs=cat_atribs)
    transformed_test_data = transform_data(test_data, num_atribs=num_atribs, cat_atribs=cat_atribs)
    transformed_submission_data = transform_data(submission_data, num_atribs=num_atribs, cat_atribs=cat_atribs,
                                                 no_fit=True)

    train_data_labels = train_data['Survived']
    test_data_labels = test_data['Survived']

    models = {
        0: RandomForestClassifier(max_depth=None, max_leaf_nodes=None, warm_start=True),
        1: LinearSVC(),
        2: NuSVC(),
        3: SVC(C=1.0),
        4: DecisionTreeClassifier(),
        5: ExtraTreeClassifier(),
        6: GaussianNB(),
        7: KNeighborsClassifier(),
        8: MLPClassifier(max_iter=1000),
        9: AdaBoostClassifier(),
    }
    model_num_single_model = 3

    param_dist = {
        'C': list(range(1, 15)),
        'kernel': ['rbf', 'sigmoid'], #['rbf'], 'poly', 'linear', 'sigmoid', 'precomputed'],
        'degree': [3],
        'gamma': ['auto'],
        'coef0': [0.0],
        'shrinking': [True],
        'probability': [False],
        'tol': [1e-3],
        'cache_size': list(range(1, 2000)),
        'class_weight': [None],
        'verbose': [False],
        'max_iter': [-1],
        'decision_function_shape': ['ovr'],
        'random_state': [42]
    }

    transformed_train_data, saved_pasid_train = drop_passenger_iD(transformed_train_data)
    transformed_submission_data, saved_pasid_pred = drop_passenger_iD(transformed_submission_data)

    rand_search = RandomizedSearchCV(models[model_num_single_model], n_iter=10000, param_distributions=param_dist)
    rand_search.fit(transformed_train_data, train_data_labels)
    print(rand_search.best_estimator_)
    submission_prediction = rand_search.predict(transformed_submission_data)

    df = pd.DataFrame()
    df['PassengerId'] = saved_pasid_pred
    df['Survived'] = submission_prediction

    df.to_csv(
        '{}submission_data_randcv_SVC.csv'.format(paths.get('saved_predictions_path')), index=False)

    # if model_num_single_model is None:
    #     for _, model in models.items():
    #         # prediction_on_train_split = run_model(model, transformed_train_data, train_data_labels, transformed_test_data)
    #         submission_prediction = run_model(model, transformed_train_data, train_data_labels,
    #                                           transformed_submission_data)
    #         submission_prediction.to_csv('{}submission_data_{}.csv'.format(paths.get('saved_predictions_path'), type(model).__name__), index=False)
    # else:
    #     model = models[model_num_single_model]
    #     submission_prediction = run_model(model, transformed_train_data, train_data_labels,
    #                                       transformed_submission_data)
    #     submission_prediction.to_csv('{}submission_data_{}.csv'.format(paths.get('saved_predictions_path'), type(model).__name__), index=False)


# Utilities
def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


if __name__ == '__main__':
    print('\n')
    project_setup()
    run()
