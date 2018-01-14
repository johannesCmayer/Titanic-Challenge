import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


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

# ToDo make it that the returned dataframe has the correct collum labels insted of ints
def transform_data(data, num_atribs, cat_atribs):
    # Feature Engenering
    pd.set_option('chained_assignment', None)
    data.loc[:,("Fare_Pclass")] = data["Fare"] / (np.power(data["Pclass"], 2))
    pd.set_option('chained_assignment', 'warn')

    # Data Transformation
    cat_data = DataFrameSelector(cat_atribs).fit_transform(data)
    one_hot_cat_data = pd.get_dummies(pd.DataFrame(cat_data), prefix='', prefix_sep='')

    saved_passengerId = data['PassengerId']
    data = data.drop(['PassengerId'], axis=1)

    #print(saved_passengerId.isnull().sum())

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_atribs)),
        ('imputer', Imputer(strategy="median")),
        #('std_scaler', StandardScaler()),
    ])

    transformed_data = pd.concat([pd.DataFrame(num_pipeline.fit_transform(data), columns=num_atribs), one_hot_cat_data], axis=1)

    transformed_data.loc[:,('PassengerId')] = np.array(saved_passengerId.values)

    print(transformed_data)

    #print(transformed_data['PassengerId'].isnull().sum(), 'in trans data')

    return pd.DataFrame(transformed_data)


def run_model(train_data, train_data_lables, prediction_data):
    # Crossvalidation of modelestClassifier()
    # model = RandomForestClassifier()

    model = MLPClassifier(hidden_layer_sizes=(100,5), max_iter=500, early_stopping=False, learning_rate_init=0.0001)

    model = LinearSVC()
    model = NuSVC()
    model = SVC(C=2.0)


    scores = cross_val_score(model, train_data, train_data_lables, cv=10)
    print('cross validation scores are {} average: {}'.format(scores, sum(scores) / len(scores)))

    # Test performance on Train data to detect potential overfitting
    model.fit(train_data, train_data_lables)
    pred_on_train_data = model.predict(train_data) == train_data_lables
    print('When model is run against training data {} rigth out of {} | {}%\n'.format(
        len([p for p in pred_on_train_data if p == True]), len(pred_on_train_data), len([p for p in pred_on_train_data if p == True]) / len(pred_on_train_data) * 100))

    return pred_on_train_data


def run():
    data = pd.read_csv('train.csv')
    submission_data = pd.read_csv('test.csv')
    train_data, test_data = split_data(data, test_size=0.2)

    num_atribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_atribs = ['Sex', 'Embarked']

    transformed_train_data = transform_data(train_data, num_atribs=num_atribs, cat_atribs=cat_atribs)
    transformed_test_data = transform_data(test_data, num_atribs=num_atribs, cat_atribs=cat_atribs)
    transformed_submission_data = transform_data(submission_data, num_atribs=num_atribs, cat_atribs=cat_atribs)

    #Stransformed_train_data.info()

    # transformed_train_data.drop(['PassengerId'])
    # transformed_submission_data.drop(['PassengerId'])
    # transformed_train_data.drop(['PassengerId'])

    train_data_labels = train_data['Survived']
    test_data_labels = test_data['Survived']

    prediction = run_model(transformed_train_data, train_data_labels, test_data)

    submission_prediction = run_model(transformed_train_data, train_data_labels, transformed_submission_data)

    submission_prediction = pd.concat([pd.DataFrame(submission_prediction, index=None), submission_data['PassengerId']], axis=1)

    # TODO make it that it has the correct submission format
    pd.DataFrame(submission_prediction).to_csv('submission_data')


if __name__ == '__main__':
    run()