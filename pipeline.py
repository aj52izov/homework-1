import pdb
import pandas as pd
import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.base import BaseEstimator
import preprocessing as prepro

class Multi_classes_classifier_on_column(BaseEstimator):
    def __init__(self, base_classifier, column):
        self.column = column
        self.classifier = ClassifierChain(base_classifier)
        self.vectorizer = None

    def _get_vectors(self, X):
        text_data = X[self.column]
        text_data = [prepro.clean_text(text) for text in text_data]  # text cleaning
        feature_vector = self.vectorizer.transform(text_data).toarray()
        return feature_vector

    def fit(self, X, y):
        if type(self.column) == type(int(1)):
            self.column= list(X.columns)[self.column]
        if type(self.vectorizer) == type(None):
            self.vectorizer = prepro.get_text_vectorizer(X, self.column)

        feature_vector = self._get_vectors(X)
        self.classifier.fit(feature_vector, y)
        return self

    def predict(self, X):
        feature_vector = self._get_vectors(X)
        result = self.classifier.predict(feature_vector)
        return result

    def predict_proba(self, X:pd.DataFrame):
        feature_vector = self._get_vectors(X)
        result = self.classifier.predict_proba(feature_vector)
        return result

    def partial_fit(self,X, y):
        feature_vector = self._get_vectors(X)

        result = self.classifier.partial_fit(feature_vector,y)
        return result

    def score(self,X, y):
        feature_vector = self._get_vectors(X)
        result = self.classifier.score(feature_vector,y)
        return result

    def set_params(self, **params):
        self.classifier.set_params(**params)
        return self

    def get_params(self, deep):
        result = self.classifier.get_params(deep)
        return result

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

class Final_pipeline(BaseEstimator):
    def __init__(self, base_classifier):
        self.base_classifier =base_classifier
        self.classifier_per_column = []
        self.vectorizer = None

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def _get_training_texts(self, X:pd.DataFrame):
        all_texts = None
        for column in list(X.columns):
            if type(all_texts) == type(None):
                all_texts = X[[column]]
                all_texts.columns = ["fusion"]
            else:
                all_texts["fusion"] = all_texts["fusion"] + " " + X[column]

        dataset = X.join(all_texts)
        self.vectorizer = prepro.get_text_vectorizer(dataset, "fusion") # fit the vectorizer
        return dataset

    def fit(self, X:pd.DataFrame, y):
        dataset = self._get_training_texts(X)
        columns = list(dataset.columns)
        for column in columns:
            column_clf = Multi_classes_classifier_on_column(base_classifier=self.base_classifier, column=column)
            column_clf.set_vectorizer(self.vectorizer)
            self.classifier_per_column.append(column_clf.fit(dataset,y))
        return self

    def _get_max(self, prediction:np.ndarray):
        mean_prediction = prediction.mean(axis=0)
        max_mean = mean_prediction.max(axis=1)
        result = []
        for row, m_mean in zip(mean_prediction,max_mean):
            result.append(row==m_mean)

        result = np.array(result , dtype='int')
        result = result.astype(int)
        return result

    def predict(self, X):
        dataset = self._get_training_texts(X)
        result = []
        for column_clf_ in self.classifier_per_column:
            result.append(column_clf_.predict(dataset))
        result = self._get_max(np.array(result))
        return result

    def predict_proba(self, X):
        dataset = self._get_training_texts(X)
        result = []
        for column_clf in self.classifier_per_column:
            result.append(column_clf.predict_proba(dataset))
        return result

    def partial_fit(self,X, y):
        dataset = self._get_training_texts(X)
        result = []
        for column_clf in self.classifier_per_column:
            result.append(column_clf.partial_fit(dataset,y))
        return result

    def score(self,X, y):
        dataset = self._get_training_texts(X)
        result = []
        for column_clf in self.classifier_per_column:
            result.append(column_clf.score(dataset, y))
        return result

    def set_params(self, **params):
        result = []
        for column_clf in self.classifier_per_column:
            result.append(column_clf.set_params(**params))
        return result

    def get_params(self, deep):
        result = []
        for column_clf in self.classifier_per_column:
            result.append(column_clf.get_params(**deep))
        return result
