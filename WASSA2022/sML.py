# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from utils import get_train_dev
from embedding import embedding

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer


class StatsModels:
    def __init__(self, df_features):
        self.df_features = df_features

        self.cat = self.transform_cat()
        self.num = self.transform_num()
        self.text_embeddings = self.transform_text()
        self.text_cat = self.concat_cat()
        self.text_num = self.concat_num()
        self.text_cat_num = self.concat_all()

    def transform_cat(self):
        cat = self.df_features[['gender', 'education', 'race']].values
        encoder = OneHotEncoder()
        return encoder.fit_transform(cat).toarray()

    def transform_num(self):
        num = self.df_features[['age', 'income', 'personality_conscientiousness',
                                'personality_openess', 'personality_extraversion',
                                'personality_agreeableness', 'personality_stability']].values
        Qt = QuantileTransformer(output_distribution='normal')
        return Qt.fit_transform(num)

    def transform_text(self):
        text = self.df_features['essay']
        return embedding(text)

    def concat_cat(self):
        return np.concatenate((self.text_embeddings, self.cat), axis=1)

    def concat_num(self):
        return np.concatenate((self.text_embeddings, self.num), axis=1)

    def concat_all(self):
        return np.concatenate((self.text_embeddings, self.cat, self.num), axis=1)

    def fit_SML_models(self, model='SVM', problem='emotion', features='text'):
        if features not in ('text', 'text_and_cat', 'text_and_num', 'all'):
            print('Features need to be "text", "text_and_cat", "text_and_num" or "all"!')

        else:
            if features == 'text':
                X_train, X_test = get_train_dev(self.text_embeddings)

            elif features == 'text_and_cat':
                X_train, X_test = get_train_dev(self.text_cat)

            elif features == 'text_and_num':
                X_train, X_test = get_train_dev(self.text_num)

            else:
                X_train, X_test = get_train_dev(self.text_cat_num)

            if problem == 'emotion':
                y_train, y_test = get_train_dev(self.df_features[problem])
                y_train = y_train.values.codes
                y_test = y_test.values.codes

                if model == 'SVM':
                    clf = SVC(gamma='auto')
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    print('Accuracy: ' + str(np.mean(preds == y_test)))

                elif model == 'RF':
                    clf = RandomForestClassifier()
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    print('Accuracy: ' + str(np.mean(preds == y_test)))

                elif model == 'GP':
                    kernel = 1.0 * RBF(1.0)
                    clf = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    print('Accuracy: ' + str(np.mean(preds == y_test)))

                elif model == 'LR':
                    clf = LogisticRegression()
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_test)
                    print('Accuracy: ' + str(np.mean(preds == y_test)))

                else:
                    print('Model needs to be "SVM", "RF", "GP", or "LR"!')

            elif problem == 'empathy' or 'distress':
                y_train, y_test = get_train_dev(self.df_features[problem])

                if model=='SVM':
                    rg = SVR(gamma='auto')
                    rg.fit(X_train, y_train)
                    preds = rg.predict(X_test)
                    cor = pearsonr(preds, y_test)[0]
                    print('Pearson correlation: ' + str(cor))

                elif model == 'RF':
                    rg = RandomForestRegressor()
                    rg.fit(X_train, y_train)
                    preds = rg.predict(X_test)
                    cor = pearsonr(preds, y_test)[0]
                    print('Pearson correlation: ' + str(cor))

                elif model == 'GP':
                    kernel = DotProduct() + WhiteKernel()
                    rg = GaussianProcessRegressor(kernel=kernel)
                    rg.fit(X_train, y_train)
                    preds = rg.predict(X_test)
                    cor = pearsonr(preds, y_test)[0]
                    print('Pearson correlation: ' + str(cor))

                elif model == 'Ridge':
                    rg = RidgeCV()
                    rg.fit(X_train, y_train)
                    preds = rg.predict(X_test)
                    cor = pearsonr(preds, y_test)[0]
                    print('Pearson correlation: ' + str(cor))

                else:
                    print('Model needs to be "SVM", "RF", "GP", or "Ridge"!')

            else:
                print('We only deal with emotion, empathy or distress problems!')


if __name__ == '__main__':

    df = pd.read_table("messages_train_ready_for_WS.tsv", delimiter='\t')
    df['emotion'] = pd.Categorical(df.emotion)
    df['gender'] = pd.Categorical(df.gender)
    df['education'] = pd.Categorical(df.education)
    df['race'] = pd.Categorical(df.race)
    df_features = df.iloc[:, [3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    sml = StatsModels(df_features)
    sml.fit_SML_models(model='SVM', problem='empathy', features='text_and_num')

