# -*- coding: UTF-8 -*-

import numpy as np
import statistics
from statistics import StatisticsError
from scipy.stats.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from utils import get_dataframe, get_train_dev
from embedding import embedding


def main(problem='classification', embedding_type='USE'):
    df = get_dataframe()
    train_texts, dev_texts = get_train_dev(df['essay'])
    train_embeddings = embedding(train_texts, type=embedding_type)
    dev_embeddings = embedding(dev_texts, type=embedding_type)
    if problem == 'classification':
        train_labels, dev_labels = get_train_dev(df['emotion'])

        print('Initializing models...')
        model1 = LogisticRegression()
        model2 = RandomForestClassifier()

        kernel = 1.0 * RBF(1.0)
        model3 = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100)

        print("Training first model!")
        model1.fit(train_embeddings, train_labels.values.codes)
        print("Training second model!")
        model2.fit(train_embeddings, train_labels.values.codes)
        print("Training third model!")
        model3.fit(train_embeddings, train_labels.values.codes)

        pred1 = model1.predict(dev_embeddings)
        pred2 = model2.predict(dev_embeddings)
        pred3 = model3.predict(dev_embeddings)

        print('Max voting...')
        final_pred = np.array([])
        for i in range(0, len(dev_embeddings)):
            try:
                final_pred = np.append(final_pred, statistics.mode([pred1[i], pred2[i], pred3[i]]))
            except StatisticsError:
                final_pred = np.append(final_pred, pred1[i])

        accuracy = (np.mean(final_pred == dev_labels.values.codes))
        print('Accuracy for Emotion prediction is ' + str(accuracy))

    elif problem == 'regression':
        em_train, em_dev = get_train_dev(df['empathy'])
        dis_train, dis_dev = get_train_dev(df['distress'])

        print('Initializing models...')
        #model1E = SVR(gamma='auto').fit(train_embeddings, em_train)
        #model1D = SVR(gamma='auto').fit(train_embeddings, dis_train)

        model2E = RandomForestRegressor().fit(train_embeddings, em_train)
        model2D = RandomForestRegressor().fit(train_embeddings, dis_train)

        model3E = RidgeCV().fit(train_embeddings, em_train)
        model3D = RidgeCV().fit(train_embeddings, dis_train)

        kernel = DotProduct() + WhiteKernel()
        model4E = GaussianProcessRegressor(kernel=kernel).fit(train_embeddings, em_train)
        model4D = GaussianProcessRegressor(kernel=kernel).fit(train_embeddings, dis_train)

        print('Training models...')
        #pred1E = model1E.predict(dev_embeddings)
        pred2E = model2E.predict(dev_embeddings)
        pred3E = model3E.predict(dev_embeddings)
        pred4E = model4E.predict(dev_embeddings)

        #pred1D = model1D.predict(dev_embeddings)
        pred2D = model2D.predict(dev_embeddings)
        pred3D = model3D.predict(dev_embeddings)
        pred4D = model4D.predict(dev_embeddings)

        print('Average voting...')
        finalpredE = (pred2E + pred3E + pred4E) / 3
        finalpredD = (pred2D + pred3D + pred4D) / 3

        pearsonE = pearsonr(finalpredE, em_dev)
        pearsonD = pearsonr(finalpredD, dis_dev)

        print('Pearson correlation for Empathy is ' + str(pearsonE))
        print('Pearson correlation for Distress is ' + str(pearsonD))

    else:
        print('Must pass problem as "classification" or "regression"!')


if __name__ == '__main__':
    main(problem='classification', embedding_type='USE')