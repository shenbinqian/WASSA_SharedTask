# -*- coding: UTF-8 -*-
import pandas as pd
import nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import spatial
import pickle
# from embedding import embedding


def get_segmented_articles(file="../data/articles_adobe_AMT.csv"):
    article_raw = pd.read_csv(file)
    articles = article_raw[["article_id", "text"]]
    nltk.download('punkt')

    segmented = []
    for i in range(len(articles['text'].values)):
        segmented.append(sent_tokenize(articles['text'].values[i]))

    articles['segmented'] = segmented

    return articles


def sentwise_cosine_similarity(essays, articles, essay_embeddings, article_embeddings):
    similarities = []
    for i in range(len(essays['article_id'].values)):
        for j in articles['article_id'].values:
            if essays['article_id'][i] == j:
                artEm = article_embeddings[articles.index[articles['article_id'] == j].tolist()[0]]
                sent_similarity = []
                for k in range(artEm.shape[0]):
                    sent_similarity.append(round(1 - spatial.distance.cosine(essay_embeddings[i], artEm[k]), 2))
        similarities.append(sent_similarity)
    return similarities


def sentiment_scores(segmented_articles, essays):
    nltk.download('vader_lexicon')
    senti = SentimentIntensityAnalyzer()

    sentiment_scores = []
    for sent_list in segmented_articles['segmented'].values:
        sentiment_scores.append([round(senti.polarity_scores(sent)['compound'], 2) for sent in sent_list])

    scores_essay_order = []
    for i in range(len(essays['article_id'].values)):
        for j in segmented_articles['article_id'].values:
            if essays['article_id'][i] == j:
                artSenti = sentiment_scores[segmented_articles.index[segmented_articles['article_id']==j].tolist()[0]]
        scores_essay_order.append(artSenti)
    return scores_essay_order


def get_index_by_similarity(similarities, threshold=0.5):
    indices_list = []
    for sent_sims in similarities:
        indices = []
        for sim in sent_sims:
            if sim >= threshold:
                index = sent_sims.index(sim)
                indices.append(index)
        if indices:
            indices_list.append(indices)
        else:
            indices_list.append(None)
    return indices_list


def get_index_by_similariy_sentiScore(similarities, sentiScores, sim_threshold=0.5, senti_threshold=0.5):
    upper = senti_threshold
    lower = -senti_threshold
    indices_list = []
    for i in range(len(similarities)):
        indices = []
        for j in range(len(similarities[i])):
            if similarities[i][j] >= sim_threshold and (sentiScores[i][j] <= lower or sentiScores[i][j] >= upper):
                indices.append(j)
        if indices:
            indices_list.append(indices)
        else:
            indices_list.append(None)
    return indices_list


def remain_percentage(similarities, indices):
    percentage = []
    for i in range(len(indices)):
        if indices[i]:
            percentage.append(round(len(indices[i])/len(similarities[i]), 2))
        else:
            percentage.append(0.00)
    return percentage


def extract_text(indices_list, essays, articles):
    filtered = []
    for i in range(len(indices_list)):
        if indices_list[i]:
            j = essays['article_id'][i]
            artIn = articles.index[articles['article_id'] == j].tolist()[0]
            filtered.append([articles['segmented'][artIn][index] for index in indices_list[i]])
        else:
            filtered.append([''])
    return filtered


def concat_filtered_articles(essays, filtered_articles):
    concats = []
    for i in range(len(essays['essay'].values)):
        result = ''
        for sent in filtered_articles[i]:
            result += (sent + ' ')
        concats.append(essays['essay'].values[i] + ' [SEP] ' + result)
    return concats


def get_extracted_text(similarity_threshold=0.2, sentScore_threshold=0.6):

    train_raw = pd.read_table("../data/messages_train_ready_for_WS.tsv", delimiter='\t')
    dev_raw = pd.read_table('../data/messages_dev_features_ready_for_WS_2022.tsv', delimiter='\t')
    essays = pd.concat([train_raw[["article_id", "essay"]], dev_raw[["article_id", "essay"]]],
                       axis=0, ignore_index=True)

    articles = get_segmented_articles()

    '''
    article_embeddings = []
    for sent_list in articles['segmented'].values:
        article_embeddings.append(embedding(sent_list).reshape(-1, 512))

    essay_embeddings = embedding(essays['essay'].values)

    f = open('../data/embedded.data', 'wb')
    pickle.dump(article_embeddings, f)
    pickle.dump(essay_embeddings, f)
    f.close()
    '''

    of = open('../data/embedded.data', 'rb')
    article_embeddings = pickle.load(of)
    essay_embeddings = pickle.load(of)
    of.close()

    similarities = sentwise_cosine_similarity(essays, articles, essay_embeddings, article_embeddings)
    sentiScores = sentiment_scores(articles, essays)

    indices = get_index_by_similariy_sentiScore(similarities, sentiScores,
                                                sim_threshold=similarity_threshold,
                                                senti_threshold=sentScore_threshold)
    # percentage = remain_percentage(similarities, indices)

    filtered = extract_text(indices, essays, articles)
    raw_input = concat_filtered_articles(essays, filtered)

    return raw_input
