# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from utils import get_dataframe
import nltk


def viz_data(df, column='emotion'):
    if column == 'emotion':
        plt.hist(df[column])
        plt.title('Histogram of {}'.format(column))
        plt.xlabel('{}'.format(column))
        plt.ylabel('Counts')
        plt.show()

    elif column == 'empathy' or 'distress':
        plt.hist(df[column], bins=[1, 2, 3, 4, 5, 6, 7])
        plt.title('Histogram of {}'.format(column))
        plt.xlabel('{}'.format(column))
        plt.ylabel('Counts')
        plt.show()

    elif column == 'text_length':
        texts = df.essay.values
        word_tokenizer = nltk.WordPunctTokenizer()
        word_tokens = [word_tokenizer.tokenize(text) for text in texts]
        text_len = [len(tokenized) for tokenized in word_tokens]

        plt.hist(text_len)
        plt.title('Histogram of Text Length')
        plt.xlabel('Text Length')
        plt.ylabel('Counts')
        plt.show()

    else:
        print('Only support visualising distribution of "emotion", "empathy", "distress" or "text_length"!')


if __name__ == '__main__':
    df = get_dataframe()
    viz_data(df)