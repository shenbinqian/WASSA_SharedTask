# -*- coding: utf-8 -*-

import pandas as pd
import re


def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def clean_data(file='../data/goEmotions.xlsx', sent_length=15):
    print('Reading goEmotions dataset...')
    raw = pd.read_excel(file)
    cleaned_text = []
    text_label = []
    for i in range(len(raw['text'].values)):
        withoutEmoji = remove_emoji(raw['text'].values[i])
        tokens = withoutEmoji.split()
        if len(tokens) >= sent_length and '[NAME]' not in tokens and '[name]' not in tokens:
            text = ' '.join(tokens)
            text = re.sub(r"\*", '', text)
            text = re.sub(r">", '', text)
            text = re.sub(r"\\", '', text)
            cleaned_text.append(text.lower())
            text_label.append(raw['labels'].values[i])

    d = {'text': cleaned_text, 'labels': text_label}
    df = pd.DataFrame(data=d)
    df['labels'] = pd.Categorical(df.labels)
    return df
