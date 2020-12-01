import pandas as pd
import random
from collections import Counter
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('corpora/wordnet')
nltk.download('wordnet')

lemm = True

examples_count = 2000

stemmer = PorterStemmer()
lemma = WordNetLemmatizer()
lemma.lemmatize('cats')
stop_words = set(stopwords.words('english'))


def preprocess(df, column):
    series = df[column]
    series = series.astype('string')
    series = series.apply(lambda x: re.sub(r'http\S+', '', x))
    series = series.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    series = series.apply(lambda x: str(x).lower())
    series = series.apply(lambda x: word_tokenize(x))
    series = series.apply(lambda x: [item for item in x if item not in stop_words])
    if not lemm:
        series = series.apply(lambda x: [stemmer.stem(item) for item in x])
    else:
        series = series.apply(lambda x: [lemma.lemmatize(word=w, pos='v') for w in x])
    series = series.apply(lambda x: [item for item in x if len(item) > 2])
    series = series.apply(lambda x: ' '.join(x))
    df[column] = series
    return df


def join_bodies(stances, bodies):
    bodies_map = {}
    for idx, row in bodies.iterrows():
        bodies_map[row['Body ID']] = row['articleBody']
    bodies_columns = []
    for idx, row in stances.iterrows():
        bodies_columns.append(bodies_map[row['Body ID']])
    stances['Body'] = bodies_columns
    return stances


def pad(data):
    c = Counter(list(data['Stance']))
    for cl in set(list(data['Stance'])):
        print(c)
        if c[cl] > examples_count:
            c[cl] = examples_count
            index_names = data.loc[data['Stance'] == cl].index
            data.drop(index_names[examples_count:], inplace=True)
        while c[cl] < examples_count:
            rows = data[data['Stance'] == cl]
            c[cl] += 1
            idx = random.randint(0, len(rows)-1)
            data = data.append(rows.iloc[idx])

    return data


if __name__ == '__main__':
    train_bodies = pd.read_csv('data-orig/train_bodies.csv')
    train_stances = pd.read_csv('data-orig/train_stances.csv')
    bodies = preprocess(train_bodies, 'articleBody')
    stances = preprocess(train_stances, 'Headline')
    data = join_bodies(stances, bodies)
    data = pad(data)
    data.to_csv(f'preprocessed_joined_data_{"lemm" if lemm else "stem"}_{examples_count}.csv')


