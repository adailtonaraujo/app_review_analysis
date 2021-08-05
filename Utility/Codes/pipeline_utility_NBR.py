from gc import collect
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.linear_model import BayesianRidge as NBR
import warnings
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def normalize_thumbs(df):
    max_ = np.max(df['thumbsUpCount'].to_list())

    thumbs_up = list()

    for num in df['thumbsUpCount'].to_list():
        thumbs_up.append(num / max_)

    return thumbs_up


def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)


class MyTokenizer:
    def __init__(self, language):
        self.wnl = WordNetLemmatizer()
        if language == 'english':
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
            self.stemmer = nltk.stem.SnowballStemmer('english')

        if language == 'spanish':
            self.STOPWORDS = nltk.corpus.stopwords.words('spanish')
            self.stemmer = nltk.stem.SnowballStemmer('spanish')

        if language == 'portuguese':
            self.STOPWORDS = nltk.corpus.stopwords.words('portuguese')
            self.stemmer = nltk.stem.SnowballStemmer('portuguese')

        if language == 'multilingual':
            self.STOPWORDS = set(nltk.corpus.stopwords.words('spanish')).union(
                set(nltk.corpus.stopwords.words('portuguese'))).union(set(nltk.corpus.stopwords.words('english')))
            self.stemmer = nltk.stem.SnowballStemmer('english')

    def __call__(self, doc):
        l1 = [t for t in word_tokenize(doc)]
        l2 = []
        for token in l1:
            if token not in self.STOPWORDS and token.isnumeric() is False and len(token) > 2 and has_numbers(
                    token) is False:
                l2.append(token)
        l3 = [self.stemmer.stem(self.wnl.lemmatize(t)) for t in l2]
        return l3


def term_weight_type(bow_type, language):
    if bow_type == 'term-frequency':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif bow_type == 'term-frequency-IDF':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    else:
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language), binary=True)

    return vectorizer


def load_dataset(all_one_dataset):
    datasets = {}
    if all_one_dataset == 'All':
        basepath = Path('../Dataset/with_embeddings/')
        files_in_basepath = basepath.iterdir()
        for item in files_in_basepath:
            datasets[item.name.replace('.plk', '')] = pd.read_pickle('../Dataset/with_embeddings/' + item.name)
    else:
        datasets[all_one_dataset] = pd.read_pickle('../Dataset/with_embeddings/' + all_one_dataset + '.plk')

    return datasets


def init_metrics():
    metrics = {
        'mse': [],
        'mae': [],
        'r2': [],
        'time': []
    }

    return metrics


def save_values(metrics, values):
    for key in metrics.keys():
        metrics[key].append(values[key])


def fold_validation(folds):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)


def train_test_split_pipeline(skf, x, y):
    x = np.array(x)
    y = np.array(y)

    bins = np.linspace(0, 1, 100)
    y_binned = np.digitize(y, bins)

    representations = []

    for train_index, test_index in skf.split(x, y_binned):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        representations.append((x_train, x_test, y_train, y_test))

    return representations


def evaluate_model(model, x_train, x_test, y_train, y_test):

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    metrics = {'mse': mse(y_test, y_pred),
               'mae': mae(y_test, y_pred),
               'r2': r2(y_test, y_pred)}

    return metrics


def evaluate_models(models, representatopns, file_name, line_parameters, path_results):
    for model in tqdm(models):

        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '.csv'

        metrics = init_metrics()

        for reps in representatopns:
            start = time.time()
            values = evaluate_model(models[model], reps[0], reps[1], reps[2], reps[3])
            end = time.time()
            time_ = end - start
            values['time'] = time_

            save_values(metrics, values)

        write_results(metrics, fn, lp, path_results)


def write_results(metrics, file_name, line_parameters, path_results):
    if not Path(path_results + file_name).is_file():
        file_ = open(path_results + file_name, 'w')
        string = 'Parameters'
        for metric in metrics.keys():
            string += ';' + metric + '-mean' + ';' + metric + '-std'
        string += '\n'
        file_.write(string)
        file_.close()

    file_ = open(path_results + file_name, 'a')
    string = line_parameters

    for metric in metrics.keys():
        string += ';' + str(np.mean(metrics[metric])) + ';' + str(np.std(metrics[metric]))

    string += '\n'
    file_.write(string)
    file_.close()


def make_representation(train_test, preprocessing, vectorizer):
    lista_reps = list()

    for old_x_train, old_x_test, old_y_train, old_y_test in train_test:

        if preprocessing == 'BoW':
            vectorizer.fit(old_x_train)

            x_train = vectorizer.transform(old_x_train)
            x_test = vectorizer.transform(old_x_test)

            lista_reps.append((x_train.toarray(), x_test.toarray(), old_y_train, old_y_test))

        elif preprocessing == 'DBERTML':

            lista_reps.append((old_x_train, old_x_test, old_y_train, old_y_test))

    return lista_reps


def make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models,
                         vectorizer=CountVectorizer()):
    representations = make_representation(train_test, preprocessing, vectorizer=vectorizer)

    evaluate_models(models, representations, file_name, line_parameters, path_results)

    del representations
    collect()


def preprocessing_evaluate(df, dataset, preprocessing, models):
    path_results = '../Results/'
    folds = 10
    skf = fold_validation(folds)
    line_parameters = ''
    term_weight_list = ['term-frequency-IDF']
    language = 'multilingual'

    thumbsup = normalize_thumbs(df)

    if preprocessing == 'BoW':

        train_test = train_test_split_pipeline(skf, df['content'].to_list(), thumbsup)

        for term_weight in term_weight_list:
            print(preprocessing + ' ' + term_weight)

            vectorizer = term_weight_type(term_weight, language)

            file_name = dataset + '_' + preprocessing + '_' + term_weight

            make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models,
                                 vectorizer=vectorizer)
    elif preprocessing == 'DBERTML':

        train_test = train_test_split_pipeline(skf, df['DBERTML'].to_list(), thumbsup)

        print(preprocessing)

        file_name = dataset + '_' + preprocessing

        make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models)

    del train_test
    collect()


def run(datasets_dictionary, models, all_one_dataset, all_one_preprocessing):
    prepros = ['DBERTML', 'BoW']

    if all_one_dataset != 'All':
        if all_one_preprocessing != 'All':
            preprocessing_evaluate(datasets_dictionary[all_one_dataset], all_one_dataset, all_one_preprocessing, models)
        else:
            for prepro in tqdm(prepros):
                preprocessing_evaluate(datasets_dictionary[all_one_dataset], all_one_dataset, prepro, models)
    else:
        for dataset in tqdm(datasets_dictionary.keys()):
            if all_one_preprocessing != 'All':
                preprocessing_evaluate(datasets_dictionary[dataset], dataset, all_one_preprocessing,
                                       models)
            else:
                for prepro in tqdm(prepros):
                    preprocessing_evaluate(datasets_dictionary[dataset], dataset, prepro, models)


def cosseno(x, y):
    dist = cosine(x, y)
    if np.isnan(dist):
        return 1
    return dist


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    os.system("./download.sh")

    models = {
        "NBR": NBR()
    }

    all_one_dataset = sys.argv[1]

    all_one_preprocessing = sys.argv[2]

    datasets_dic = load_dataset(all_one_dataset)

    run(datasets_dic, models, all_one_dataset, all_one_preprocessing)
