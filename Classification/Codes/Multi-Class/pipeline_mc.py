from gc import collect
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import warnings
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


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
    if bow_type == 'TF':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif bow_type == 'TFIDF':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif bow_type == 'Binary':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language), binary=True)
    elif bow_type == 'TF-Bg':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif bow_type == 'TFIDF-Bg':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language))
    else:
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language), binary=True)

    return vectorizer


def init_metrics():
    metrics = {
        'pbr_precision': [],
        'pbr_recall': [],
        'pbr_f1-score': [],
        'inq_precision': [],
        'inq_recall': [],
        'inq_f1-score': [],
        'irr_precision': [],
        'irr_recall': [],
        'irr_f1-score': [],
        'macro avg_precision': [],
        'macro avg_recall': [],
        'macro avg_f1-score': [],
        'accuracy': [],
        'time': []
    }
    return metrics


def save_values(metrics, values, time_):
    for key in metrics.keys():
        if key == 'time':
            metrics[key].append(time_)
        else:
            parts = key.split('_')
            class_ = parts[0]
            if len(parts) == 2:
                metric_ = parts[1]
                metrics[key].append(values[class_][metric_])
            else:
                metrics[key].append(values[class_])


def fold_validation(folds):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)


def train_test_split_pipeline(skf, x, y):
    x = np.array(x)
    y = np.array(y)

    representations = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        representations.append((x_train, x_test, y_train, y_test))

    return representations


def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    dic = classification_report(y_test, y_pred, output_dict=True)

    return dic


def evaluate_models(models, representatons, file_name, line_parameters, path_results):
    for model in tqdm(models):

        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '.csv'

        metrics = init_metrics()

        for reps in representatons:
            start = time.time()
            values = evaluate_model(models[model], reps[0], reps[1], reps[2], reps[3])
            end = time.time()
            time_ = end - start

            save_values(metrics, values, time_)

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

        else:
            lista_reps.append((old_x_train, old_x_test, old_y_train, old_y_test))

    return lista_reps


def make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models,
                         vectorizer=CountVectorizer()):
    representations = make_representation(train_test, preprocessing, vectorizer=vectorizer)

    evaluate_models(models, representations, file_name, line_parameters, path_results)

    del representations
    collect()


def preprocessing_evaluate(df, preprocessing, models):
    path_results = '../../Results/'
    folds = 10
    skf = fold_validation(folds)
    line_parameters = ''
    term_weight_list = ['TFIDF', 'TF', 'Binary', 'TFIDF-Bg', 'TF-Bg', 'Binary-Bg']
    language = 'english'

    if preprocessing == 'BoW':

        train_test = train_test_split_pipeline(skf, df['text'].to_list(), df['category'].to_list())

        for term_weight in term_weight_list:
            print(preprocessing + ' ' + term_weight)

            vectorizer = term_weight_type(term_weight, language)

            file_name = 'ARE_' + preprocessing + '_' + term_weight

            make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models,
                                 vectorizer=vectorizer)
    else:

        train_test = train_test_split_pipeline(skf, df[preprocessing].to_list(), df['category'].to_list())

        file_name = 'ARE_' + preprocessing

        make_prepro_evaluate(train_test, preprocessing, line_parameters, file_name, path_results, models)

    del train_test
    collect()


def run(dataset, models, all_one_preprocessing):
    prepros = ['BERT', 'DBERT', 'RoBERTa', 'DBERTML', 'BoW']

    if all_one_preprocessing != 'All':
        preprocessing_evaluate(dataset, all_one_preprocessing, models)
    else:
        for prepro in tqdm(prepros):
            preprocessing_evaluate(dataset, prepro, models)


def cosseno(x, y):
    dist = cosine(x, y)
    if np.isnan(dist):
        return 1
    return dist


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    os.system("./download.sh")

    models = {
        "SVM_kernel linear_random 42": SVC(kernel='linear', random_state=42),
        "SVM_kernel rbf_scale_random 42": SVC(kernel='rbf', gamma='scale', random_state=42),
        "SVM_kernel rbf_auto_random 42": SVC(kernel='rbf', gamma='auto', random_state=42),
        "SVM_kernel sigmoid_auto_random 42": SVC(kernel='sigmoid', gamma='auto', random_state=42),
        "SVM_kernel sigmoid_scale_random 42": SVC(kernel='sigmoid', gamma='scale', random_state=42),
        "SVM_kernel poly_g2_auto_random 42": SVC(kernel='poly', degree=2, gamma='auto', random_state=42),
        "SVM_kernel poly_g2_scale_random 42": SVC(kernel='poly', degree=2, gamma='scale', random_state=42),
        "SVM_kernel poly_g3_auto_random 42": SVC(kernel='poly', degree=3, gamma='auto', random_state=42),
        "SVM_kernel poly_g3_scale_random 42": SVC(kernel='poly', degree=3, gamma='scale', random_state=42),
        "SVM_kernel poly_g4_auto_random 42": SVC(kernel='poly', degree=4, gamma='auto', random_state=42),
        "SVM_kernel poly_g4_scale_random 42": SVC(kernel='poly', degree=4, gamma='scale', random_state=42),
        "SVM_kernel poly_g5_scale_random 42": SVC(kernel='poly', degree=5, gamma='scale', random_state=42),
        "SVM_kernel poly_g5_auto_random 42": SVC(kernel='poly', degree=5, gamma='auto', random_state=42),
        "SVM_kernel poly_g6_auto_random 42": SVC(kernel='poly', degree=6, gamma='auto', random_state=42),
        "MLP_Relu_layers (1)_neurons (50)": MLPClassifier(activation='relu', hidden_layer_sizes=(50,)),
        "MLP_Logistic_layers (1)_neurons(50)": MLPClassifier(activation='logistic', hidden_layer_sizes=(50,)),
        "MLP_TanHip_layers (1)_neurons(50)": MLPClassifier(activation='tanh', hidden_layer_sizes=(50,)),
        "MLP_Relu_layers(3)_neurons (50)": MLPClassifier(activation='relu', hidden_layer_sizes=(50, 50, 50)),
        "MLP_Logistic_layers (3)_neurons(50)": MLPClassifier(activation='logistic', hidden_layer_sizes=(50, 50, 50)),
        "MLP_TanHip_layers (3)_neurons(50)": MLPClassifier(activation='tanh', hidden_layer_sizes=(50, 50, 50)),
        "MLP_Relu_layers(6)_neurons (50)": MLPClassifier(activation='relu',
                                                         hidden_layer_sizes=(50, 50, 50, 50, 50, 50)),
        "MLP_Logistic_layers (6)_neurons(50)": MLPClassifier(activation='logistic',
                                                             hidden_layer_sizes=(50, 50, 50, 50, 50, 50)),
        "MLP_TanHip_layers (6)_neurons(50)": MLPClassifier(activation='tanh',
                                                           hidden_layer_sizes=(50, 50, 50, 50, 50, 50)),
        "MLP_Relu_layers (1)_neurons (100)": MLPClassifier(activation='relu', hidden_layer_sizes=(100,)),
        "MLP_Logistic_layers (1)_neurons(100)": MLPClassifier(activation='logistic', hidden_layer_sizes=(100,)),
        "MLP_TanHip_layers (1)_neurons(100)": MLPClassifier(activation='tanh', hidden_layer_sizes=(100,)),
        "MLP_Relu_layers(3)_neurons (100)": MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100, 100)),
        "MLP_Logistic_layers (3)_neurons(100)": MLPClassifier(activation='logistic',
                                                              hidden_layer_sizes=(100, 100, 100)),
        "MLP_TanHip_layers (3)_neurons(100)": MLPClassifier(activation='tanh', hidden_layer_sizes=(100, 100, 100)),
        "MLP_Relu_layers(6)_neurons (100)": MLPClassifier(activation='relu',
                                                          hidden_layer_sizes=(100, 100, 100, 100, 100, 100)),
        "MLP_Logistic_layers (6)_neurons(100)": MLPClassifier(activation='logistic',
                                                              hidden_layer_sizes=(100, 100, 100, 100, 100, 100)),
        "MLP_TanHip_layers (6)_neurons(100)": MLPClassifier(activation='tanh',
                                                            hidden_layer_sizes=(100, 100, 100, 100, 100, 100)),
        "MLP_Relu_layers (1)_neurons (150)": MLPClassifier(activation='relu', hidden_layer_sizes=(150,)),
        "MLP_Logistic_layers (1)_neurons(150)": MLPClassifier(activation='logistic', hidden_layer_sizes=(150,)),
        "MLP_TanHip_layers (1)_neurons(150)": MLPClassifier(activation='tanh', hidden_layer_sizes=(150,)),
        "MLP_Relu_layers(3)_neurons (150)": MLPClassifier(activation='relu', hidden_layer_sizes=(150, 150, 150)),
        "MLP_Logistic_layers (3)_neurons(150)": MLPClassifier(activation='logistic',
                                                              hidden_layer_sizes=(150, 150, 150)),
        "MLP_TanHip_layers (3)_neurons(150)": MLPClassifier(activation='tanh', hidden_layer_sizes=(150, 150, 150)),
        "MLP_Relu_layers(6)_neurons (150)": MLPClassifier(activation='relu',
                                                          hidden_layer_sizes=(150, 150, 150, 150, 150, 150)),
        "MLP_Logistic_layers (6)_neurons(150)": MLPClassifier(activation='logistic',
                                                              hidden_layer_sizes=(150, 150, 150, 150, 150, 150)),
        "MLP_TanHip_layers (6)_neurons(150)": MLPClassifier(activation='tanh',
                                                            hidden_layer_sizes=(150, 150, 150, 150, 150, 150)),
        "KNN_k=1": KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', metric=cosseno),
        "KNN_k=2": KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', metric=cosseno),
        "KNN_k=3": KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', metric=cosseno),
        "KNN_k=4": KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree', metric=cosseno),
        "KNN_k=5": KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', metric=cosseno),
        "KNN_k=6": KNeighborsClassifier(n_neighbors=6, algorithm='ball_tree', metric=cosseno),
        "KNN_k=7": KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', metric=cosseno),
        "KNN_k=8": KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree', metric=cosseno),
        "KNN_k=9": KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree', metric=cosseno),
        "KNN_k=10": KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', metric=cosseno),
        "KNN_k=11": KNeighborsClassifier(n_neighbors=11, algorithm='ball_tree', metric=cosseno),
        "KNN_k=12": KNeighborsClassifier(n_neighbors=12, algorithm='ball_tree', metric=cosseno),
        "KNN_k=13": KNeighborsClassifier(n_neighbors=13, algorithm='ball_tree', metric=cosseno),
        "KNN_k=14": KNeighborsClassifier(n_neighbors=14, algorithm='ball_tree', metric=cosseno),
        "KNN_k=15": KNeighborsClassifier(n_neighbors=15, algorithm='ball_tree', metric=cosseno),
        "KNN_k=16": KNeighborsClassifier(n_neighbors=16, algorithm='ball_tree', metric=cosseno),
        "KNN_k=17": KNeighborsClassifier(n_neighbors=17, algorithm='ball_tree', metric=cosseno),
        "KNN_k=18": KNeighborsClassifier(n_neighbors=18, algorithm='ball_tree', metric=cosseno),
        "KNN_k=19": KNeighborsClassifier(n_neighbors=19, algorithm='ball_tree', metric=cosseno),
        "KNN_k=20": KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree', metric=cosseno),
        "KNN_k=21": KNeighborsClassifier(n_neighbors=21, algorithm='ball_tree', metric=cosseno),
        "KNN_k=22": KNeighborsClassifier(n_neighbors=22, algorithm='ball_tree', metric=cosseno),
        "KNN_k=23": KNeighborsClassifier(n_neighbors=23, algorithm='ball_tree', metric=cosseno),
        "KNN_k=24": KNeighborsClassifier(n_neighbors=24, algorithm='ball_tree', metric=cosseno),
        "KNN_k=25": KNeighborsClassifier(n_neighbors=25, algorithm='ball_tree', metric=cosseno),
        "KNN_k=26": KNeighborsClassifier(n_neighbors=26, algorithm='ball_tree', metric=cosseno),
        "KNN_k=27": KNeighborsClassifier(n_neighbors=27, algorithm='ball_tree', metric=cosseno),
        "KNN_k=28": KNeighborsClassifier(n_neighbors=28, algorithm='ball_tree', metric=cosseno),
        "KNN_k=29": KNeighborsClassifier(n_neighbors=29, algorithm='ball_tree', metric=cosseno),
        "KNN_k=30": KNeighborsClassifier(n_neighbors=30, algorithm='ball_tree', metric=cosseno),
        "MNB_alpha 1.0_fit_prior": MultinomialNB(alpha=1.0, fit_prior=True),
        "MNB_alpha 1.0": MultinomialNB(alpha=1.0, fit_prior=False),
        "MNB_alpha 0.9_fit_prior": MultinomialNB(alpha=0.9, fit_prior=True),
        "MNB_alpha 0.9": MultinomialNB(alpha=0.9, fit_prior=False),
        "MNB_alpha 0.8_fit_prior": MultinomialNB(alpha=0.8, fit_prior=True),
        "MNB_alpha 0.8": MultinomialNB(alpha=0.8, fit_prior=False),
        "MNB_alpha 0.7_fit_prior": MultinomialNB(alpha=0.7, fit_prior=True),
        "MNB_alpha 0.7": MultinomialNB(alpha=0.7, fit_prior=False),
        "MNB_alpha 0.6_fit_prior": MultinomialNB(alpha=0.6, fit_prior=True),
        "MNB_alpha 0.6": MultinomialNB(alpha=0.6, fit_prior=False),
        "MNB_alpha 0.5_fit_prior": MultinomialNB(alpha=0.5, fit_prior=True),
        "MNB_alpha 0.5": MultinomialNB(alpha=0.5, fit_prior=False),
        "MNB_alpha 0.4_fit_prior": MultinomialNB(alpha=0.4, fit_prior=True),
        "MNB_alpha 0.4": MultinomialNB(alpha=0.4, fit_prior=False),
        "MNB_alpha 0.3_fit_prior": MultinomialNB(alpha=0.3, fit_prior=True),
        "MNB_alpha 0.3": MultinomialNB(alpha=0.3, fit_prior=False),
        "MNB_alpha 0.2_fit_prior": MultinomialNB(alpha=0.2, fit_prior=True),
        "MNB_alpha 0.2": MultinomialNB(alpha=0.2, fit_prior=False),
        "MNB_alpha 0.1_fit_prior": MultinomialNB(alpha=0.1, fit_prior=True),
        "MNB_alpha 0.1": MultinomialNB(alpha=0.1, fit_prior=False),
        "MNB_alpha 0.0_fit_prior": MultinomialNB(alpha=0.0, fit_prior=True),
        "MNB_alpha 0.0": MultinomialNB(alpha=0.0, fit_prior=False)
    }

    all_one_preprocessing = sys.argv[1]

    dataset = pd.read_pickle('../../Dataset/ARE.plk')

    run(dataset, models, all_one_preprocessing)
