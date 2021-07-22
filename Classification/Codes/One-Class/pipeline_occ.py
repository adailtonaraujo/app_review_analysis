from gc import collect
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM as OCSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from os import path
import warnings
import sys
import os

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def init_metrics():
    metrics = {
        'precision': [],
        'recall': [],
        'f1-score': [],
        'auc_roc': [],
        'accuracy': [],
        'time': []
    }

    return metrics


def save_values(metrics, values):
    for key in metrics.keys():
        metrics[key].append(values[key])


def evaluation_one_class(preds_interest, preds_outliers):
    y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
    y_pred = list(preds_interest) + list(preds_outliers)
    return classification_report(y_true, y_pred, output_dict=True)


def fold_validation(folds):
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def train_test_split_pipeline(classes, kf, df, column_df):
    train_test_classes = {}

    for class_ in classes:

        rep_list = list()

        for train_index, test_index in kf.split(df[df['category'] == class_]):
            df_train = np.array(df[df['category'] == class_][column_df].to_list())[train_index]
            df_test = np.array(df[df['category'] == class_][column_df].to_list())[test_index]
            df_outlier = np.array(df[df['category'] != class_][column_df].to_list())

            rep_list.append((df_train, df_test, df_outlier))

            del df_train
            del df_test
            del df_outlier
            collect()

        train_test_classes[class_] = rep_list
        del rep_list
        collect()

    return train_test_classes


def evaluate_model(x_train, x_test, x_outlier, model):
    one_class_classifier = model.fit(x_train)

    y_pred_interest = one_class_classifier.predict(x_test)

    y_pred_ruido = one_class_classifier.predict(x_outlier)

    score_interest = one_class_classifier.decision_function(x_test)

    score_outlier = one_class_classifier.decision_function(x_outlier)

    y_true = np.array([1] * len(x_test) + [-1] * len(x_outlier))

    fpr, tpr, _ = roc_curve(y_true, np.concatenate([score_interest, score_outlier]))

    dic = evaluation_one_class(y_pred_interest, y_pred_ruido)

    metrics = {'precision': dic['1']['precision'], 'recall': dic['1']['recall'], 'f1-score': dic['1']['f1-score'],
               'auc_roc': roc_auc_score(y_true, np.concatenate([score_interest, score_outlier])),
               'accuracy': dic['accuracy']}

    return metrics, fpr, tpr


def evaluate_models(models, classes, reps_classes, file_name, line_parameters, path_results):
    for model in tqdm(models):

        metrics_classes = {}
        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '.csv'

        for class_ in classes:

            metrics = init_metrics()
            rep_list = reps_classes[class_]

            for reps in rep_list:
                start = time.time()
                values, fpr, tpr = evaluate_model(reps[0], reps[1], reps[2], models[model])
                end = time.time()
                temp = end - start
                values['time'] = temp

                save_values(metrics, values)

            metrics_classes[class_] = metrics

        write_results(metrics_classes, fn, lp, path_results)


def write_results(metrics_classes, file_name, line_parameters, path_):
    if not path.exists(path_ + file_name):
        file_ = open(path_ + file_name, 'w')
        string = 'Parameters'

        for class_ in metrics_classes.keys():
            for metric in metrics_classes[class_].keys():
                string += ';' + metric + '-' + class_ + '-mean' + ';' + metric + '-' + class_ + '-std'
        for metric in metrics_classes[list(metrics_classes.keys())[0]].keys():
            string += ';' + metric
        string += '\n'
        file_.write(string)
        file_.close()

    file_ = open(path_ + file_name, 'a')
    string = line_parameters

    for class_ in metrics_classes.keys():
        for metric in metrics_classes[class_].keys():
            string += ';' + str(np.mean(metrics_classes[class_][metric])) + ';' + str(
                np.std(metrics_classes[class_][metric]))

    for metric in metrics_classes[list(metrics_classes.keys())[0]].keys():
        soma = 0
        for class_ in metrics_classes.keys():
            soma += np.mean(metrics_classes[class_][metric])

        string += ';' + str(soma / len(metrics_classes.keys()))

    string += '\n'
    file_.write(string)
    file_.close()


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


def term_weight_type(term_weight, language):
    if term_weight == 'TF':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif term_weight == 'TFIDF':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif term_weight == 'Binary':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language), binary=True)
    elif term_weight == 'TF-Bg':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language))
    elif term_weight == 'TFIDF-BG':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language))
    else:
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2), min_df=1,
                                     tokenizer=MyTokenizer(language), binary=True)

    return vectorizer


def make_representation(train_test_classes, preprocessing, classes, vectorizer):
    reps_classes = {}

    for class_ in classes:

        rep_list = list()

        for df_train, df_test, df_outlier in train_test_classes[class_]:

            if preprocessing == 'BoW':
                vectorizer.fit(df_train)

                x_train = vectorizer.transform(df_train)
                x_test = vectorizer.transform(df_test)
                x_outlier = vectorizer.transform(df_outlier)

                rep_list.append((x_train.toarray(), x_test.toarray(), x_outlier.toarray()))

            else:

                rep_list.append((df_train, df_test, df_outlier))

        reps_classes[class_] = rep_list
        del rep_list
        collect()

    return reps_classes


def make_prepro_evaluate(train_test_classes, preprocessing, classes, line_parameters, file_name, path_results, models,
                         vectorizer=CountVectorizer()):
    representations = make_representation(train_test_classes, preprocessing, classes, vectorizer=vectorizer)

    evaluate_models(models, classes, representations, file_name, line_parameters, path_results)

    del representations
    collect()


def preprocessing_evaluate(df, dataset, preprocessing, models):
    path_results = '../../Results/'
    folds = 10
    kf = fold_validation(folds)
    line_parameters = ''
    term_weight_list = ['TFIDF', 'TF', 'Binary', 'TFIDF-Bg', 'TF-Bg', 'Binary-Bg']
    language = 'english'
    classes = df['category'].unique()

    if preprocessing == 'BoW':

        train_test_classes = train_test_split_pipeline(classes, kf, df, 'text')

        for term_weight in term_weight_list:
            print(preprocessing + ' ' + term_weight)

            vectorizer = term_weight_type(term_weight, language)

            file_name = dataset + '_' + preprocessing + '_' + term_weight

            make_prepro_evaluate(train_test_classes, preprocessing, classes, line_parameters, file_name, path_results,
                                 models, vectorizer=vectorizer)
    else:

        train_test_classes = train_test_split_pipeline(classes, kf, df, preprocessing)

        file_name = dataset + '_' + preprocessing

        make_prepro_evaluate(train_test_classes, preprocessing, classes, line_parameters, file_name, path_results,
                             models)

    del train_test_classes
    collect()


def run(datasets_dictionary, models, all_one_dataset, all_one_preprocessing):
    prepros = ['DBERT', 'DBERTML', 'RoBERTa', 'BERT', 'BoW']

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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    os.system("./download.sh")

    models = {
        'OCSVM_kernel-linear_scale_01': OCSVM(kernel='linear', gamma='scale', nu=0.1),
        'OCSVM_kernel-linear_scale_02': OCSVM(kernel='linear', gamma='scale', nu=0.2),
        'OCSVM_kernel-linear_scale_03': OCSVM(kernel='linear', gamma='scale', nu=0.3),
        'OCSVM_kernel-linear_scale_04': OCSVM(kernel='linear', gamma='scale', nu=0.4),
        'OCSVM_kernel-linear_scale_05': OCSVM(kernel='linear', gamma='scale', nu=0.5),
        'OCSVM_kernel-linear_scale_06': OCSVM(kernel='linear', gamma='scale', nu=0.6),
        'OCSVM_kernel-linear_scale_07': OCSVM(kernel='linear', gamma='scale', nu=0.7),
        'OCSVM_kernel-linear_scale_08': OCSVM(kernel='linear', gamma='scale', nu=0.8),
        'OCSVM_kernel-linear_scale_09': OCSVM(kernel='linear', gamma='scale', nu=0.9),
        'OCSVM_kernel-linear_auto_01': OCSVM(kernel='linear', gamma='auto', nu=0.1),
        'OCSVM_kernel-linear_auto_02': OCSVM(kernel='linear', gamma='auto', nu=0.2),
        'OCSVM_kernel-linear_auto_03': OCSVM(kernel='linear', gamma='auto', nu=0.3),
        'OCSVM_kernel-linear_auto_04': OCSVM(kernel='linear', gamma='auto', nu=0.4),
        'OCSVM_kernel-linear_auto_05': OCSVM(kernel='linear', gamma='auto', nu=0.5),
        'OCSVM_kernel-linear_auto_06': OCSVM(kernel='linear', gamma='auto', nu=0.6),
        'OCSVM_kernel-linear_auto_07': OCSVM(kernel='linear', gamma='auto', nu=0.7),
        'OCSVM_kernel-linear_auto_08': OCSVM(kernel='linear', gamma='auto', nu=0.8),
        'OCSVM_kernel-linear_auto_09': OCSVM(kernel='linear', gamma='auto', nu=0.9),
        'OCSVM_kernel-poly_2_scale_01': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.1),
        'OCSVM_kernel-poly_2_scale_02': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.2),
        'OCSVM_kernel-poly_2_scale_03': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.3),
        'OCSVM_kernel-poly_2_scale_04': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.4),
        'OCSVM_kernel-poly_2_scale_05': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.5),
        'OCSVM_kernel-poly_2_scale_06': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.6),
        'OCSVM_kernel-poly_2_scale_07': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.7),
        'OCSVM_kernel-poly_2_scale_08': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.8),
        'OCSVM_kernel-poly_2_scale_09': OCSVM(kernel='poly', degree=2, gamma='scale', nu=0.9),
        'OCSVM_kernel-poly_2_auto_01': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.1),
        'OCSVM_kernel-poly_2_auto_02': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.2),
        'OCSVM_kernel-poly_2_auto_03': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.3),
        'OCSVM_kernel-poly_2_auto_04': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.4),
        'OCSVM_kernel-poly_2_auto_05': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.5),
        'OCSVM_kernel-poly_2_auto_06': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.6),
        'OCSVM_kernel-poly_2_auto_07': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.7),
        'OCSVM_kernel-poly_2_auto_08': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.8),
        'OCSVM_kernel-poly_2_auto_09': OCSVM(kernel='poly', degree=2, gamma='auto', nu=0.9),
        'OCSVM_kernel-poly_3_scale_01': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.1),
        'OCSVM_kernel-poly_3_scale_02': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.2),
        'OCSVM_kernel-poly_3_scale_03': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.3),
        'OCSVM_kernel-poly_3_scale_04': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.4),
        'OCSVM_kernel-poly_3_scale_05': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.5),
        'OCSVM_kernel-poly_3_scale_06': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.6),
        'OCSVM_kernel-poly_3_scale_07': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.7),
        'OCSVM_kernel-poly_3_scale_08': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.8),
        'OCSVM_kernel-poly_3_scale_09': OCSVM(kernel='poly', degree=3, gamma='scale', nu=0.9),
        'OCSVM_kernel-poly_3_auto_01': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.1),
        'OCSVM_kernel-poly_3_auto_02': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.2),
        'OCSVM_kernel-poly_3_auto_03': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.3),
        'OCSVM_kernel-poly_3_auto_04': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.4),
        'OCSVM_kernel-poly_3_auto_05': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.5),
        'OCSVM_kernel-poly_3_auto_06': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.6),
        'OCSVM_kernel-poly_3_auto_07': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.7),
        'OCSVM_kernel-poly_3_auto_08': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.8),
        'OCSVM_kernel-poly_3_auto_09': OCSVM(kernel='poly', degree=3, gamma='auto', nu=0.9),
        'OCSVM_kernel-rbf_scale_01': OCSVM(kernel='rbf', gamma='scale', nu=0.1),
        'OCSVM_kernel-rbf_scale_02': OCSVM(kernel='rbf', gamma='scale', nu=0.2),
        'OCSVM_kernel-rbf_scale_03': OCSVM(kernel='rbf', gamma='scale', nu=0.3),
        'OCSVM_kernel-rbf_scale_04': OCSVM(kernel='rbf', gamma='scale', nu=0.4),
        'OCSVM_kernel-rbf_scale_05': OCSVM(kernel='rbf', gamma='scale', nu=0.5),
        'OCSVM_kernel-rbf_scale_06': OCSVM(kernel='rbf', gamma='scale', nu=0.6),
        'OCSVM_kernel-rbf_scale_07': OCSVM(kernel='rbf', gamma='scale', nu=0.7),
        'OCSVM_kernel-rbf_scale_08': OCSVM(kernel='rbf', gamma='scale', nu=0.8),
        'OCSVM_kernel-rbf_scale_09': OCSVM(kernel='rbf', gamma='scale', nu=0.9),
        'OCSVM_kernel-rbf_auto_01': OCSVM(kernel='rbf', gamma='auto', nu=0.1),
        'OCSVM_kernel-rbf_auto_02': OCSVM(kernel='rbf', gamma='auto', nu=0.2),
        'OCSVM_kernel-rbf_auto_03': OCSVM(kernel='rbf', gamma='auto', nu=0.3),
        'OCSVM_kernel-rbf_auto_04': OCSVM(kernel='rbf', gamma='auto', nu=0.4),
        'OCSVM_kernel-rbf_auto_05': OCSVM(kernel='rbf', gamma='auto', nu=0.5),
        'OCSVM_kernel-rbf_auto_06': OCSVM(kernel='rbf', gamma='auto', nu=0.6),
        'OCSVM_kernel-rbf_auto_07': OCSVM(kernel='rbf', gamma='auto', nu=0.7),
        'OCSVM_kernel-rbf_auto_08': OCSVM(kernel='rbf', gamma='auto', nu=0.8),
        'OCSVM_kernel-rbf_auto_09': OCSVM(kernel='rbf', gamma='auto', nu=0.9),
        'OCSVM_kernel-sigmoid_scale_01': OCSVM(kernel='sigmoid', gamma='scale', nu=0.1),
        'OCSVM_kernel-sigmoid_scale_02': OCSVM(kernel='sigmoid', gamma='scale', nu=0.2),
        'OCSVM_kernel-sigmoid_scale_03': OCSVM(kernel='sigmoid', gamma='scale', nu=0.3),
        'OCSVM_kernel-sigmoid_scale_04': OCSVM(kernel='sigmoid', gamma='scale', nu=0.4),
        'OCSVM_kernel-sigmoid_scale_05': OCSVM(kernel='sigmoid', gamma='scale', nu=0.5),
        'OCSVM_kernel-sigmoid_scale_06': OCSVM(kernel='sigmoid', gamma='scale', nu=0.6),
        'OCSVM_kernel-sigmoid_scale_07': OCSVM(kernel='sigmoid', gamma='scale', nu=0.7),
        'OCSVM_kernel-sigmoid_scale_08': OCSVM(kernel='sigmoid', gamma='scale', nu=0.8),
        'OCSVM_kernel-sigmoid_scale_09': OCSVM(kernel='sigmoid', gamma='scale', nu=0.9),
        'OCSVM_kernel-sigmoid_auto_01': OCSVM(kernel='sigmoid', gamma='auto', nu=0.1),
        'OCSVM_kernel-sigmoid_auto_02': OCSVM(kernel='sigmoid', gamma='auto', nu=0.2),
        'OCSVM_kernel-sigmoid_auto_03': OCSVM(kernel='sigmoid', gamma='auto', nu=0.3),
        'OCSVM_kernel-sigmoid_auto_04': OCSVM(kernel='sigmoid', gamma='auto', nu=0.4),
        'OCSVM_kernel-sigmoid_auto_05': OCSVM(kernel='sigmoid', gamma='auto', nu=0.5),
        'OCSVM_kernel-sigmoid_auto_06': OCSVM(kernel='sigmoid', gamma='auto', nu=0.6),
        'OCSVM_kernel-sigmoid_auto_07': OCSVM(kernel='sigmoid', gamma='auto', nu=0.7),
        'OCSVM_kernel-sigmoid_auto_08': OCSVM(kernel='sigmoid', gamma='auto', nu=0.8),
        'OCSVM_kernel-sigmoid_auto_09': OCSVM(kernel='sigmoid', gamma='auto', nu=0.9)
    }

    all_one_preprocessing = sys.argv[1]

    dataset = pd.read_pickle('../../Dataset/ARE.plk')

    run({'ARE': dataset}, models, 'ARE', all_one_preprocessing)
