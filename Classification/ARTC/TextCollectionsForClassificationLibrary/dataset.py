import pandas as pd
import os
from pathlib import Path
import csv
import json

def read_csv_berts(file_name, df):
  with open(file_name, 'r') as arq:
    mat_embedding = list()
    reader = csv.reader(arq)
    for doc in reader:
      mat_embedding.append([float(i) for i in doc])

    col = len(mat_embedding[0])
    df2 = pd.DataFrame(mat_embedding, columns=range(col))

    df2['class'] = df['label']

    return df2

def load(path_files=''):

  '''
  this function download, load the dataset,  and return a dictionary in which any key represent the dataset
  
  if the dataset file exist, to load them, pass the path where the directory are through the path_files parameter
  
  this function changes the current directory, remember to go back to the directory you were in
  '''

  if path_files != '':
    os.chdir(path_files)

  if not Path('RevisoesSoftware.json').is_file():
    os.system('gdown --id 1D8kjCsLi1JteJSGNqwF7sxu13MpPmcn-')
  
  if not Path('bert-large-nli-stsb-mean-tokens.csv').is_file():
    os.system('gdown --id 1opBT5gZuSplDX2rd34GjubOF4Mk-GosA')
  
  if not Path('roberta-large-nli-stsb-mean-tokens.csv').is_file():
    os.system('gdown --id 1LaU3V8NtKWI3cB-12PjObRMHeYHeqIRL')
  
  if not Path('distilbert-base-nli-stsb-mean-tokens.csv').is_file():
    os.system('gdown --id 1Ar1Q_xSBUltlu6J-NyTDFAP60kDBuF6S')
  
  if not Path('distiluse-base-multilingual-cased.csv').is_file():
    os.system('gdown --id 1m0h4c5Tmgys-Po246qm9PtW1Q3O60Uqe')


  with open('RevisoesSoftware.json', 'r') as f:
    data = json.load(f)

  df = pd.DataFrame(data)

  bases = {
    'Text' : df,
    'BERT' : read_csv_berts('bert-large-nli-stsb-mean-tokens.csv',df),
    'RoBERTa' : read_csv_berts('roberta-large-nli-stsb-mean-tokens.csv',df),
    'DistilBERT' : read_csv_berts('distilbert-base-nli-stsb-mean-tokens.csv',df),
    'DistilBERT Multilingua' : read_csv_berts('distiluse-base-multilingual-cased.csv',df),

  }

  return bases



