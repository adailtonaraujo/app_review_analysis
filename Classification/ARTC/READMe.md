# App Review Text Collection (ARTC)

# TextCollectionsForClassificationLibrary
Library to use a text collectin present in the article: On the automatic classification of app reviews (Maalej, W., Kurtanovi ́c, Z., Nabil, H., and Stanik, C. (2016)). The article are avaliable at "https://link.springer.com/article/10.1007/s00766-016-0251-9"

# How To use

!pip install git+https://github.com/adailtonaraujo/app_review_analysis/tree/master/Classification/ARTC

from TextCollectionsForClassificationLibrary import dataset

bases = dataset.load()

bases[key] return a DataFrame

# Keys
- Text (all atributtes from dataset)
- BERT
- DistilBERT
- DistilBERT Multilingua
- RoBERTa
