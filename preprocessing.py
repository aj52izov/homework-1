import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pdb

nltk.download('punkt')
nltk.download('stopwords')

stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))


def separe_data_per_class(data: pd.DataFrame, label_col: str = "label"):
    cathegories = list(dict.fromkeys(list(data[label_col])).keys())
    result =[]
    for cathegory in cathegories:
        result.append(data.loc[data[label_col] == cathegory])
    return result, cathegories

def multiply_data_in_part(data: pd.DataFrame, nbr_time: int = 2):
    decimal_ = nbr_time - int(nbr_time)
    rest = int(len(data)*decimal_)
    result =data
    for i in range(int(nbr_time)-1):
        result = result.append(data , ignore_index=True)

    result = result.append(result.sample(n=rest, random_state=42) , ignore_index=True)
    return result

def divise_data_in_balanced_data(data: pd.DataFrame):
    list_dataset_set_per_category, list_cathegories = separe_data_per_class(data)

    aug_dataset_income = multiply_data_in_part(list_dataset_set_per_category[list_cathegories.index('income')], 4)
    aug_dataset_living = multiply_data_in_part(list_dataset_set_per_category[list_cathegories.index('living')], 2.5)
    aug_dataset_private = multiply_data_in_part(list_dataset_set_per_category[list_cathegories.index('private')], 3)
    aug_dataset_standardOfLiving = multiply_data_in_part(
        list_dataset_set_per_category[list_cathegories.index('standardOfLiving')], 1.4)
    aug_dataset_leisure = multiply_data_in_part(list_dataset_set_per_category[list_cathegories.index('leisure')], 1)
    aug_dataset_finance = multiply_data_in_part(list_dataset_set_per_category[list_cathegories.index('finance')], 2)

    all_data_set = aug_dataset_income.append(aug_dataset_living, ignore_index=True).append(aug_dataset_private, ignore_index=True). \
        append(aug_dataset_standardOfLiving, ignore_index=True).append(aug_dataset_leisure, ignore_index=True). \
        append(aug_dataset_finance, ignore_index=True)

    return all_data_set

def plot_train_test_per_class(training_data: pd.DataFrame, testing_data: pd.DataFrame, label_col: str = "label",
                              one_feature_col="Verwendungszweck",train_title='Training data-set',test_title='Testing data-set'):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    training_data.groupby(label_col)[one_feature_col].count().plot.bar(ylim=0, ax=ax1, xlabel="Classes").set_title(train_title)
    testing_data.groupby(label_col)[one_feature_col].count().plot.bar(ylim=0, ax=ax2, xlabel="Classes").set_title(test_title)
    plt.show()


def plot_data_per_class(training_data: pd.DataFrame, label_col: str = "label", one_feature_col="Verwendungszweck",title='Final Training data-set'):
    fig = plt.figure(figsize=(8, 6))
    training_data.groupby(label_col)[one_feature_col].count().plot.bar(ylim=0 ,xlabel="Classes").set_title(title)
    plt.show()


def read_data(file_name: str, col: list = []) -> pd.DataFrame:
    if len(col) != 0:
        data = pd.read_csv(file_name, usecols=col, sep=';', )
    else:
        data = pd.read_csv(file_name, index_col=0, sep=';')
    return data.fillna("")

model_de = spacy.load('de_dep_news_trf')
def _lemmatizer(text):
    """
    Lemmetize words using spacy
    :param: text as string
    :return: lemmetized text as string
    """
    sent = []
    doc = model_de(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)


def clean_text(text):
    """
        - remove any html tags (< /br> often found)
        - Keep only ASCII + European Chars and whitespace, no digits
        - remove single letter chars
        - convert all whitespaces (tabs etc.) to single wspace
        - all lowercase
        - remove stopwords, punctuation and stemm
    """
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_TAGS = re.compile(r"<[^>]+>")
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)

    text = re.sub(RE_TAGS, " ", text)
    text = re.sub(RE_ASCII, " ", text) # remove digits and punctuation
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text)
    words_tokens_lower = [word.lower() for word in word_tokens]

    #words_filtered = [stemmer.stem(word) for word in words_tokens_lower if word not in stop_words]
    words_filtered = [word for word in words_tokens_lower if word not in stop_words]

    text_clean = " ".join(words_filtered)
    text_clean = _lemmatizer(text_clean)
    return text_clean

# text to vector
def get_text_vectorizer(dataset: pd.DataFrame, text_data_col_as_string: str):
    text_data = dataset[text_data_col_as_string]
    text_data = [clean_text(text) for text in text_data] # text cleaning
    # vectorizer_word = CountVectorizer(vocabulary=vocabulary[0])
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    vectorizer.fit(text_data)
    return vectorizer
