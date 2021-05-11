import pandas as pd
import preprocessing as prepro
import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import pdb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
from sklearn.metrics import accuracy_score
import numpy as np

one_hot_label_encoder = OneHotEncoder(handle_unknown='ignore')
label_encoder = LabelEncoder()


def plot_confusion_matrix(cm,
                          target_names,accuracy=0,
                          title='Normalized Confusion matrix',
                          cmap=None,
                          normalize=True):
    if accuracy==0:
        accuracy = np.trace(cm) / np.sum(cm).astype('float')

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels\naccuracy={:0.4f}'.format(float(accuracy)))
    plt.show()


def get_vectorizer(dataset: pd.DataFrame, to_read_cols: list):
    all_texts = None
    for column in to_read_cols:
        if type(all_texts) == type(None):
            all_texts = dataset[[column]]
            all_texts.columns = ["fusion"]
        else:
            all_texts["fusion"] = all_texts["fusion"] + " " + dataset[column]

    vectorizer = prepro.get_text_vectorizer(all_texts, "fusion")
    return vectorizer


def split_label_column(dataset: pd.DataFrame, label_col: str = "label"):
    label_encoder.fit(dataset[[label_col]].values)
    dump(label_encoder.classes_, "label_encoder.joblib")
    labels = one_hot_label_encoder.fit_transform(dataset[[label_col]].values.reshape(-1, 1)).toarray()
    # np.save('one_hot_label_encoder.npy', one_hot_label_encoder.categories_)
    dump(one_hot_label_encoder.categories_, "one_hot_label_encoder.joblib")
    dataset = dataset.drop(columns=label_col)
    return dataset, labels


def model_training(train_dataset: pd.DataFrame, model: pipeline.Final_pipeline, to_read_cols: list, label_col: str):
    dataset = train_dataset[to_read_cols]
    train__dataset, labels = split_label_column(dataset, label_col)
    model.fit(train__dataset, labels)
    dump(model, "transaction_clf.joblib")


def prediction(test_dataset: pd.DataFrame, model: pipeline.Final_pipeline, to_read_cols: list, label_col: str):
    test_dataset = test_dataset[to_read_cols]
    test_dataset, Y_test = split_label_column(test_dataset, label_col)
    if type(model) == type(str("l")):
        model = load(model)
    result = model.predict(test_dataset)

    label_encoder.classes_ = load('label_encoder.joblib')
    one_hot_label_encoder.categories_ = load('one_hot_label_encoder.joblib')
    result = np.array(label_encoder.transform(one_hot_label_encoder.inverse_transform(result)))
    Y_test = np.array(label_encoder.transform(one_hot_label_encoder.inverse_transform(Y_test)))
    return result, Y_test


def evaluation(Y_pred, Y_true):
    labels = list(one_hot_label_encoder.categories_[0])
    conf_mat = confusion_matrix(Y_true, Y_pred)
    plot_confusion_matrix(conf_mat, labels,accuracy_score(Y_true, Y_pred))


# get the dataset
dataset_path = "Data_Set.csv"
dataset = prepro.read_data(file_name=dataset_path)  # read data

# split data in training and testing data
trainingdataset, testing_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Plot training and testing data-set
prepro.plot_train_test_per_class(trainingdataset, testing_dataset)

# resampling the trainin data-set for balancing
oversampled =  prepro.divise_data_in_balanced_data(trainingdataset)

# Plot the final balanced training data-set
prepro.plot_data_per_class(oversampled)

# get the  model
training_pipeline = pipeline.Final_pipeline(base_classifier=GaussianNB())

# Train the model with specific columns and store the model in "transaction_clf.joblib"
model_training(oversampled , training_pipeline, ["Buchungstext","Verwendungszweck","Beguenstigter/Zahlungspflichtiger","label"], label_col="label")

# make prediction on test data-set
Y_pred, Y_test_true = prediction(testing_dataset, "transaction_clf.joblib",
                           ["Buchungstext", "Verwendungszweck", "Beguenstigter/Zahlungspflichtiger", "label"],
                            label_col="label")

# evaluate the model
evaluation(Y_pred, Y_test_true)
