import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
NUM_CLASSES = 5

class_mapping = {
    0: 'Bok',
    1: 'Peace',
    2: 'Like',
    3: 'Dislike',
    4: 'Okej'
}


model = load_model('model')
dataset = 'data.csv'
datasetNew = 'newData.csv'

x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (42) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED, stratify=y_dataset)

x_dataset_new = np.loadtxt(datasetNew, delimiter=',', dtype='float32', usecols=list(range(1, (42) + 1)))
y_dataset_new = np.loadtxt(datasetNew, delimiter=',', dtype='int32', usecols=(0))

def print_confusion_matrix(y_true, y_pred, title, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    plt.title(title)
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False, cmap='viridis')
    ax.set_ylim(len(set(y_true)), 0)
    
    plt.show()
    
    if report:
        print('Classification Report '+title)
        print(classification_report(y_test_text, y_pred_text))

Y_pred = model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)

y_pred_text = [class_mapping[pred] for pred in y_pred]
y_test_text = [class_mapping[true] for true in y_test]

print_confusion_matrix(y_test_text, y_pred_text,'Testni podaci')

train_predictions = model.predict(x_train)
train_predictions = np.argmax(train_predictions, axis=1)

y_train_text = [class_mapping[true] for true in y_train]
train_pred_text = [class_mapping[pred] for pred in train_predictions]

print_confusion_matrix(y_train_text, train_pred_text, 'Trening podaci')

test_predictions = model.predict(x_dataset_new)
test_predictions = np.argmax(test_predictions, axis=1)

y_newTest_text = [class_mapping[true] for true in y_dataset_new]
y_newPred_text = [class_mapping[pred] for pred in test_predictions]

print_confusion_matrix(y_newTest_text, y_newPred_text, 'Novi testni podaci')