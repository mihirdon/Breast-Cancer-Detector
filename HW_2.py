from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def generate_random_bit(length, seed):
    random.seed(seed)
    bit_list = np.array([])

    for i in range(length):
        rand_int = random.randint(0, 1)
        bit_list = np.append(bit_list, rand_int)

    return bit_list


def generate_table(data, alpha, epochs, iters):
    final_table = {}
    columns = np.array(list(range(9))) + 1
    spec = []
    sens = []
    seed = 1

    for i in range(iters):
        bit_list = generate_random_bit(9, seed)
        seed += 1

        active_cols = bit_list * columns
        active_cols = active_cols[active_cols != 0]
        active_cols -= 1
        X = data.iloc[:, active_cols]

        data_subset = X.copy()
        data_subset['outcome'] = data['outcome']

        weights, progress = perceptron(data_subset, alpha, epochs)
        yp = predict(X, weights)
        met = metrics(data_subset['outcome'], yp)
        final_table[i] = np.concatenate((bit_list, list(met.values())), axis=0)
        spec.append(met['specificity'])
        sens.append(met['sensitivity'])

    return final_table, spec, sens


def graph_line(x, y, xlabel, ylabel, title, scatterplot=False):
    plt.figure(figsize=(8, 8))
    if scatterplot:
        plt.scatter(x, y)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def linear_eqn (X, w):
    return w[0] + X.dot(w[1:])


def get_confusion_matrix(y, y_pred):
    unique_classes = set(y) | set(y_pred)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    pred_pair = list(zip(y, y_pred))

    for i, j in pred_pair:
        matrix[int(i), int(j)] += 1

    return matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]

def metrics(y, y_pred):
    n = len(y)

    tn, tp, fp, fn = get_confusion_matrix(y, y_pred)

    accuracy = (tp + tn) / n
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    met = {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'f1': f1}
    return met


def mpe(X, Y, Ypred, w):
    y_score = np.abs(Y - Ypred)
    raw_pred = linear_eqn(X, w)
    perc_err = np.abs(raw_pred) * y_score
    mpe = sum(perc_err) / len(Y)
    return mpe


def predict(X, w):
    y_pred = linear_eqn(X, w)

    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = 0

    return y_pred


def back_prop(X, Y, y_pred, alpha, weights):
    weights[0] += alpha * sum(Y - y_pred)

    for i in range(1, len(weights), 1):
        weights[i] += alpha * sum((Y - y_pred) * X[:, i - 1])

    return weights


def perceptron(data, alpha, epochs):
    w = [0] * data.shape[1]
    Y = data['outcome'].to_numpy()
    X = data.loc[:, data.columns != 'outcome'].to_numpy()
    progress = defaultdict(list)
    progress['epochs'] = list(range(epochs))

    for e in range(epochs):
        y_pred = predict(X, w)
        w = back_prop(X, Y, y_pred, alpha, w)
        met = metrics(Y, y_pred)
        progress['mpe'].append(mpe(X, Y, y_pred, w))
        progress['accuracy'].append(met['accuracy'])

    return w, progress


df = pd.read_csv('/Users/mihir/DS4400 - Spring ML 1/Homework 2/breast_cancer.csv')
df = df.drop(df.columns[0], axis=1)
df_columns = ['size_uni', 'shape_uni', 'marg_adh', 'size_adh', 'size_sec', 'bare_nuc', 'bland_chromo',
              'norm_chromo', 'mitosis', 'outcome']
df.set_axis(df_columns, axis=1, inplace=True)
df = df.replace('?', np.NaN)
df = df.dropna()
df = df.astype('int64')
df = df.reset_index()
del df[df.columns[0]]
df['outcome'] -= 2
df['outcome'] //= 2

weights, progress = perceptron(df, 0.0001, 1000)
X = df.loc[:, df.columns != 'outcome'].to_numpy()
ypred = predict(X, weights)
print("All 9 columns metrics: ", metrics(df['outcome'], ypred))

graph_line(progress['epochs'], progress['mpe'], 'Epochs', 'Mean Perceptron Error', "MPE Over Time")
graph_line(progress['epochs'], progress['accuracy'], 'Epochs', 'Accuracy', "Accuracy Over Time")


final_table, spec, sens = generate_table(df, 0.0001, 1000, 20)
print(final_table)
graph_line(spec, sens, 'Specificity', 'Sensitivity', "Specificity vs Sensitivity", scatterplot=True)


# If I had to choose a model from those that I have tested, I would choose model 10: ( 1, 1, 1, 0, 0, 1, 0, 0, 1,
#  Accuracy: 0.95307918, Specificity: 0.89539749 , Sensitivity: 0.98419865, Precision: 0.96832579, F1: 0.93043478),
# because it has the highest Sensitivity score and therefore we will be able to detect most of the people who have
# breast cancer, even if it means that we have a lot of people who we accidentally scare. The main problems with this
# model is its low specificity score, meaning that there may be a lot of people who test positive for breast cancer
# despite not having breast cancer, and there are models who have higher specificity and very good sensitivity (15:
# Specificity: ~98%, Sensitivity: ~96%, butI feel getting even a 2% increase in the number of True Positives may
# lead to thousands of lives saved.