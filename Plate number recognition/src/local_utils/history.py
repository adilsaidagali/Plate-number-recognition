import os

from pandas import DataFrame, read_csv
from tensorflow.keras.callbacks import History

def history2csv(history: History, csv_path):
    print('Saving history to CSV file')

    columns = ['loss', 'val_loss', 'accuracy', 'val_accuracy']

    if os.path.exists(csv_path):
        df_main = read_csv(csv_path, index_col=False, names=columns, header=0)
    else:
        df_main = DataFrame([], columns=columns)

    values = []

    for epoch in history.epoch:
        values.append([
            history.history['loss'][epoch],
            history.history['val_loss'][epoch],
            history.history['accuracy'][epoch],
            history.history['val_accuracy'][epoch],
        ])

    df = DataFrame(values, columns=columns)

    df_main = df_main.append(df, ignore_index=True)
    df_main.to_csv(csv_path, index=False)

    print('\tDone')


def csv2history(csv_path):
    print('Reading history from CSV file')

    df = read_csv(csv_path, index_col=False)

    res = dict()
    res['epochs'] = len(df.loss)
    res['loss'] = df.loss.to_list()
    res['val_loss'] = df.val_loss.to_list()
    res['accuracy'] = df.accuracy.to_list()
    res['val_accuracy'] = df.val_accuracy.to_list()

    print('\tDone')

    return res
