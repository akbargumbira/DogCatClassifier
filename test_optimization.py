# coding=utf-8
import os
import sys
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from preprocess import load_dataset


cb='sift_250'
dataset_path = os.path.join(os.curdir, 'model/%s/training_data.dat' % cb)
data, label = load_dataset(dataset_path)
X, X_val, y, y_val = train_test_split(data, label, test_size=0.3,
                                      random_state=0)
X, X_val, y, y_val = np.array(X), np.array(X_val), np.array(y), np.array(y_val)


space = {
    'choice': hp.choice('num_layers',
                        [
                            {'layers': 'two'},
                            {'layers': 'three',
                             'units3': hp.uniform('units3', 64, 1024),
                             'dropout3': hp.uniform('dropout3', .25,.75)}
                        ]),
    'units1': hp.uniform('units1', 64, 1024),
    'units2': hp.uniform('units2', 64, 1024),
    'dropout1': hp.uniform('dropout1', .25, .75),
    'dropout2': hp.uniform('dropout2',  .25, .75),
    'batch_size': hp.uniform('batch_size', 28, 128),
    'nb_epochs': 100,
    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
    'activation': 'relu'
}


def f_nn(params):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=int(params['units1']), input_dim = X.shape[1]))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=int(params['units2']), init = "glorot_uniform"))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers'] == 'three':
        model.add(Dense(output_dim=int(params['choice']['units3']), init="glorot_uniform"))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=params['optimizer'])

    model.fit(
        X,
        y,
        nb_epoch=int(params['nb_epochs']),
        batch_size=int(params['batch_size']),
        verbose=1
    )

    pred_auc = model.predict_proba(X_val, batch_size=128, verbose=0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush()
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
