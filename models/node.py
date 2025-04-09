#!git clone https://github.com/Qwicen/node.git

## Librerias
import math
import time
import pandas as pd
import numpy as np

from qhoptim.pyt import QHAdam
import torch, torch.nn as nn
import torch.nn.functional as F

# We access the NODE functionality through the /lib/ subfolder.
import lib

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report

# Métricas de valoración
def sensitivity(y_true,y_pred):
  res = classification_report(y_true, y_pred, output_dict=True)
  return res['1']['recall']

def specificity(y_true,y_pred):
  res =  classification_report(y_true, y_pred, output_dict=True)
  return res['0']['recall']

def add_matrix(matrixA, matrixB):
    result = [[0,0],[0,0]]
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            result[i][j] = matrixA[i][j] + matrixB[i][j]
    return result


# Utilizar CUDA si disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(f'Using {device}')


class Node:

    def __init__(self, learning_rate=0.001, layer_dim=128, num_layers=3, depth=6) -> None:
        self.learning_rate = learning_rate
        self.layer_dim = layer_dim
        self.num_layers = num_layers
        self.depth = depth
        self.batch_size = 32
        pass

    def fit(self, X, y, epochs, random_state=1234):
        x_train , x_test, y_train , y_test = train_test_split(X , y , random_state = random_state , test_size = 0.3, stratify=y)
        x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , random_state = random_state , test_size = 0.1, stratify=y_train)

        x_train = x_train.values.astype('float32')
        x_test = x_test.values.astype('float32')
        x_val = x_val.values.astype('float32')
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_val = np.array(y_val)

        # MODELO NODE
        # Limpiampos y definimos el modelo NODE
        num_features = x_train.shape[1]
        num_classes = len(set(y_train))
        ts = math.floor(time.time())
        experiment_name = f'node_adhd_{ts}'

        model = None
        model = nn.Sequential(
            lib.DenseBlock(num_features,
                        layer_dim=self.layer_dim,
                        num_layers=self.num_layers,
                        tree_dim=num_classes,
                        flatten_output=False,
                        depth=self.depth,
                        choice_function=lib.entmax15,
                        bin_function=lib.entmoid15),
            lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
        ).to(device)

        # Initialize a Trainer object
        trainer = None
        trainer = lib.Trainer(
            model=model, loss_function=F.cross_entropy,
            experiment_name=experiment_name,
            warm_start=False,
            Optimizer=QHAdam,
            optimizer_params=dict(lr=self.learning_rate, nus=(0.7, 1.0), betas=(0.95, 0.998)),
            verbose=False,
            n_last_checkpoints=5
        )

        # Training loop
        loss_history, err_history = [], []
        best_val_err = 1.0
        best_step = 0
        early_stopping_rounds = 10
        report_frequency = 5

        for batch in lib.iterate_minibatches(x_train,
                                            y_train,
                                            batch_size=self.batch_size, 
                                            shuffle=True,
                                            epochs=epochs):
            metrics = trainer.train_on_batch(*batch, device=device)
            
            loss_history.append(metrics['loss'])

            if trainer.step % report_frequency == 0:
                err = trainer.evaluate_classification_error(
                    x_val,
                    y_val,
                    device=device,
                    batch_size=self.batch_size)
                
                if err < best_val_err:
                    best_val_err = err
                    best_step = trainer.step
                    trainer.save_checkpoint(tag='best')
                
                err_history.append(err)
            
            # Early Stopper
            if trainer.step > best_step + early_stopping_rounds:
                trainer.load_checkpoint(tag='best')  # best
                break
        # Calcular recall
        with torch.no_grad():
            model.eval()
            test_preds = model(torch.tensor(x_test, dtype=torch.float32)).numpy()
            test_preds = np.argmax(test_preds, axis=1)
        #recall = np.round(sensitivity(y_test, test_preds), 6)
        roc_auc = np.round(roc_auc_score(y_test, test_preds), 4)
        return roc_auc
    
    def fit_evaluate(self, X, y, cv=10, epochs=50, random_state=None, verbose=True):

        ## KFOLD
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        lst_result = []
        confusion = [[0,0], [0,0]]

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Particion para validación en el fit
            x_train_fold , val_X , y_train_fold , val_y = train_test_split(x_train_fold , y_train_fold , random_state = random_state , test_size = 0.1, stratify=y_train_fold)

            # Types
            x_train_fold = x_train_fold.values.astype('float32')
            x_test_fold = x_test_fold.values.astype('float32')
            val_X = val_X.values.astype('float32')

            y_train_fold = np.array(y_train_fold)
            y_test_fold = np.array(y_test_fold)
            val_y = np.array(val_y)
            

            # MODELO NODE
            # Limpiampos y definimos el modelo NODE
            num_features = x_train_fold.shape[1]
            num_classes = len(set(y_train_fold))
            ts = math.floor(time.time())
            experiment_name = f'node_adhd_{ts}'

            model = None
            model = nn.Sequential(
                lib.DenseBlock(num_features,
                            layer_dim=self.layer_dim,
                            num_layers=self.num_layers,
                            tree_dim=num_classes,
                            flatten_output=False,
                            depth=self.depth,
                            choice_function=lib.entmax15,
                            bin_function=lib.entmoid15),
                lib.Lambda(lambda x: x[..., :num_classes].mean(dim=-2)),
            ).to(device)

            # Initialize a Trainer object
            trainer = None
            trainer = lib.Trainer(
                model=model, loss_function=F.cross_entropy,
                experiment_name=experiment_name,
                warm_start=False,
                Optimizer=QHAdam,
                optimizer_params=dict(lr=self.learning_rate, nus=(0.7, 1.0), betas=(0.95, 0.998)),
                verbose=False,
                n_last_checkpoints=5
            )

            # Training loop
            loss_history, err_history = [], []
            best_val_err = 1.0
            best_step = 0
            early_stopping_rounds = 10
            report_frequency = 5

            for batch in lib.iterate_minibatches(x_train_fold,
                                                y_train_fold,
                                                batch_size=self.batch_size, 
                                                shuffle=True,
                                                epochs=epochs):
                metrics = trainer.train_on_batch(*batch, device=device)
                
                loss_history.append(metrics['loss'])

                if trainer.step % report_frequency == 0:
                    #trainer.save_checkpoint()
                    #trainer.average_checkpoints(out_tag='avg')
                    #trainer.load_checkpoint(tag='avg')
                    err = trainer.evaluate_classification_error(
                        val_X,
                        val_y,
                        device=device,
                        batch_size=self.batch_size)
                    
                    if err < best_val_err:
                        best_val_err = err
                        best_step = trainer.step
                        trainer.save_checkpoint(tag='best')
                    
                    err_history.append(err)
                    #trainer.remove_old_temp_checkpoints()
                        
                    #print("Loss %.5f" % (metrics['loss']))
                    #print("Val Error Rate: %0.5f" % (err))
                
                # Early Stopper
                if trainer.step > best_step + early_stopping_rounds:
                    #print('BREAK. There is no improvement for {} steps'.format(early_stopping_rounds))
                    #print("Best step: ", best_step)
                    #print("Best Val Error Rate: %0.5f" % (best_val_err))
                    trainer.load_checkpoint(tag='best')  # best
                    break

            # Calcular roc en test
            with torch.no_grad():
                model.eval()
                test_preds = model(torch.tensor(x_test_fold, dtype=torch.float32)).numpy()
                test_preds = np.argmax(test_preds, axis=1)
                

            acc = np.round(accuracy_score(y_test_fold, test_preds), 4)
            roc_auc = np.round(roc_auc_score(y_test_fold, test_preds), 4)
            recall = np.round(sensitivity(y_test_fold, test_preds), 4)
            spec = np.round(specificity(y_test_fold, test_preds), 4)
            confusion_fold = np.round(confusion_matrix(y_test_fold, test_preds), 4)
            confusion = add_matrix(confusion, confusion_fold)

            if verbose:
                print(f'[NODE] Fold {fold+1}/{cv} - Completed')
            
            result = [acc, roc_auc, recall, spec]

            lst_result.append(result)

        ## FIN
        return lst_result, confusion

# My avg Results
# RESULT AVG TEST OF 10-CV KFOLD
# - accuracy = 0.79123
# - roc_auc = 0.76432
# - recall = 0.92384
# - specificity = 0.60482
# [[174, 114], [31, 375]]