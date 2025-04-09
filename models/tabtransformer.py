## Librerias

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

#new
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Métricas de valoración
def sensitivity(y_true,y_pred):
  res =  metrics.classification_report(y_true, y_pred, output_dict=True)
  return res['1']['recall']

def specificity(y_true,y_pred):
  res =  metrics.classification_report(y_true, y_pred, output_dict=True)
  return res['0']['recall']

def add_matrix(matrixA, matrixB):
    result = [[0,0],[0,0]]
    for i in range(len(matrixA)):
        for j in range(len(matrixA[0])):
            result[i][j] = matrixA[i][j] + matrixB[i][j]
    return result



# Column information
NUMERIC_FEATURES = ['BD', 'SI', 'DS', 'PCn', 'CD', 'VC', 'LN', 'MR', 'CO', 'SS', 'VCI', 'PRI', 'WMI', 'PSI', 'FSIQ', 'GAI', 'CPI', 'GAI-CPI', 'WMI-PSI', 'GAI-WMI', 'GAI-FSIQ', 'PRI-CPI', 'FSIQ-CPI', 'VCI-CPI', 'PRI-WMI']
CATEGORICAL_FEATURES = []
FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
LABEL = 'ADHD'


class TabTransformer:
    def __init__(self, learning_rate=0.001, weight_decay=0.0001, embedding_dim=32, depth=4, heads=8, dropout=0.2):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout

        self.optimizer = tfa.optimizers.AdamW( learning_rate=learning_rate, weight_decay=weight_decay )
        self.early = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
        return
    

    # For hyperopt
    def fit(self, X, y, epochs, random_state=1234):
        # Set data types
        X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)

        x_train , x_test , y_train , y_test = train_test_split(X , y , random_state = random_state , test_size = 0.3, stratify=y)
        x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , random_state = random_state , test_size = 0.1, stratify=y_train)
        #TF
        train_tmp = x_train
        train_tmp[LABEL] = y_train
        val_tmp = x_val
        val_tmp[LABEL] = y_val

        train_dataset = df_to_dataset(train_tmp, LABEL, batch_size=32)
        val_dataset = df_to_dataset(val_tmp, LABEL, shuffle=False, batch_size=32)
        test_dataset = df_to_dataset(x_test[FEATURES], None, shuffle=False, batch_size=32)

        # Class weight
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights))
        
        ## MODEL TABTRANSFORMER
        # Limpiampos y definimos el modelo TABTRANSFORMER
        ft_linear_encoder = FTTransformerEncoder(
            numerical_features = NUMERIC_FEATURES,
            categorical_features = CATEGORICAL_FEATURES,
            numerical_data = x_train[NUMERIC_FEATURES].values,
            categorical_data = x_train[CATEGORICAL_FEATURES].values,
            y = None,
            numerical_embedding_type='linear',
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            heads=self.heads,
            attn_dropout=self.dropout,
            ff_dropout=self.dropout,
            explainable=True
        )

        # Pass the encoder to the model
        ft_linear_transformer = FTTransformer(
            encoder=ft_linear_encoder,
            out_dim=1,
            out_activation='sigmoid',
        )


        ft_linear_transformer.compile(
            optimizer = self.optimizer,
            loss = {"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
            metrics= {"output": [tf.keras.metrics.AUC(name="PR AUC", curve='PR')], "importances": None},
        )

        # Train fold
        ft_linear_history = ft_linear_transformer.fit(
            train_dataset, 
            epochs=epochs, 
            validation_data=val_dataset,
            callbacks=self.early,
            verbose=0,
            class_weight=class_weights
        )

        # Calcular roc en test
        test_preds = np.round(ft_linear_transformer.predict(test_dataset, verbose=0)['output'].ravel(), 0)
        #recall = np.round(sensitivity(y_test, test_preds), 4)
        roc_auc = np.round(roc_auc_score(y_test, test_preds), 4)

        return roc_auc




    def fit_evaluate(self, X, y, cv=10, epochs=50, random_state=None, verbose=True):

        # Set data types
        X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(str)
        X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)

        ## KFOLD
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

        lst_result = []
        confusion = [[0,0], [0,0]]

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            # Particion para validación en el fit
            x_train_fold , val_X , y_train_fold , val_y = train_test_split(x_train_fold , y_train_fold , random_state = random_state , test_size = 0.1, stratify=y_train_fold)
            
            #TF
            train_tmp = x_train_fold
            train_tmp[LABEL] = y_train_fold
            val_tmp = val_X
            val_tmp[LABEL] = val_y

            train_dataset = df_to_dataset(train_tmp, LABEL, batch_size=32)
            val_dataset = df_to_dataset(val_tmp, LABEL, shuffle=False, batch_size=32)
            test_dataset = df_to_dataset(x_test_fold[FEATURES], None, shuffle=False, batch_size=32)

            # Class weight
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weights = dict(zip(np.unique(y_train_fold), class_weights))
            
            ## MODEL TABTRANSFORMER
            # Limpiampos y definimos el modelo TABTRANSFORMER
            ft_linear_encoder = None
            ft_linear_encoder = FTTransformerEncoder(
                numerical_features = NUMERIC_FEATURES,
                categorical_features = CATEGORICAL_FEATURES,
                numerical_data = x_train_fold[NUMERIC_FEATURES].values,
                categorical_data = x_train_fold[CATEGORICAL_FEATURES].values,
                y = None,
                numerical_embedding_type='linear',
                embedding_dim=self.embedding_dim,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.dropout,
                ff_dropout=self.dropout,
                explainable=True
            )

            # Pass the encoder to the model
            ft_linear_transformer = None
            ft_linear_transformer = FTTransformer(
                encoder=ft_linear_encoder,
                out_dim=1,
                out_activation='sigmoid',
            )


            ft_linear_transformer.compile(
                optimizer = self.optimizer,
                loss = {"output": tf.keras.losses.BinaryCrossentropy(), "importances": None},
                metrics= {"output": [tf.keras.metrics.AUC(name="PR AUC", curve='PR')], "importances": None},
            )

            
            callback_list = [self.early]

            # Train fold
            ft_linear_history = ft_linear_transformer.fit(
                train_dataset, 
                epochs=epochs, 
                validation_data=val_dataset,
                callbacks=callback_list,
                verbose=0,
                class_weight=class_weights
            )

            # Calcular roc en test
            test_preds = ft_linear_transformer.predict(test_dataset, verbose=0)['output']
            test_preds = np.round(test_preds.ravel(), 0)

            acc = np.round(accuracy_score(y_test_fold, test_preds), 4)
            roc_auc = np.round(roc_auc_score(y_test_fold, test_preds), 4)
            recall = np.round(sensitivity(y_test_fold, test_preds), 4)
            spec = np.round(specificity(y_test_fold, test_preds), 4)
            confusion_fold = np.round(confusion_matrix(y_test_fold, test_preds), 4)
            confusion = add_matrix(confusion, confusion_fold)

            if verbose:
                print(f'[TABTRANSFORMER] Fold {fold+1}/{cv} - Completed')
            
            result = [acc, roc_auc, recall, spec]

            lst_result.append(result)

        ## FIN
        return lst_result, confusion

# My avg Results
# RESULT AVG TEST OF 10-CV KFOLD
# - accuracy = 0.86616
# - roc_auc = 0.91409
# - recall = 0.85995
# - specificity = 0.8755
# [[252, 36], [57, 349]]
