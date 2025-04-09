## Librerias
import math
import time
import pandas as pd
import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split
from statistics import mean

from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


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




# device in which the model will be trained
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DCNN:
    def __init__(self, sign_size=16, cha_input=64, cha_hidden=64, K=2, drop_input=0.3, drop_hidden=0.3, drop_out=0.2) -> None:
        self.sign_size=sign_size
        self.cha_input=cha_input
        self.cha_hidden=cha_hidden
        self.K=K
        self.dropout_input=drop_input
        self.dropout_hidden=drop_hidden
        self.dropout_output=drop_out
        self.batch_size = 32


        # Earlystopper
        self.early_stop_callback = EarlyStopping(
            monitor='valid_loss',
            min_delta=.0,
            patience=10,
            verbose=True,
            mode='min'
        )

    # For hyperopt
    def fit(self, X, y, epochs, random_state=1234):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True

        input_features = X.columns.tolist()

        x_train , x_test , y_train , y_test = train_test_split(X , y , random_state = random_state , test_size = 0.3, stratify=y)
        x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , random_state = random_state , test_size = 0.1, stratify=y_train)

        # Types
        train_dataset = TensorDataset( torch.tensor(x_train.values, dtype=torch.float), torch.tensor(y_train.values.reshape(-1,1), dtype=torch.float  ))
        val_dataset = TensorDataset( torch.tensor(x_val.values, dtype=torch.float), torch.tensor(y_val.values.reshape(-1,1), dtype=torch.float  ))
        test_dataset = TensorDataset( torch.tensor(x_test.values, dtype=torch.float), torch.tensor(y_test.values.reshape(-1,1), dtype=torch.float  ))
        

        # MODELO SoftOrdering1DCNN
        # Limpiampos y definimos el modelo
        model = SoftOrdering1DCNN(
            input_dim=len(input_features), 
            output_dim=1,
            sign_size=self.sign_size, 
            cha_input=self.cha_input, 
            cha_hidden=self.cha_hidden, 
            K=self.K, 
            dropout_input=self.dropout_input, 
            dropout_hidden=self.dropout_hidden, 
            dropout_output=self.dropout_output
        )
        
        # Initialize a Trainer object
        trainer = pl.Trainer(callbacks=[self.early_stop_callback], min_epochs=10, max_epochs=epochs, enable_model_summary=False, enable_progress_bar=False)

        # Train
        trainer.fit(
            model, 
            DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0),
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        )

        # Calcular roc en test
        with torch.no_grad():
            model.eval()
            test_preds = model(torch.tensor(x_test.values, dtype=torch.float))
            test_preds = model(torch.tensor(x_test.values, dtype=torch.float)).numpy()
            #print(np.round(np.clip(test_preds, 0, 1), 0) )
            test_preds = np.round(np.clip(test_preds, 0, 1), 0).flatten().astype(int)

        #recall = np.round(sensitivity(y_test.values, test_preds), 4)
        roc_auc = np.round(roc_auc_score(y_test.values, test_preds), 4)
        return roc_auc


    def fit_evaluate(self, X, y, cv=10, epochs=50, random_state=None, verbose=True):
        # needed for deterministic output
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True


        input_features = X.columns.tolist()

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
            train_dataset = TensorDataset( torch.tensor(x_train_fold.values, dtype=torch.float), torch.tensor(y_train_fold.values.reshape(-1,1), dtype=torch.float  ))
            val_dataset = TensorDataset( torch.tensor(val_X.values, dtype=torch.float), torch.tensor(val_y.values.reshape(-1,1), dtype=torch.float  ))
            test_dataset = TensorDataset( torch.tensor(x_test_fold.values, dtype=torch.float), torch.tensor(y_test_fold.values.reshape(-1,1), dtype=torch.float  ))
            

            # MODELO SoftOrdering1DCNN
            # Limpiampos y definimos el modelo
            ts = math.floor(time.time())

            model = None
            model = SoftOrdering1DCNN(
                input_dim=len(input_features), 
                output_dim=1,
                sign_size=self.sign_size, 
                cha_input=self.cha_input, 
                cha_hidden=self.cha_hidden, 
                K=self.K, 
                dropout_input=self.dropout_input, 
                dropout_hidden=self.dropout_hidden, 
                dropout_output=self.dropout_output
            )

            
            # Initialize a Trainer object
            trainer = None
            trainer = pl.Trainer(callbacks=[self.early_stop_callback], min_epochs=10, max_epochs=epochs, enable_model_summary=False, enable_progress_bar=False)

            # Train
            trainer.fit(
                model, 
                DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0),
                DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            )

            

            # Calcular roc en test
            with torch.no_grad():
                model.eval()
                test_preds = model(torch.tensor(x_test_fold.values, dtype=torch.float))
                test_preds = model(torch.tensor(x_test_fold.values, dtype=torch.float)).numpy()
                #print(np.round(np.clip(test_preds, 0, 1), 0) )
                test_preds = np.round(np.clip(test_preds, 0, 1), 0).flatten().astype(int)
                

            acc = np.round(accuracy_score(y_test_fold.values, test_preds), 4)
            roc_auc = np.round(roc_auc_score(y_test_fold.values, test_preds), 4)
            recall = np.round(sensitivity(y_test_fold.values, test_preds), 4)
            spec = np.round(specificity(y_test_fold.values, test_preds), 4)
            confusion_fold = np.round(confusion_matrix(y_test_fold.values, test_preds), 4)
            confusion = add_matrix(confusion, confusion_fold)

            if verbose:
                print(f'[1DCNN] Fold {fold+1}/{cv} - Completed')
            
            result = [acc, roc_auc, recall, spec]

            lst_result.append(result)

        ## FIN
        return lst_result, confusion


# Modelo
class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size*cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size//2
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input, 
            cha_input*K, 
            kernel_size=5, 
            stride = 1, 
            padding=2,  
            groups=cha_input, 
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        y_probs = torch.sigmoid(y_logit).detach().cpu().numpy()
        loss = self.loss(y_logit, y)
        metric = roc_auc_score(y.cpu().numpy(), y_probs)
        self.log('test_loss', loss)
        self.log('test_metric', metric)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]




# My avg Results
# RESULT AVG TEST OF 10-CV KFOLD
# - accuracy = 0.96113
# - roc_auc = 0.96471
# - recall = 0.94346
# - specificity = 0.98596
# [[284, 4], [23, 383]]