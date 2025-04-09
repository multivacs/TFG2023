## Librerias
import math
import time
import datetime
import pandas as pd
import numpy as np

# Hyperopt
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss

import statsmodels.api as sm
import statsmodels.stats as sms

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

import sys

# Modelos
from .models.tabnet import TabNet
from .models.tabtransformer import TabTransformer
from .models.node import Node
from .models.DCNN import DCNN

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler


# Parameters
FEATURE_ENGINEERING = False
HYPEROPT = False
EVALUATE = True
PREPROCESS = "Standard" # Standard/Robust
NUM_FOLDS = 10
NUM_EPOCHS = 100
SEED = 1234

# Hiperparámetros
best_tabnet = {'bn_momentum': 0.942772441153329, 'feature_dim': 128, 'learning_rate': 0.03737086735655166, 'n_steps': 5, 'relaxation_factor': 1.2000000000000002, 'sparsity': 0.0595285322427291}
best_tabtransformer = {'depth': 4, 'dropout': 0.28774402657808856, 'embedding_dim': 64, 'heads': 8, 'learning_rate': 0.0015283238179232778, 'weight_decay': 0.07992657295500534}
best_node = {'depth': 2, 'layer_dim': 384, 'learning_rate': 0.0012095595812000894, 'num_layers': 6}
best_1dcnn = {'K': 8, 'cha_hidden': 128, 'cha_input': 192, 'drop_hidden': 0.11412807906220623, 'drop_input': 0.250317695794937, 'drop_out': 0.23413921322423242, 'sign_size': 128}



################ Data ###############
# Column information
NUMERIC_FEATURES = ['BD', 'SI', 'DS', 'PCn', 'CD', 'VC', 'LN', 'MR', 'CO', 'SS', 'VCI', 'PRI', 'WMI', 'PSI', 'FSIQ', 'GAI', 'CPI', 'GAI-CPI', 'WMI-PSI', 'GAI-WMI', 'GAI-FSIQ', 'PRI-CPI', 'FSIQ-CPI', 'VCI-CPI', 'PRI-WMI']
CATEGORICAL_FEATURES = []
FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
LABEL = 'ADHD'

# Uso de dataset sintético
# Definir la cantidad de ejemplos
n_samples = 1000

# Generar los datos sintéticos para cada columna
data = {
    'BD': np.random.randint(4, 11, n_samples),
    'SI': np.random.randint(7, 16, n_samples),
    'DS': np.random.randint(3, 16, n_samples),
    'PCn': np.random.randint(5, 16, n_samples),
    'CD': np.random.randint(4, 15, n_samples),
    'VC': np.random.randint(2, 15, n_samples),
    'LN': np.random.randint(4, 15, n_samples),
    'MR': np.random.randint(5, 15, n_samples),
    'CO': np.random.randint(6, 16, n_samples),
    'SS': np.random.randint(4, 12, n_samples),
    'VCI': np.random.randint(85, 120, n_samples),
    'PRI': np.random.randint(65, 115, n_samples),
    'WMI': np.random.randint(60, 110, n_samples),
    'PSI': np.random.randint(70, 120, n_samples),
    'FSIQ': np.random.randint(85, 130, n_samples),
    'GAI': np.random.randint(80, 125, n_samples),
    'CPI': np.random.randint(80, 120, n_samples),
    'GAI-CPI': np.random.randint(-30, 30, n_samples),
    'WMI-PSI': np.random.randint(-20, 20, n_samples),
    'GAI-WMI': np.random.randint(-10, 20, n_samples),
    'GAI-FSIQ': np.random.randint(-30, 30, n_samples),
    'PRI-CPI': np.random.randint(-20, 20, n_samples),
    'FSIQ-CPI': np.random.randint(-25, 25, n_samples),
    'VCI-CPI': np.random.randint(-20, 20, n_samples),
    'PRI-WMI': np.random.randint(-15, 15, n_samples),
    'ADHD': np.random.randint(0, 2, n_samples)  # 0 = No TDAH, 1 = TDAH
}

# Crear un DataFrame
df = pd.DataFrame(data)
#df = pd.read_csv('./data/dataset.csv', sep=';', decimal=',')

# Eliminar duplicados
df.drop_duplicates()


################ FEATURE ENGINEERING ###############

if FEATURE_ENGINEERING:
  
  # Diferencias entre grupos clínico y control (medias)
  g_clinic = df.loc[df['ADHD'] == 1]
  g_control = df.loc[df['ADHD'] == 0]
  x = ['FSIQ', 'VCI', 'PRI', 'WMI', 'PSI', 'GAI', 'CPI']

  g_clinic = g_clinic[x].mean(axis=0).to_numpy()
  g_control = g_control[x].mean(axis=0).to_numpy()

  plt.plot(x, g_clinic, label='Grupo clínico')
  plt.plot(x, g_control, label='Grupo control')
  plt.legend()
  plt.show()

  # Matriz de correlacion
  cormat = df.corr()
  print(cormat)
  sns.heatmap(cormat, annot=True)


################ Data Preprocessing ###############

# Separar target
X = df.drop(['ADHD'],axis =1)
y = df['ADHD'].astype('int64')

# Standard Scaler
if PREPROCESS == "Standard":
  scaler = StandardScaler()
  X.loc[:, NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])

# Robust Scaler
elif PREPROCESS == "Robust":
  scaler = RobustScaler(quantile_range=(20, 80))
  X.loc[:, NUMERIC_FEATURES] = scaler.fit_transform(X[NUMERIC_FEATURES])


################ Hyperparameter Tuning ###############
if HYPEROPT == True:
  best_tabnet = {}
  best_tabtransformer = {}
  best_node = {}
  best_1dcnn = {}

  HO_EPOCHS = 100
  HO_MAXEVALS = 60
  ho_time = time.time()

  # Espacio de búsqueda
  TabNet()
  space_tabnet = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'feature_dim': hp.quniform('feature_dim', 64, 256, 64),
    'n_steps': hp.quniform('n_steps', 2, 6, 1),
    'relaxation_factor': hp.quniform('relaxation_factor', 1, 3, q=0.1),
    'sparsity': hp.uniform('sparsity', 0.001, 0.1),
    'bn_momentum': hp.uniform('bn_momentum', 0.9, 0.9999)
  }

  space_tabtransformer = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'weight_decay': hp.uniform('weight_decay', 0.0001, 0.1),
    'embedding_dim': hp.quniform('embedding_dim', 16, 64, 16),
    'depth': hp.quniform('depth', 2, 8, 1),
    'heads': hp.quniform('heads', 4, 16, 1),
    'dropout': hp.uniform('dropout', 0.1, 0.3)
  }

  space_node = {
    'learning_rate': hp.loguniform('learning_rate', -7, 0),
    'layer_dim': hp.quniform('layer_dim', 128, 512, 128),
    'num_layers': hp.quniform('num_layers', 2, 6, 1),
    'depth': hp.quniform('depth', 2, 6, 2)
  }

  space_1dcnn = {
    'sign_size': hp.quniform('sign_size', 16, 128, 16),
    'cha_input': hp.quniform('cha_input', 64, 256, 64),
    'cha_hidden': hp.quniform('cha_hidden', 32, 128, 32),
    'K': hp.quniform('K', 2, 8, 1),
    'drop_input': hp.uniform('drop_input', 0.1, 0.3),
    'drop_hidden': hp.uniform('drop_hidden', 0.1, 0.3),
    'drop_out': hp.uniform('drop_out', 0.1, 0.3)
  }


  # Funciones objetivo
  def objective_tabnet(space):
    model = TabNet(learning_rate=space['learning_rate'],
                   feature_dim=int(space['feature_dim']),
                   n_steps=int(space['n_steps']),
                   relaxation_factor=space['relaxation_factor'],
                   sparsity=space['sparsity'],
                   bn_momentum=space['bn_momentum'])
    
    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    # Negativo para maximizar el valor, positivo para minimización
    return{'loss': -roc, "status":STATUS_OK}

  def objective_tabtransformer(space):
    model = TabTransformer(learning_rate=space['learning_rate'],
                           weight_decay=space['weight_decay'],
                           embedding_dim=int(space['embedding_dim']),
                           depth=int(space['depth']),
                           heads=int(space['heads']),
                           dropout=space['dropout'])

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    # Negativo para maximizar el valor, positivo para minimización
    return{'loss': -roc, "status":STATUS_OK}

  def objective_node(space):
    model = Node(learning_rate=space['learning_rate'],
                 layer_dim=int(space['layer_dim']),
                 num_layers=int(space['num_layers']),
                 depth=int(space['depth']))

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    # Negativo para maximizar el valor, positivo para minimización
    return{'loss': -roc, "status":STATUS_OK}

  def objective_1dcnn(space):
    model = DCNN(sign_size=int(space['sign_size']),
                 cha_input=int(space['cha_input']),
                 cha_hidden=int(space['cha_hidden']),
                 K=int(space['K']),
                 drop_input=space['drop_input'],
                 drop_hidden=space['drop_hidden'],
                 drop_out=space['drop_out'])

    roc = model.fit(X, y, epochs=HO_EPOCHS, random_state=SEED)

    # Negativo para maximizar el valor, positivo para minimización
    return{'loss': -roc, "status":STATUS_OK}


  # Búsqueda
  trials = Trials()
  time_tabnet = time.time()
  best_tabnet = fmin( fn = objective_tabnet, space=space_tabnet, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(10))
  time_tabnet = str(datetime.timedelta(seconds=time.time() - time_tabnet))
  loss_tabnet=trials.best_trial['result']['loss']
  print(f"TABNET HYPEROPT = {best_tabnet}")

  trials = Trials()
  time_tabtransformer = time.time()
  best_tabtransformer = fmin( fn = objective_tabtransformer, space=space_tabtransformer, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(10))
  time_tabtransformer = str(datetime.timedelta(seconds=time.time() - time_tabtransformer))
  loss_tabtransformer=trials.best_trial['result']['loss']
  print(f"TABTRANSFORMER HYPEROPT = {best_tabtransformer}")

  trials = Trials()
  time_node = time.time()
  best_node = fmin( fn = objective_node, space=space_node, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(10))
  time_node = str(datetime.timedelta(seconds=time.time() - time_node))
  loss_node=trials.best_trial['result']['loss']
  print(f"NODE HYPEROPT = {best_node}")

  trials = Trials()
  time_1dcnn = time.time()
  best_1dcnn = fmin( fn = objective_1dcnn, space=space_1dcnn, algo= tpe.suggest, max_evals=HO_MAXEVALS, trials=trials, early_stop_fn=no_progress_loss(10))
  time_1dcnn = str(datetime.timedelta(seconds=time.time() - time_1dcnn))
  loss_1dcnn=trials.best_trial['result']['loss']
  print(f"1DCNN HYPEROPT = {best_1dcnn}")


  # Save results
  ho_time = time.time() - ho_time
  ho_time = str(datetime.timedelta(seconds=ho_time))
  print(f'Tiempo ejecución HyperOpt: {ho_time}')

  now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
  ts = math.floor(time.time())
  script_dir = os.path.dirname(__file__)
  rel_path = f'hyperopt\\HO_{ts}.txt'
  filepath = os.path.join(script_dir, rel_path)

  f = open(filepath, "w")
  output = f'# {now}\n# Ajuste de hiperparametros con: [Tiempo={ho_time}, Epochs={HO_EPOCHS}, Max evals={HO_MAXEVALS}]\n\n'
  output += f'TABNET HYPEROPT = {best_tabnet}\ntime={time_tabnet} ; best_loss={loss_tabnet}\n\n'
  output += f'TABTRANSFORMER HYPEROPT = {best_tabtransformer}\ntime={time_tabtransformer} ; best_loss={loss_tabtransformer}\n\n'
  output += f'NODE HYPEROPT = {best_node}\ntime={time_node} ; best_loss={loss_node}\n\n'
  output += f'1DCNN HYPEROPT = {best_1dcnn}\ntime={time_1dcnn} ; best_loss={loss_1dcnn}'
  f.write( output )
  f.close()


################ KFOLD ###############
if EVALUATE == False:
  sys.exit()

# Create models
models = [TabNet(learning_rate=best_tabnet['learning_rate'],
                 feature_dim=int(best_tabnet['feature_dim']),
                 n_steps=int(best_tabnet['n_steps']),
                 relaxation_factor=best_tabnet['relaxation_factor'],
                 sparsity=best_tabnet['sparsity'],
                 bn_momentum=best_tabnet['bn_momentum']), 
          TabTransformer(learning_rate=best_tabtransformer['learning_rate'],
                         weight_decay=best_tabtransformer['weight_decay'],
                         embedding_dim=int(best_tabtransformer['embedding_dim']),
                         depth=int(best_tabtransformer['depth']),
                         heads=int(best_tabtransformer['heads']),
                         dropout=best_tabtransformer['dropout']), 
          Node(learning_rate=best_node['learning_rate'],
               layer_dim=int(best_node['layer_dim']),
               num_layers=int(best_node['num_layers']),
               depth=int(best_node['depth'])), 
          DCNN(sign_size=int(best_1dcnn['sign_size']),
               cha_input=int(best_1dcnn['cha_input']),
               cha_hidden=int(best_1dcnn['cha_hidden']),
               K=int(best_1dcnn['K']),
               drop_input=best_1dcnn['drop_input'],
               drop_hidden=best_1dcnn['drop_hidden'],
               drop_out=best_1dcnn['drop_out'])
          ]
names = ["TabNet", "TabTransformer", "Node", "1DCNN"]



all_results = []
for model, name in zip(models, names):
  t_start = time.time()
  result, confusion = model.fit_evaluate(X, y, cv=NUM_FOLDS, epochs=NUM_EPOCHS, random_state = SEED)
  t_end = time.time() - t_start
  thisdict = dict(model_name = name, score = result, confmatrix = confusion, avg_score = np.average(result, axis=0), execution_time_sec = t_end)
  all_results.append(thisdict)


# Print results
df_result = pd.DataFrame.from_dict(all_results)
df_result['accuracy'] = list(zip(*df_result['avg_score'].values))[0]
df_result['auc_roc'] = list(zip(*df_result['avg_score'].values))[1]
df_result['recall'] = list(zip(*df_result['avg_score'].values))[2]
df_result['specificity'] = list(zip(*df_result['avg_score'].values))[3]
df_result = df_result.drop(columns=['score'])
df_result = df_result.drop(columns=['avg_score'])
df_result = df_result.drop(columns=['confmatrix'])


print(df_result)


# Save results
now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
ts = math.floor(time.time())
script_dir = os.path.dirname(__file__)
rel_path = f'results\\{ts}.txt'
filepath = os.path.join(script_dir, rel_path)

f = open(filepath, "w")
f.write( f'# {now}\n# Results for: [Hyperparameters: True, Preprocess_scaler: {PREPROCESS}, Folds: {NUM_FOLDS}, Epochs: {NUM_EPOCHS}, Seed: {SEED}]\n\n' +
        str(all_results[0]) + '\n' + str(all_results[1]) + '\n' + str(all_results[2]) + '\n' + str(all_results[3]))
f.close()
