## Librerias

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from statistics import mean

from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from tensorflow_addons.activations import sparsemax

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


class TabNet:
    def __init__(self, learning_rate=0.001, feature_dim = 128, n_steps = 2, relaxation_factor = 2.2, sparsity=2.37e-07, bn_momentum = 0.9245):

        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.n_steps = n_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity = sparsity
        self.bn_momentum = bn_momentum

        # Early stopping based on validation loss    
        self.cbs = tf.keras.callbacks.EarlyStopping( monitor="val_loss", patience=10, restore_best_weights=True )

        # Optimiser 
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=10)

        # Second loss in None because we also output the importances
        #self.loss = [tf.keras.losses.CategoricalCrossentropy(from_logits=False), None]
        self.loss = [tf.keras.losses.BinaryCrossentropy(from_logits=False), None]


    # Transformar dataset en objetos TF, con el objetivo de acelerar el entrenamiento
    def prepare_tf_dataset(
        self,
        X,
        batch_size,
        y = None,
        shuffle = False,
        drop_remainder = False,
    ):
        size_of_dataset = len(X)
        if y is not None:
            y = tf.one_hot(y.astype(int), 2)
            ds = tf.data.Dataset.from_tensor_slices((np.array(X.astype(np.float32)), y))
        else:
            ds = tf.data.Dataset.from_tensor_slices(np.array(X.astype(np.float32)))
        if shuffle:
            ds = ds.shuffle(buffer_size=size_of_dataset)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

        autotune = tf.data.experimental.AUTOTUNE
        ds = ds.prefetch(autotune)
        return ds
    
    # For hyperopt
    def fit(self, X, y, epochs, random_state=1234):
        x_train , x_test , y_train , y_test = train_test_split(X , y , random_state = random_state , test_size = 0.3, stratify=y)
        x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , random_state = random_state , test_size = 0.1, stratify=y_train)

        #TF
        train_ds = self.prepare_tf_dataset(X=x_train, batch_size=32, y=y_train)
        test_ds = self.prepare_tf_dataset(X=x_test, batch_size=32, y=y_test)
        val_ds = self.prepare_tf_dataset(X=x_val, batch_size=32, y=y_val)

        # Class weight
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(zip(np.unique(y_train), class_weights))
        
        # Limpiamos el modelo
        tabnet = TabNetBlock(num_features = X.shape[1],
                    output_dim = self.feature_dim,
                    feature_dim = self.feature_dim,
                    n_step = self.n_steps, 
                    relaxation_factor= self.relaxation_factor,
                    sparsity_coefficient=self.sparsity,
                    n_shared = 2,
                    bn_momentum = self.bn_momentum)
        tabnet.compile(self.optimizer, loss=self.loss)

        # Train fold
        tabnet.fit(train_ds, 
            epochs=epochs, 
            validation_data=val_ds,
            callbacks=self.cbs,
            verbose=0,
            class_weight=class_weights)
        
        # Calcular roc en test
        test_preds, test_imps = tabnet.predict(test_ds, verbose=0) # WARNING tensorflow:5 out of the last 5 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000021A68F1C550> triggered tf.function retracing
        #recall = np.round(sensitivity(y_test, np.round(test_preds[:, 1], 0)), 4)
        roc_auc = np.round(roc_auc_score(y_test, np.round(test_preds[:, 1], 0)), 4)
        return roc_auc


    # Fit
    def fit_evaluate(self, X, y, cv=10, epochs=50, random_state=None, verbose=True):

        features = X.shape[1]

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
            train_ds = self.prepare_tf_dataset(X=x_train_fold, batch_size=32, y=y_train_fold)
            test_ds = self.prepare_tf_dataset(X=x_test_fold, batch_size=32, y=y_test_fold)
            val_ds = self.prepare_tf_dataset(X=val_X, batch_size=32, y=val_y)

            # Class weight
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_fold), y=y_train_fold)
            class_weights = dict(zip(np.unique(y_train_fold), class_weights))
            
            # Limpiamos el modelo
            tabnet = None
            tabnet = TabNetBlock(num_features = features,
                        output_dim = self.feature_dim,
                        feature_dim = self.feature_dim,
                        n_step = self.n_steps, 
                        relaxation_factor= self.relaxation_factor,
                        sparsity_coefficient=self.sparsity,
                        n_shared = 2,
                        bn_momentum = self.bn_momentum)
            tabnet.compile(self.optimizer, loss=self.loss)

            # Train fold
            tabnet.fit(train_ds, 
                epochs=epochs, 
                validation_data=val_ds,
                callbacks=self.cbs,
                verbose=0,
                class_weight=class_weights)

            # Calcular roc en test
            test_preds, test_imps = tabnet.predict(test_ds, verbose=0) # WARNING tensorflow:5 out of the last 5 calls to <function Model.make_predict_function.<locals>.predict_function at 0x0000021A68F1C550> triggered tf.function retracing


            acc = np.round(accuracy_score(y_test_fold, np.round(test_preds[:, 1], 0)), 4)
            roc_auc = np.round(roc_auc_score(y_test_fold, np.round(test_preds[:, 1], 0)), 4)
            recall = np.round(sensitivity(y_test_fold, np.round(test_preds[:, 1], 0)), 4)
            spec = np.round(specificity(y_test_fold, np.round(test_preds[:, 1], 0)), 4)
            confusion_fold = np.round(confusion_matrix(y_test_fold, np.round(test_preds[:, 1], 0)), 4)
            confusion = add_matrix(confusion, confusion_fold)

            if verbose:
                print(f'[TABNET] Fold {fold+1}/{cv} - Completed')
            
            result = [acc, roc_auc, recall, spec]

            lst_result.append(result)

        # Devolvemos los resultados de los fold y la matriz de confusion
        return lst_result, confusion




## MODEL TABNET
# Definimos el modelo TABNET

# GLU activation layer: permite propagar los gradientes más profundo
def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])

# Bloque con una capa Densa, un Batch Normalisation y una capa GLU
class FeatureBlock(tf.keras.Model):
    """
    Implementation of a FL->BN->GLU block
    """
    def __init__(
        self,
        feature_dim,
        apply_glu = True,
        bn_momentum = 0.9,
        fc = None,
        epsilon = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_gpu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim # desired dimension gets multiplied by 2
                                                              # because GLU activation halves it

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc # shared layers can get re-used
        self.bn = tf.keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=epsilon)

    def call(self, x, training = None):
        x = self.fc(x) # inputs passes through the FC layer
        x = self.bn(x, training=training) # FC layer output gets passed through the BN
        if self.apply_gpu: 
            return glu(x, self.feature_dim) # GLU activation applied to BN output
        return x

# FeatureBlocks en serie que recoge los feature seleccionados en el Attentive, y los transforma para que el modelo los utilice
# (en el paper original, utilizan 4 bloques, 2 con pesos compartidos y 2 independientes)
class FeatureTransformer(tf.keras.Model):
    def __init__(
        self,
        feature_dim,
        fcs = [],
        n_total = 4,
        n_shared = 2,
        bn_momentum = 0.9,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kwrgs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
        }

        # build blocks
        self.blocks = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kwrgs, fc=fcs[n])) # Building shared blocks by providing FC layers
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kwrgs)) # Step dependent blocks without the shared FC layers

    def call(self, x, training = None):
        # input passes through the first block
        x = self.blocks[0](x, training=training) 
        # for the remaining blocks
        for n in range(1, self.n_total):
            # output from previous block gets multiplied by sqrt(0.5) and output of this block gets added
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training) 
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]

# Se encarga de la selección de las feature características teniendo en cuenta la selección de los bloques anteriores
class AttentiveTransformer(tf.keras.Model):
    def __init__(self, feature_dim):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            apply_glu=False,
        )

    def call(self, x, prior_scales, training=None):
        x = self.block(x, training=training)
        return sparsemax(x * prior_scales)

# AttentiveTransformer > FeatureBlock > AttentiveTransformer > ... > Outputs de cada step se pasan a una capa Densa para la regresión
class TabNetBlock(tf.keras.Model):
    def __init__(
        self,
        num_features,
        feature_dim,
        output_dim,
        n_step = 2,
        n_total = 4,
        n_shared = 2,
        relaxation_factor = 1.5,
        bn_epsilon = 1e-5,
        bn_momentum = 0.7,
        sparsity_coefficient = 1e-5
    ):
        super(TabNetBlock, self).__init__()
        self.output_dim, self.num_features = output_dim, num_features
        self.n_step, self.relaxation_factor = n_step, relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

        self.bn = tf.keras.layers.BatchNormalization(
            momentum=bn_momentum, epsilon=bn_epsilon
        )

        kargs = {
            "feature_dim": feature_dim + output_dim,
            "n_total": n_total,
            "n_shared": n_shared,
            "bn_momentum": bn_momentum
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms = [FeatureTransformer(**kargs)]
        self.attentive_transforms = []
            
        # each step consists out of FT and AT
        for i in range(n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(num_features)
            )
        
        # Final output layer
        self.head = tf.keras.layers.Dense(2, activation="softmax", use_bias=False)

    def call(self, features, training = None):

        bs = tf.shape(features)[0] # get batch shape
        out_agg = tf.zeros((bs, self.output_dim)) # empty array with outputs to fill
        prior_scales = tf.ones((bs, self.num_features)) # prior scales initialised as 1s
        importance = tf.zeros([bs, self.num_features]) # importances
        masks = []

        features = self.bn(features, training=training) # Batch Normalisation
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.n_step + 1):
            # (masked) features go through the FT
            x = self.feature_transforms[step_i](
                masked_features, training=training
            )
            
            # first FT is not used to generate output
            if step_i > 0:
                # first half of the FT output goes towards the decision 
                out = tf.keras.activations.relu(x[:, : self.output_dim])
                out_agg += out
                scale_agg = tf.reduce_sum(out, axis=1, keepdims=True) / (self.n_step - 1)
                importance += mask_values * scale_agg
                

            # no need to build the features mask for the last step
            if step_i < self.n_step:
                # second half of the FT output goes as input to the AT
                x_for_mask = x[:, self.output_dim :]
                
                # apply AT with prior scales
                mask_values = self.attentive_transforms[step_i](
                    x_for_mask, prior_scales, training=training
                )

                # recalculate the prior scales
                prior_scales *= self.relaxation_factor - mask_values
                
                # multiply the second half of the FT output by the attention mask to enforce sparsity
                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(-mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )
                
                # append mask values for later explainability
                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))
                
        #Per step selection masks        
        self.selection_masks = masks
        
        # Final output
        final_output = self.head(out)
        
        # Add sparsity loss
        loss = total_entropy / (self.n_step-1)
        self.add_loss(self.sparsity_coefficient * loss)
        
        return final_output, importance








# My avg Results
# RESULT AVG TEST OF 10-CV KFOLD
# - accuracy = 0.8216600000000001
# - roc_auc = 0.89178
# - recall = 0.80579
# - specificity = 0.84435
# [[243, 45], [79, 327]]