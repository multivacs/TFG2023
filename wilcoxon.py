from scipy.stats import wilcoxon
import numpy as np

a = [[0.9057, 0.9629, 0.88, 0.9286], [0.8302, 0.8843, 0.8, 0.8571], [0.7736, 0.88, 0.76, 0.7857], [0.8302, 0.8793, 0.8, 0.8571], [0.7925, 0.8271, 0.6, 0.9643], [0.7736, 0.7364, 0.56, 0.9643], [0.7736, 0.7779, 0.64, 0.8929], [0.7358, 0.8471, 0.6, 0.8571], [0.9057, 0.9593, 0.84, 0.9643], [0.7692, 0.8681, 0.8, 0.7407]]
b=[[0.8679, 0.8686, 0.88, 0.8571], [0.8491, 0.8486, 0.84, 0.8571], [0.8868, 0.8886, 0.92, 0.8571], [0.8491, 0.8486, 0.84, 0.8571], [0.7736, 0.7729, 0.76, 0.7857], [0.8491, 0.8421, 0.72, 0.9643], [0.8113, 0.8064, 0.72, 0.8929], [0.8679, 0.8621, 0.76, 0.9643], [0.9434, 0.9421, 0.92, 0.9643], [0.8654, 0.863, 0.8, 0.9259]]
c=[[0.7358, 0.735, 0.72, 0.75], [0.8302, 0.8329, 0.88, 0.7857], [0.7925, 0.7843, 0.64, 0.9286], [0.8302, 0.835, 0.92, 0.75], [0.7358, 0.7307, 0.64, 0.8214], [0.8113, 0.8043, 0.68, 0.9286], [0.6792, 0.6836, 0.76, 0.6071], [0.7925, 0.7821, 0.6, 0.9643], [0.9245, 0.9264, 0.96, 0.8929], [0.8077, 0.8059, 0.76, 0.8519]]
d=[[0.7925, 0.7864, 0.68, 0.8929], [0.8491, 0.8464, 0.8, 0.8929], [0.8868, 0.8886, 0.92, 0.8571], [0.9057, 0.9021, 0.84, 0.9643], [0.8113, 0.8021, 0.64, 0.9643], [0.8302, 0.8243, 0.72, 0.9286], [0.8113, 0.8, 0.6, 1.0], [0.8491, 0.84, 0.68, 1.0], [0.9245, 0.9221, 0.88, 0.9643], [0.8462, 0.8415, 0.72, 0.963]]

# Métricas ROC AUC de los modelos
model1_roc_auc = [subarray[1] for subarray in a]
model2_roc_auc = [subarray[1] for subarray in b]
model3_roc_auc = [subarray[1] for subarray in c]
model4_roc_auc = [subarray[1] for subarray in d]

models = [model1_roc_auc, model2_roc_auc, model3_roc_auc, model4_roc_auc]
num_models = len(models)
model_name = ['TabNet', 'TabTransformer', 'Node', '1DCNN']

# Realizar el test de Wilcoxon para todas las comparaciones
p_values = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(num_models):
        if i != j:
            _, p_value = wilcoxon(models[i], models[j], alternative='greater')
            p_values[i, j] = p_value

# Ajustar los p-valores con el método de Bonferroni
'''
* "num_models": representa el número total de modelos que se están comparando.
* "(num_models - 1)": representa el número de comparaciones posibles entre los modelos (es decir, el número de pares únicos que se pueden formar entre los modelos).
* "(num_models - 1) / 2": se utiliza para tener en cuenta que cada comparación se realiza dos veces (una vez para cada dirección del par).
* "num_models * (num_models - 1) / 2": representa el número total de comparaciones realizadas en todas las posibles combinaciones de modelos.
'''
adjusted_p_values = p_values * num_models * (num_models - 1) / 2

# Imprimir los p-valores ajustados
for i in range(num_models):
    for j in range(num_models):
        if i != j:
            print(f"Comparación entre modelos {model_name[i]} y {model_name[j]}: p-valor ajustado = {adjusted_p_values[i, j]}")
