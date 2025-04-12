# Predicción del trastorno por déficit de atención con redes neuronales

## Resumen
En la actualidad existen diferentes técnicas basadas en algoritmos de Machine Learning (ML) para la predicción del diagnóstico del trastorno por déficit de atención e hiperactividad (TDAH). Para conseguir un diagnóstico preciso se necesitan pruebas realizadas en hospitales que conllevan un alto coste económico, de recursos y de tiempo para el paciente. En cambio, con la aplicación de un test de escala de inteligencia aplicado por un experto (psicólogo u orientador) se pueden obtener ciertos factores que definen el perfil de una persona que se pueden usar para predecir si tiene o no este trastorno. Como beneficios tenemos que es un sistema no intrusivo y se puede aplicar en diferentes contextos más cercanos a la persona evaluada como en consultas de psicólogos, o bien, en los colegios, así que es fácilmente escalable o generalizable. Este trabajo se centra en el estudio de diversas arquitecturas de redes neuronales artificiales recientes aplicadas a datos tabulados con las que encontrar así aquellos modelos más precisos y que mejor diagnostiquen este trastorno, a la vez que se compararán los resultados con otros algoritmos ML. Finalmente, conoceremos si estas redes neuronales recientes pueden mejorar al resto de estilos de aprendizaje en el caso de diagnóstico de TDAH según los datos disponibles.

## Descripción
En este repositorio se recoge el código utilizado para evalúar el rendimiento de algoritmos de Deep Learning aplicados a datos estructurados de tipo tabular. Se utilizaron los modelos:
- TabNet
- TabTransformer
- Node
- 1DCNN


## Dataset
En el estudio original, se utilizaron datos anonimizados de niños entre 6 y 17 años con las respuestas obtenidas en los tests de WISC-IV.
Dado que el objetivo de este repositorio es meramente referencial, en el archivo `main.py` se genera de manera aleatoria 1000 muestras sintéticas a modo ilustrativo.


## How to use

Primeramente, instala las dependencias con `pip install -r requirements.txt`. Es recomendable usar un entorno virtual como venv o conda para que no haya conflictos de librerías.

En la carpeta donde se sitúe el archivo main.py deben de existir a su vez dos carpetas más de nombre: "hyperopt" y "results" para volcar la salida de los algoritmos.
En main.py podemos utilizar las variables globales al principio del fichero para configurar diferentes situaciones (feature engineering, ajuste de hiperparámetros con hyperopt, evaluación de los algoritmos).
