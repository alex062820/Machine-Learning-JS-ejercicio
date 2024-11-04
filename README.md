# Machine Learning PY ejercicio
El código entrena un modelo de SVM para clasificar el conjunto de datos Iris usando dos características. Evalúa el rendimiento del modelo e imprime la precisión y un informe de clasificación. Finalmente, visualiza cómo el modelo separa las diferentes clases con un gráfico que muestra las fronteras de decisión.




  # librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# Cargar el conjunto de datos Iris
data = datasets.load_iris()
X = data.data[:, :2]  # Usar solo dos características para la visualización
y = data.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)

# Imprimir los resultados
print(f"Precisión del modelo: {accuracy:.2f}")
print("\nReporte de clasificación:\n", report)

# Visualizar las fronteras de decisión
h = .02  # Paso en la malla
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("Fronteras de decisión del modelo SVM")
plt.show()
