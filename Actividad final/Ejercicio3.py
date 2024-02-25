# Evaluación del Modelo con Datos Filtrados

# Variable clave: etnia
# Población objetivo: mujer

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Leer los datos del archivo
datos = pd.read_csv("Datos/sample_endi_model_10p.txt", sep=";")
print(datos)

# Elimina los datos faltantes en dcronica
datos = datos[~datos["dcronica"].isna()]
datos = datos[(datos['sexo'] == "Mujer") & (datos['etnia'].isin(['Afroecuatoriana/o', 'Indígena', 'Mestizo/Blanco', 'Montubia/o']))]
print(datos)

# Elimina los datos faltantes en todas las categorias guardadas en variables
variables = ['n_hijos', 'sexo', 'etnia']
for i in variables:
    datos = datos[~datos[i].isna()]

# Transformación de variables
variables_categoricas = ['sexo', 'etnia']
variables_numericas = ['dcronica', 'n_hijos']

transformador = StandardScaler()
datos_escalados = datos.copy()
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
print(datos_dummies.columns)

# Variables predictoras y peso
X = datos_dummies[['n_hijos','etnia_Indígena', 'etnia_Mestizo/Blanco', 'etnia_Montubia/o']]
y = datos_dummies["dcronica"]
weights = datos_dummies['fexp_nino']

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Convertir todas las variables a tipo entero
X_train = X_train.apply(pd.to_numeric, errors='coerce').astype(int)
y_train = y_train.apply(pd.to_numeric, errors='coerce').astype(int)
X_test = X_test.apply(pd.to_numeric, errors='coerce').astype(int)

# Reemplazar los valores de 2 por 1
y_train.replace(2, 1, inplace=True)
y_train.unique()

# Ajuste del modelo
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_class == y_test


# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

# Gráfica 1
# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

plt.hist(accuracy_scores, bins=30, edgecolor='black')
# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()


# Gráfica 2
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()