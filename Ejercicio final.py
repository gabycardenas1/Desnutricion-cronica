# Ejercicio final

# Variable clave: etnia
# Población objetivo: mujer

# Librerías a usar
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Leer los datos del archivo
datos = pd.read_csv("Datos\sample_endi_model_10p.txt", sep=";")

#Eliminamos filas con valores nulos
datos = datos[~datos["dcronica"].isna()]


######################################################################################################
# Ejercicio 1: Exploración de datos
# Seleccionamos las niñas
datos = datos[datos['sexo'] == "Mujer"]

# Agrupamos para obetener el número de niñas en cada una de las categorías
agrup = datos.groupby("etnia").size()
print(f"\nEl número de niñas correspondientes a cada categoria de la variable: {agrup}")


######################################################################################################
# Ejercicio 2: Modelo Logit
# Definimos las variables de importancia añadiendo la variable asignada, en este caso 'etnia'
variables = ['n_hijos', 'region', 'condicion_empleo', 'etnia']

# Eliminamos los valores vacíos de todas las variables
for i in variables:
    datos = datos[~datos[i].isna()]

# Reemplazamos los códigos 1, 2 y 3 por "Costa", "Sierra" y "Oriente"
datos["region"] = datos["region"].apply(lambda x: "Costa" if x == 1 else "Sierra" if x == 2 else "Oriente")

# Separamos variables categóricas de numéricas.
variables_categoricas = ['region', 'etnia', 'condicion_empleo']    #Se transforma en dummies
variables_numericas = ['dcronica']     #Transformación escalar

# Transformación de datos
transformador = StandardScaler()
datos_escalados = datos.copy()
# No es necesario transformar la variable 'dcronica' ya que está en binario.
# Cuando convertimos en datos dummies con drop_first le indicamos al modelo que tome la primera categoría como referencia por lo que esta no se refleja como una columna al usar datos_dummies.columns
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
print(f"\nLos datos dummies son:\n {datos_dummies.columns}") #Esta función nos permite ver todas las columnas dummies

# Seleccionamos las variables predictoras 
# Eliminamos la condicion 'sexo_Mujer' ya que filtramos anteriormente solo para mujeres y 'condicion_empleo_Menor a 15 años' ya que no existe ninguna niña en esta categoria
# Añadimos 'region_oriente' y todas las categorias de etnia tomando en cuenta que la categoria 'Afroecuatoriana/o' es la categoria de referencia
X = datos_dummies[['n_hijos', 'region_Sierra', 'region_Oriente', 'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'etnia_Indígena', 'etnia_Mestizo/Blanco', 'etnia_Montubia/o']]
y = datos_dummies["dcronica"]
weights = datos_dummies['fexp_nino']

# Definimos la cantidad de datos que serán de entrenamiento y de prueba
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42) #Si test es de 20% entonces entrenamos con el 80% de la base

# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

# Ajustamos el modelo
# Itera hasta encontrar un modelo estable que explique la relación de las variables
modelo = sm.Logit(y_train, X_train) #Primero se declara la variable dependiente y después el df de variables
result = modelo.fit()
print(f"\nModelo Logit:\n{result.summary()}")

# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']
print(f"\n{df_coeficientes}")

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente') #pivot_table  ->  Formato de columnas
df_pivot.reset_index(drop=True, inplace=True)
print(f"\nTabla pivote:\n{df_pivot}")

# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predict = predictions_class == y_test
print(f"\nComparación con los valores reales:\n{predict}")
casos = np.mean(predict) #Interpreta el df como 1 y 0
print(f"\nEvaluación de la predicción: {casos}\nEl 0,81 de los casos son igual a 1\n")


#################################################################################################################
# Pregunta 3
# ¿Cuál es el valor del parámetro asociado a la variable clave si ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?
# La variable clave tiene 4 categorías, tomando en cuenta la categoría 'Afroecuatoriana/o' como categoría de referencia, entonces la única significativa sería la categoría Mestizo/Blanco
# ya que su valor p es igual a 0 y el valor del parametro es -1.5241. El resto de categorías no son significativas pues su valor p es distinto a 0

# Interpretación de los resultados
# En base al resumen del modelo podemos concluir que la única categoria de etnia que tiene un parámetro significativo es Mestizo/Blanco
# Respecto a la categoria Afroecuatorina/o las niñas de las etnias Mestizo/Blanco, Indígena y Montubia/o tienen menor probabilidad de sufrir desnutrición crónica 
#################################################################################################################


# Validación cruzada
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

print(f"\nPrecisión promedio de validación cruzada:{np.mean(accuracy_scores)}\n")


#######################################################################################################################
# Ejercicio 3: Evaluación del Modelo con Datos Filtrados
# Calcular la precisión promedio
precision_promedio = np.mean(accuracy_scores)

# Histograma
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

# Histograma 'n_hijos'
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


########################################################################################################
# Pregunta 3
# ¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? (Incremento o disminuye ¿Cuanto?)
# Después de la validación cruzada y con 100 pliegues la precisión del modelo disminuye de 0.81 a 0.79

# ¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior? (Incrementa o disminuye ¿Cuanto?)
# Los coeficientes beta disminuyen. La etnia Indígena de -0.21 a -0.26, la etnia Mestizo/Blanco de -1.52 a -1.56 y la etnia Montubia/o de -1.55 a -1.59.