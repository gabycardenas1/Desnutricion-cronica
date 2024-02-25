# Exploración de Datos

# Variable clave: etnia
# Población objetivo: mujer

import pandas as pd

#Leer los datos del archivo
datos = pd.read_csv("Datos\sample_endi_model_10p.txt", sep=";")

#Eliminamos filas con valores nulos
datos = datos[~datos["dcronica"].isna()]

#Separamos las variables que necesitamos
variables = ['etnia', 'sexo']

#Seleccionamos las niñas
niñas = datos[datos['sexo'] == "Mujer"]

#Agrupamos para obetener el número de niñas en cada una de las categorías
agrup = niñas.groupby("etnia").size()
print(agrup)