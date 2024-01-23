# Esto es una variable de texto
mi_variable = "Hola Mundo"

# Esto es una lista de números
mi_lista = [1, 2, 3, 4, 5]

# Esto es un diccionario 
mi_diccionario = {"clave": "valor", "clave_2": "valor_2"}

# Creemos vectores con 5 elementos repetidos cada uno
vector_entero = [10] * 5
vector_flotante = [3.14] * 5 # Con decimales
vector_complejo = [(1 + 2j)] * 5

# Crear un diccionario que contenga estos vectores
diccionario = {
    "entero": vector_entero,
    "flotante": vector_flotante,
    "complejo": vector_complejo
}

print(diccionario)

#Cadenas
cadena_simple = 'Hola, mundo!'

cadena_doble = ["¡Python es poderoso!", "Me gusta aprender"]

#Valores booleanos
valores_logicos = [True, False]


#Dataframe
import pandas as pd

datos = {
    "Nombre": ["Juan", "María", "Carlos", "Ana"],
    "Juego 1 (puntos)": [150, 100, 130, 200],
    "Juego 2 (puntos)": [120, 90, 110, 160],
    "Juego 3 (puntos)": [200, 160, 180, 190]
}

df = pd.DataFrame(datos)
print(df)