# Modelo Logit para Análisis de Desnutrición Crónica Infantil

Este proyecto implementa un **modelo de regresión logística (Logit)** para analizar la **prevalencia de desnutrición crónica en niñas ecuatorianas**, con especial atención a la variable **etnia** como factor clave. El análisis se basa en una muestra del ENDI (Encuesta Nacional de Desnutrición Infantil), y tiene como objetivo identificar grupos con mayor riesgo según características socioeconómicas y demográficas.

---

## Estructura del análisis

### 1. Exploración de datos
- Filtrado de los datos para trabajar exclusivamente con **niñas**.
- Exploración de la distribución de la variable **etnia** en la población objetivo.

### 2. Preparación de datos y modelado
- Selección de variables predictoras: `n_hijos`, `region`, `condicion_empleo`, `etnia`.
- Transformación de variables categóricas en **dummies**.
- Escalamiento (solo si es necesario).
- Ajuste de un **modelo logit ponderado** usando `statsmodels`.
- Análisis de significancia de coeficientes.
- Evaluación de precisión del modelo en el conjunto de prueba.

### 3. Validación cruzada
- Implementación de **KFold con 100 pliegues** para validar la estabilidad del modelo.
- Almacenamiento de los coeficientes beta estimados en cada pliegue.
- Cálculo de la precisión media del modelo.
- Visualización de histogramas de precisión y coeficientes.

### 4. Interpretación de resultados
- Identificación de **etnias con menor probabilidad** de presentar desnutrición crónica.
- Comparación entre el modelo completo y el promedio de los pliegues.
- Evaluación de la **disminución leve en la precisión** con validación cruzada.

---

## Resultados clave

- La etnia **Mestizo/Blanco** muestra una asociación significativa y negativa con la desnutrición crónica en niñas.
- La precisión del modelo pasa de **0.81 (entrenamiento/prueba)** a **0.79** con validación cruzada.
- Los **coeficientes beta disminuyen ligeramente**, lo cual indica estabilidad y consistencia del modelo.

---

## Requisitos

- Python 3.x  
- pandas  
- numpy  
- matplotlib  
- statsmodels  
- scikit-learn  

---

## Datos

- `sample_endi_model_10p.txt`: Submuestra del ENDI con información de salud, etnia y contexto socioeconómico.


Este análisis forma parte de un ejercicio de evaluación final y está enfocado en el aprendizaje y aplicación de modelos logit para problemas sociales reales.

