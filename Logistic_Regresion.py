# -*- coding: utf-8 -*-
"""
@author: Elsy
# 

Elsy Yuliana Silgado Rivera
ID: 502194
elsy.silgado@upb.edu.co
"""

# LAS LIBRERÍAS PARA EL TRATAMIENTO DE DATOS
# ==============================================================================
import pandas as pd
import numpy as np

# LAS LIBRERÍAS PARA LOS GRÁFICOS
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# LAS LIBRERÍAS PARA LOS PREPROCESADO Y MODELADO
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# LAS LIBRERÍAS PARA LA CONFIGURACIÓN MATPLOTLIB
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# LA CONFIGURACIÓN WARNINGS
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# DATOS: LA VARIABLE MATRÍCULA ESTÁ ENCRIPTADA 
# COMO 0 SI ÉSTE NO SE TIENE MATRÍCULA Y 1 SI SE TIENE.
# ==============================================================================
matricula = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                     0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                     1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                     1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 1, 1])

matematicas = np.array([
                  41, 53, 54, 47, 57, 51, 42, 45, 54, 52, 51, 51, 71, 57, 50, 43,
                  51, 60, 62, 57, 35, 75, 45, 57, 45, 46, 66, 57, 49, 49, 57, 64,
                  63, 57, 50, 58, 75, 68, 44, 40, 41, 62, 57, 43, 48, 63, 39, 70,
                  63, 59, 61, 38, 61, 49, 73, 44, 42, 39, 55, 52, 45, 61, 39, 41,
                  50, 40, 60, 47, 59, 49, 46, 58, 71, 58, 46, 43, 54, 56, 46, 54,
                  57, 54, 71, 48, 40, 64, 51, 39, 40, 61, 66, 49, 65, 52, 46, 61,
                  72, 71, 40, 69, 64, 56, 49, 54, 53, 66, 67, 40, 46, 69, 40, 41,
                  57, 58, 57, 37, 55, 62, 64, 40, 50, 46, 53, 52, 45, 56, 45, 54,
                  56, 41, 54, 72, 56, 47, 49, 60, 54, 55, 33, 49, 43, 50, 52, 48,
                  58, 43, 41, 43, 46, 44, 43, 61, 40, 49, 56, 61, 50, 51, 42, 67,
                  53, 50, 51, 72, 48, 40, 53, 39, 63, 51, 45, 39, 42, 62, 44, 65,
                  63, 54, 45, 60, 49, 48, 57, 55, 66, 64, 55, 42, 56, 53, 41, 42,
                  53, 42, 60, 52, 38, 57, 58, 65])

datos = pd.DataFrame({'matricula': matricula, 'matematicas': matematicas})
datos.head(3)


# PARA GENERAR UN MODELO DE REGRESIÓN LOGÍSTICA SIMPLE EL PRIMER PASO  
# ES REPRESENTAR LOS DATOS PARA PODER INTUIR SI EXISTE UNA RELACIÓN 
# ENTRE LA VARIABLE INDEPENDIENTE Y LA VARIABLE RESPUESTA

# EL NÚMERO DE OBSERVACIONES POR CLASE
# ==============================================================================
datos.matricula.value_counts().sort_index()

# Se puede identificar a partir del Gráfico que si existe una relacion
# ya que a partir de 60 puntos de nota en matematica aprox. hay una relacion
# entre tener o no matricula de honor. Esta información es útil para 
# considerar la nota de matemáticas como un buen predictor para el modelo.
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

sns.violinplot(
        x     = 'matricula',
        y     = 'matematicas',
        data  = datos,
        #color = "white",
        ax    = ax
    )

ax.set_title('Distribución notas de matemáticas por clase');


# T-TEST ENTRE CLASES
# T-TEST: COMPARACIÓN DE MEDIAS POBLACIONALES INDEPENDIENTES 
# PARA ESTUDIAR SI LA DIFERENCIA OBSERVADA ENTRE LAS MEDIAS DE DOS GRUPOS 
# ES SIGNIFICATIVA, SE PUEDE RECURRIR A MÉTODOS PARAMÉTRICOS COMO EL BASADO 
# EN Z-SCORES O EN LA DISTRIBUCIÓN T-STUDENT. EN AMBOS CASOS, 
# SE PUEDEN CALCULAR TANTO INTERVALOS DE CONFIANZA PARA SABER ENTRE QUE 
# VALORES SE ENCUENTRA LA DIFERENCIA REAL DE LAS MEDIAS POBLACIONALES O 
# TEST DE HIPÓTESIS PARA DETERMINAR SI LA DIFERENCIA ES SIGNIFICATIVA.

# ==============================================================================
res_ttest = ttest_ind(
                x1 = matematicas[matricula == 0],
                x2 = matematicas[matricula == 1],
                alternative='two-sided'
            )
print(f"t={res_ttest[0]}, p-value={res_ttest[1]}")


# DIVISIÓN DE LOS DATOS EN (TRAIN Y TEST)
# SE ARREGLA UN MODELO EMPLEANDO COMO VARIABLE RESPUESTA MATRICULA Y 
# COMO PREDICTOR MATEMATICAS. COMO EN TODO ESTUDIO PREDICTIVO, NO SOLO ES 
# IMPORTANTE AJUSTAR EL MODELO, SINO TAMBIÉN CUANTIFICAR SU CAPACIDAD PARA 
# PREDECIR NUEVAS OBSERVACIONES. PARA PODER HACER ESTA EVALUACIÓN, SE DIVIDEN 
# LOS DATOS EN DOS GRUPOS, UNO DE ENTRENAMIENTO Y OTRO DE TEST.
# ==============================================================================
X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# //LA CREACIÓN DEL MODELO//
# ==============================================================================
# Para no incluir ningún tipo de regularización en el modelo se indica
# penalty='none'
modelo = LogisticRegression(penalty='none')
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)

# //LA INFORMACIÓN DEL MODELO//
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Accuracy de entrenamiento:", modelo.score(X, y))

# //LAS PREDICCIONES PROBABILÍSTICAS//
# ==============================================================================
# CON .PREDICT_PROBA() SE OBTIENE, PARA CADA OBSERVACIÓN, LA PROBABILIDAD PREDICHA
# DE PERTENECER A CADA UNA DE LAS DOS CLASES.
# UNA VEZ ENTRENADO EL MODELO, SE PUEDEN PREDECIR NUEVAS OBSERVACIONES.
# EN ESTE CASO UTILIZANDO LA MUESTRA ALMACENADA EN TEST
predicciones = modelo.predict_proba(X = X_test)
predicciones = pd.DataFrame(predicciones, columns = modelo.classes_)
predicciones.head(3)

# //LAS PREDICCIONES CON CLASIFICACIÓN FINAL//
# ==============================================================================
# Con .predict() se obtiene, para cada observación, la clasificación predicha por
# el modelo. Esta clasificación se corresponde con la clase con mayor probabilidad.
predicciones = modelo.predict(X = X_test)
predicciones

# ==============================================================================

# //STATSMODELS//
# LA IMPLEMENTACIÓN DE REGRESIÓN LOGÍSTICA DE STATSMODELS, ES MÁS COMPLETA 
# QUE LA DE SCIKITLEARN YA QUE, ADEMÁS DE AJUSTAR EL MODELO, 
# PERMITE CALCULAR LOS TEST ESTADÍSTICOS Y ANÁLISIS NECESARIOS PARA 
# VERIFICAR QUE SE CUMPLEN LAS CONDICIONES SOBRE LAS QUE SE BASA ESTE 
# TIPO DE MODELOS. STATSMODELS TIENE DOS FORMAS DE ENTRENAR EL MODELO:

# INDICANDO LA FÓRMULA DEL MODELO Y PASANDO LOS DATOS DE ENTRENAMIENTO 
# COMO UN DATAFRAME QUE INCLUYE LA VARIABLE RESPUESTA Y LOS PREDICTORES. 
# ESTA FORMA ES SIMILAR A LA UTILIZADA EN R.

# PASAR DOS MATRICES, UNA CON LOS PREDICTORES Y OTRA CON LA VARIABLE RESPUESTA.
# ESTA ES IGUAL A LA EMPLEADA POR SCIKITLEARN CON LA DIFERENCIA DE QUE A LA 
# MATRIZ DE PREDICTORES HAY QUE AÑADIRLE UNA PRIMERA COLUMNA DE 1S.



# LA DIVISIÓN DE LOS DATOS EN (TRAIN Y TEST)
# ==============================================================================
X = datos[['matematicas']]
y = datos['matricula']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# CREACIÓN DEL MODELO UTILIZANDO EL MODO FÓRMULA (SIMILAR A R)
# ==============================================================================
# DATOS_TRAIN = PD.DATAFRAME(NP.HSTACK((X_TRAIN, Y_TRAIN)),
#                            COLUMNS=['MATEMATICAS', 'MATRICULA'])
# MODELO = SMF.LOGIT(FORMULA = 'MATRICULA ~MATEMATICAS', DATA = DATOS_TRAIN)
# MODELO = MODELO.FIT()
# PRINT(MODELO.SUMMARY())

# CREACIÓN DEL MODELO UTILIZANDO MATRICES COMO EN SCIKITLEARN

# ==============================================================================
# A LA MATRIZ DE PREDICTORES SE LE TIENE QUE AÑADIR UNA COLUMNA 
# DE 1S PARA EL INTERCEPT DEL MODELO
X_train = sm.add_constant(X_train, prepend=True)
modelo = sm.Logit(endog=y_train, exog=X_train,)
modelo = modelo.fit()
print(modelo.summary())

# INTERVALOS DE CONFIANZA PARA LOS COEFICIENTES DEL MODELO
# ADEMÁS DEL VALOR DE LAS ESTIMACIONES DE LOS COEFICIENTES PARCIALES
# DE CORRELACIÓN DEL MODELO, ES CONVENIENTE CALCULAR SUS CORRESPONDIENTES 
# INTERVALOS DE CONFIANZA.
# ==============================================================================
intervalos_ci = modelo.conf_int(alpha=0.05)
intervalos_ci = pd.DataFrame(intervalos_ci)
intervalos_ci.columns = ['2.5%', '97.5%']
intervalos_ci


# //LAS PREDICCIONES//

# UNA VEZ ENTRENADO EL MODELO, SE PUEDEN OBTENER PREDICCIONES PARA NUEVOS DATOS. 
# LOS MODELOS DE REGRESIÓN LOGÍSTICA DE STATSMODELS DEVUELVEN LA PROBABILIDAD DE 
# PERTENECER A LA CLASE DE REFERENCIA.

# PREDICCIÓN DE PROBABILIDADES
# ==============================================================================
predicciones = modelo.predict(exog = X_train)
predicciones[:4]

# Para obtener la clasificación final, se convierten los valores de 
# probabilidad mayores de 0.5 a 1 y los mejores a 0.

# //CLASIFICACIÓN PREDICHA//
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion


# ADEMÁS DE LA LÍNEA DE MÍNIMOS CUADRADOS, ES RECOMENDABLE INCLUIR LOS LÍMITES 
# SUPERIOR E INFERIOR DEL INTERVALO DE CONFIANZA. ESTO PERMITE IDENTIFICAR 
# LA REGIÓN EN LA QUE, SEGÚN EL MODELO GENERADO Y PARA UN DETERMINADO NIVEL 
# DE CONFIANZA, SE ENCUENTRA EL VALOR PROMEDIO DE LA VARIABLE RESPUESTA.

# PREDICCIONES EN TODO EL RANGO DE X
# ==============================================================================
# SE CREA UN VECTOR CON NUEVOS VALORES INTERPOLADOS EN EL RANGO DE OBSERVACIONES
grid_X = np.linspace(
            start = min(datos.matematicas),
            stop  = max(datos.matematicas),
            num   = 200
         ).reshape(-1,1)

grid_X = sm.add_constant(grid_X, prepend=True)
predicciones = modelo.predict(exog = grid_X)


# //GRÁFICO DEL MODELO//
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(
    X_train[(y_train == 1).flatten(), 1],
    y_train[(y_train == 1).flatten()].flatten()
)
ax.scatter(
    X_train[(y_train == 0).flatten(), 1],
    y_train[(y_train == 0).flatten()].flatten()
)
ax.plot(grid_X[:, 1], predicciones, color = "gray")
ax.set_title("Modelo regresión logística")
ax.set_ylabel("P(matrícula = 1 | matemáticas)")
ax.set_xlabel("Nota matemáticas");


X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")


# CONCLUSIÓN!


# ES CREADO PARA PREDECIR LA PROBABILIDAD DEL MODELO LOGÍSTICO DE QUE UN 
# ALUMNO OBTENGA MATRÍCULA DE HONOR MEDIANTE LA NOTA DE MATEMÁTICAS ES 
# UN CONJUNTO SIGNIFICATIVO (LIKELIHOOD RATIO P-VALUE = 9.831E-11). 
# EL P-VALUE DEL PREDICTOR MATEMÁTICAS ES SIGNIFICATIVO (P-VALUE = 7.17E-08).
# P(MATRICULA)=E−8.9848+0.1439∗NOTA MATEMATICAS1+E−8.9848+0.1439∗NOTA MATEMATICAS
# EL CONJUNTO DE TEST CON LOS RESULTADOS OBTENIDOS  INDICAN QUE EL 
# MODELO ES CAPAZ DE CLASIFICAR EL 87.5% DE LAS OBSERVACIONES CORRECTAMENTE.


