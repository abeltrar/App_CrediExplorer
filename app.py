

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#Importamos los datos y miramos los primero 5
df = pd.read_excel('Data/solicitudes_credito.xlsx')
df.head()

#Miramos que datos se pueden utilizar de esta base de datos
df.describe()

#Ver los opciones tenemos de las columnas
df['estado_civil'].unique()

#Ver que columnas tenemos
df.columns

#Vemos clasificados los datos que tenemos
sns.histplot(df['estado_civil'])

#En gastos mensuales ahi 21 personas que tienen gastos en negativo ¿Datos erroneos?
df[df['gastos_mensuales']<=0].count()

#Al igual que los ingresos ¿Datos erroneos?
df[df['ingresos_mensuales']<=0].count()

#Creamos una columna nueva donde volvemos los ingresos mensuales de dolares a pesos Colombiano
df['sueldo en pesos']=np.round(df['ingresos_mensuales']*4155,0)
df

#Gastos
df['gasto en pesos']=np.round(df['gastos_mensuales']*4155,0)
df

#Solicitado
df['monto_solicitado (Cop)']=np.round(df['monto_solicitado (USD)']*4155,0)
df

#Volvemos los valores en numeros para que la computadora pueda entenderlo
encodess=OneHotEncoder(sparse_output=False)
#Crea las nuevas columnas binarias para cada categoría única encontrada en 'estado_civil'. La salida es una matriz con 0s y 1s.
encoder=encodess.fit_transform(df[['estado_civil']])
#cómo se llaman las nuevas columnas que se genero basándote en la columna 'estado_civil', y guarda esos nombres en una lista que llamaré columns
columns=encodess.get_feature_names_out(['estado_civil'])
#Se utiliza los nombres de columna obtenidos en el paso anterior para nombrar las nuevas columnas.
df[columns]=encoder
df

#Columnas de genero
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['genero']])
columns=encodess.get_feature_names_out(['genero'])
df[columns]=encoder
df

#Columnas de nivel_educativo
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['nivel_educativo']])
columns=encodess.get_feature_names_out(['nivel_educativo'])
df[columns]=encoder
df

#Columnas de ciudad
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['ciudad']])
columns=encodess.get_feature_names_out(['ciudad'])
df[columns]=encoder
df

#Columnas de tipo_empleo
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['tipo_empleo']])
columns=encodess.get_feature_names_out(['tipo_empleo'])
df[columns]=encoder
df

#Columnas de tiene_empleo
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['tiene_empleo']])
columns=encodess.get_feature_names_out(['tiene_empleo'])
df[columns]=encoder
df

#Columnas de tipo_credito
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['tipo_credito']])
columns=encodess.get_feature_names_out(['tipo_credito'])
df[columns]=encoder
df

#Columnas de Banco
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['banco']])
columns=encodess.get_feature_names_out(['banco'])
df[columns]=encoder
df

#Columnas de tiene_empleo
encodess=OneHotEncoder(sparse_output=False)
encoder=encodess.fit_transform(df[['aprobado']])
columns=encodess.get_feature_names_out(['aprobado'])
df[columns]=encoder
df

#Eliminamos las columnas que ya convertimos
df.drop(columns=['ingresos_mensuales','gastos_mensuales','monto_solicitado (USD)','estado_civil','genero','nivel_educativo','ciudad','tipo_empleo','tiene_empleo','tipo_credito','banco','aprobado'],inplace=True)

#Creamos un archivo con los datos que tenemos actualmente en df
df.to_csv('solicitudes_credito.csv',index=False)

"""$$\frac{x-x_{min}}{x_{max}-x_{min}}$$"""

#La formula de arriba la aplicamos en los montos de dinero
df['Sueldo']=(df['sueldo en pesos']-df['sueldo en pesos'].min())/(df['sueldo en pesos'].max()-df['sueldo en pesos'].min())

df

#Gastos
df['gasto']=(df['gasto en pesos']-df['gasto en pesos'].min())/(df['gasto en pesos'].max()-df['gasto en pesos'].min())

#monto_solicitado
df['solicitado']=(df['monto_solicitado (Cop)']-df['monto_solicitado (Cop)'].min())/(df['monto_solicitado (Cop)'].max()-df['monto_solicitado (Cop)'].min())

#Eliminamos las columnas que ya convertimos
df.drop(columns=['sueldo en pesos','gasto en pesos','monto_solicitado (Cop)'],inplace=True)

df

#Aca equivocacion mia ,dividi el resultado y este no se tiene que dividir ,asi que con este comando la vuelvo a unificar
#y lo pongo para que me de el resultado
X=df.drop(columns=['aprobado_False','aprobado_True'],axis=1)
y=df['aprobado_True']
X

#Entrenamos ahora al modelo
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_test

x_train

#Angela vea de aqui para abajo utilizo varios modelos para probar cual es el mas exacto
model_selection=KNeighborsClassifier()
model_selection.fit(x_train,y_train)

#El primero modelo me dio 87% de precision
model_selection.score(x_train,y_train)

y_test

model_selection1=GaussianNB()
model_selection1.fit(x_train,y_train)

#El segundo modelo es mas bajo que el anterior con 81% de precision
model_selection1.score(x_train,y_train)

model_selection2=MultinomialNB()
model_selection2.fit(x_train,y_train)

#Y el tercero es como el peorcito con 70% ,asi que el mejor modelo es KNeighborsClassifier
#aunque si quieres te tomas la tarea de probar otros modelos para ver si hay una mas precisa
model_selection2.score(x_train,y_train)



#aca le digo que me muestre un dato random de la lista para probar el modelo
n=np.array(x_test.iloc[1].to_list())
n

#le doy el valor de n
n=[2.30000000e+01, 2.06000000e+02, 3.60000000e+01, 0.00000000e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00,
       1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 1.00000000e+00, 6.06539375e-01, 5.58752905e-01,
       8.03170359e-01]

#Le digo que me prediga si me aprobarian o no ,donde 1 es aprobado y 0 es rechazado
model_selection.predict([n])


import joblib
# Escaladores (los mismos que usaste para transformar en entrenamiento)
scaler = StandardScaler()
X[['Sueldo', 'gasto', 'solicitado']] = scaler.fit_transform(X[['Sueldo', 'gasto', 'solicitado']])

# No escales nuevamente, solo guarda la estructura
joblib.dump(X.columns.tolist(), 'columnas_modelo.pkl')
joblib.dump(model_selection, 'modelo_credito.pkl')
