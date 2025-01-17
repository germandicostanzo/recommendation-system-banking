#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabajo Final Integrador

Recomendación de productos financieros en la organización bancaria Santander mediante técnicas de aprendizaje automático.

2022

German Di Costanzo
"""

''' INSTALACIÓN DE MÓDULOS (ejecutar linea por linea)'''

pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install kneed
pip install scikit-learn
pip install xgboost
pip install joblib
pip install scipy
pip install kaggle

''' IMPORTAR LIBRERIAS '''

import pandas as pd
import os
import numpy as np
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import xgboost  as xgb
from xgboost import DMatrix
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pandas import DataFrame
from sklearn.metrics import average_precision_score
import joblib
import scipy.stats as st
from sklearn import datasets, metrics, model_selection
import zipfile

''' OBTENER DATASET DESDE LA COMPETENCIA DE SANTANDER EN KAGGLE

1. Acceder a https://www.kaggle.com/competitions/santander-product-recommendation/data
2. Crear cuenta en Kaggle
3. Ir al perfil --> account --> Create New Api Token

Se descargará un archivo .json, reemplazar "Users/germandicostanzo" por la ubicación donde esté ese archivo .json

'''

!mkdir ~/.kaggle
!cp /Users/germandicostanzo/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c santander-product-recommendation

!unzip /Users/germandicostanzo/santander-product-recommendation.zip

zf = zipfile.ZipFile('/Users/germandicostanzo/train_ver2.csv.zip') 
df = pd.read_csv(zf.open('train_ver2.csv'))

''' TRANFORMACIÓN DEL CONJUNTO DE DATOS '''
df[df.duplicated(keep=False)]

''' Se eliminan los registros con ind_empleado nulo ya que tiene valores nulos en el resto de las variables
(pais_residencia,sexo,fecha_alta,ind_nuevo,indrel,indresi,indext,indfall,ind_actividad_cliente) '''
df = df[df.ind_empleado.isnull()==False]
df = df.reset_index(drop = True)

''' La columna conyuemp tiene mayoria de valores  nulos. Se elimina '''
df = df.drop(columns=['conyuemp'])

''' La columna ult_fec_cli_1t tiene mayoria de valores  nulos. Se elimina '''
df = df.drop(columns=['ult_fec_cli_1t'])

''' La columna tipodom tiene 1 o nulos y ademas existe el nombre y código de provincia. Se elimina '''
df = df.drop(columns=["tipodom"])

''' Se crea la columna período y no se utiliza la columna fecha_dato ''' 
df["periodo"]= pd.DatetimeIndex(df["fecha_dato"]).year*100+pd.DatetimeIndex(df["fecha_dato"]).month
df.periodo.unique()

''' La columna ind_empleado se transforma a categóricas '''
map_dict_ind_empleado = { "N": 1, "A":2,"B":3,"F":4,"S": 5}
df.ind_empleado = df.ind_empleado.apply(lambda x: map_dict_ind_empleado.get(x,x))

df.ind_empleado.value_counts()
df.ind_empleado.unique()

''' La columna sexo se transforma a categóricas y se imputan los nulos con valor mas frecuente (2)'''
map_dict_sexo = { "H": 1, "V":2}
df.sexo = df.sexo.apply(lambda x: map_dict_sexo.get(x,x))
df.sexo.fillna(2,inplace=True)

df.sexo.value_counts()
df.sexo.unique()

''' Para la columna edad los faltantes, se imputan por la media según criterio definido en el documento'''
df["age"]   = pd.to_numeric(df["age"], errors="coerce")
df.loc[df.age < 18,"age"]  = df.loc[(df.age >= 18) & (df.age <= 30),"age"].mean(skipna=True)
df.loc[df.age > 100,"age"] = df.loc[(df.age >= 30) & (df.age <= 100),"age"].mean(skipna=True)
df["age"].fillna(df["age"].mean(),inplace=True)
df["age"] = df["age"].astype(int)

''' En la columna antiguedad existen valores -999999 se reemplazan por 0'''
df.antiguedad = pd.to_numeric(df.antiguedad,errors="coerce")
df.loc[df.antiguedad <0, "antiguedad"] = 0 

''' La columna indrel_1mes se transforma a categoria y los nulos con el valor más frecuente (1)'''
map_dict = { 1.0  : 1,  "1.0" : 1,"1"   : 1,
            "3.0" : 3, "P"   : 5, 3.0   : 3,
            2.0   : 2,"3"   : 3,"2.0" : 2,
            "4.0" : 4, 4.0 : 4, "4"   : 4,
            "2"   : 2}

df.indrel_1mes.fillna(1,inplace=True)
df.indrel_1mes = df.indrel_1mes.apply(lambda x: map_dict.get(x,x))

df.indrel_1mes.value_counts()
df.indrel_1mes.unique()

''' La columna tiprel_1mes se transforma a categoria y los nulos con el valor más frecuente (1)'''
map_dict_tiprel = { "A" : 1,"I" : 2, "P" : 3,"R" : 4, "N" : 5}
df.tiprel_1mes.fillna(1,inplace=True)
df.tiprel_1mes = df.tiprel_1mes.apply(lambda x: map_dict_tiprel.get(x,x))

df.tiprel_1mes.value_counts()
df.tiprel_1mes.unique()

''' La columna indresi se transforma a categoria '''
map_dict_indresi= { "N" : 0, "S" : 1 }
df.indresi = df.indresi.apply(lambda x: map_dict_indresi.get(x,x))

df.indresi.value_counts()
df.indresi.unique()

''' La columna indext se transforma a categoria '''
map_dict_indext = { "N" : 0,
                    "S" : 1
                    }
df.indext = df.indext.apply(lambda x: map_dict_indext.get(x,x))

df.indext.value_counts()
df.indext.unique()

''' En base a la columna pais_residencia se genera una nueva indicando si es España o no''' 

df.loc[df['pais_residencia'] == 'ES', 'pais_residencia_ES'] = 1
df.pais_residencia_ES.fillna(0,inplace=True)

df = df.drop(columns=["pais_residencia"])
df.shape

df.pais_residencia_ES.unique()

''' En base a la columna canal_entrada se generan 4 indicando si fue canal_entreada_KHE, canal_entrada_KAT, canal_entrada_KFC, canal_entrada_OTRO''' 
df.canal_entrada.fillna("KHE",inplace=True)
df.canal_entrada.unique()
df.canal_entrada.value_counts()

df.loc[df['canal_entrada'] == 'KHE', 'canal_entrada_KHE'] = 1
df.loc[df['canal_entrada'] == 'KAT', 'canal_entrada_KAT'] = 1
df.loc[df['canal_entrada'] == 'KFC', 'canal_entrada_KFC'] = 1
df.loc[(df['canal_entrada'] !='KHE') & (df['canal_entrada'] !='KAT') & (df['canal_entrada'] !='KFC'), 'canal_entrada_OTRO'] = 1

df.canal_entrada_KHE.fillna(0,inplace=True)
df.canal_entrada_KAT.fillna(0,inplace=True)
df.canal_entrada_KFC.fillna(0,inplace=True)
df.canal_entrada_OTRO.fillna(0,inplace=True)

df.canal_entrada_KHE.unique()
df.canal_entrada_KAT.unique()
df.canal_entrada_KFC.unique()
df.canal_entrada_OTRO.unique()

df = df.drop(columns=["canal_entrada"])
df.shape

''' La columna indfall se transforma a categoria '''
map_dict_indfall = { "N" : 0,"S" : 1}
df.indfall = df.indfall.apply(lambda x: map_dict_indfall.get(x,x))

df.indfall.value_counts()
df.indfall.unique()

''' Nulos en codigo de provincia valor mas frecuente'''
df.cod_prov.fillna(28,inplace=True)

'''Correccion nombre provincia'''
df.loc[df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
df.nomprov.unique()

'''Para los valores faltantes en la columna rena se calcula la media por provincia y se asigna a los faltantes ese valor, según provincia'''
incomes = df.loc[df.renta.notnull(),:].groupby("nomprov").agg({"renta":{np.median}})
incomes.sort_values(by=("renta","median"),inplace=True)
incomes.reset_index(inplace=True)
incomes.nomprov = incomes.nomprov.astype(CategoricalDtype(categories=[i for i in df[df["nomprov"].notnull()].nomprov.unique()],ordered=False))

grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
df.sort_values("nomprov",inplace=True)
df             = df.reset_index()
new_incomes    = new_incomes.reset_index()

df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
df.sort_values(by="fecha_dato",inplace=True)

df.renta.isnull().sum()
df.renta.unique()

''' La columna nomprov se elimina y se utiliza el codigo como categoría '''
df = df.drop(columns=["nomprov"])
df.shape

'''Segmento a categoría, nulos más frecuente (2)'''
map_dict_segmento = { "01 - TOP" : 1, "02 - PARTICULARES" : 2,"03 - UNIVERSITARIO": 3}
df.segmento = df.segmento.apply(lambda x: map_dict_segmento.get(x,x))
df.segmento.fillna(2,inplace=True)
df.segmento.value_counts()
df.segmento.unique()

'''ind_nomina_ult1 nulos a 0''' 
df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0

'''ind_nom_pens_ult1 nulos a 0''' 
df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

df.shape
df.isnull().sum()
   
'''VARIABLES A UTILIZAR'''
df2 = df[['ncodpers', 'ind_empleado', 'sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indext', 'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1', 'periodo', 'pais_residencia_ES', 'canal_entrada_KHE', 'canal_entrada_KAT', 'canal_entrada_KFC', 'canal_entrada_OTRO']]

''' LIBERAR MEMORIA'''
del df, grouped, incomes, map_dict,map_dict_ind_empleado, map_dict_indext, map_dict_indfall,map_dict_indresi, map_dict_segmento, map_dict_sexo, map_dict_tiprel, new_incomes, zf

df2.columns
df2.shape
df2.dtypes

'''TOP PRODUCTOS'''
df2[['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']].sum().sort_values(ascending=False)

'''FIGURA 2'''

df_g = pd.DataFrame(df2.groupby('segmento').ncodpers.nunique())
df_g=df_g.set_axis(['cantidad'], axis=1, inplace=False)
df_g.columns
sns.barplot(x="cantidad", y = df_g.index , data=df_g,orient='h', palette="light:#5A9")

'''PRODUCTOS MAS VENDIDOS POR SEGMENTO (reemplazar df3.segmento == XX ) con 1,2 y 3 para ir obteniendo la tabla de cada segmento'''
df3=df2
df4 = df3[(df3.segmento == 3)].groupby(['segmento'])[['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']].sum()
df4= df4.T
df4.sort_values(ascending=False,by=df4.columns[0])
del df3
del df4

'''FIGURA 3'''
sns.boxplot(x="renta",data=df2)

'''FIGURA 4'''
sns.boxplot(x="renta",data=df2[(df2.renta < 500000)])

''' CLUSTERING '''

df_z_scaled = df2[['ncodpers','ind_empleado', 'sexo', 'age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indext',
                 'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento', 'periodo', 'pais_residencia_ES', 
                 'canal_entrada_KHE', 'canal_entrada_KAT', 'canal_entrada_KFC', 'canal_entrada_OTRO' ]].copy() 

''' ESTANDARIZACION COLUMNA RENTA '''

df_z_scaled['renta'] = (df_z_scaled['renta'] - df_z_scaled['renta'].mean()) / df_z_scaled['renta'].std()     

'''NUMERO APROPIADO DE CLUSTERS '''
kmeans_kwargs = {"init": "random",
                    "n_init": 10,
                    "max_iter": 300,
                    "random_state": 42}

sse = []
for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_z_scaled)
        sse.append(kmeans.inertia_)

'''' FIGURA 5 ''''
plt.style.use("fivethirtyeight")
plt.figure(facecolor='white')
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Número de clusters")
plt.ylabel("SSE")
ax = plt.axes()
ax.set_facecolor("white")
plt.show()
    
'''APLICAION CLUSTERING CON ALGORITMO K-MEDIAS'''
kmeans = KMeans( init="random", n_clusters=2, n_init=10, max_iter=300,random_state=42)
kmeans.fit(df_z_scaled)

''' SE GUARDA EL CLUSTER AL QUE PERTENECE CADA REGISTRO EN UNA NUEVA COLUMNA '''
df_z_scaled['cluster'] = kmeans.labels_

'''FIGURA 6'''
df_g = pd.DataFrame(df_z_scaled.groupby('cluster').ncodpers.nunique())
df_g=df_g.set_axis(['cantidad'], axis=1, inplace=False)
df_g.columns
sns.set_theme(style='white')
sns.barplot(x=df_g.index, y ="cantidad"  , data=df_g,orient='v', palette="crest")


'''DESCRIPTIVO'''

df3= df2.copy()
df3['cluster'] = df_z_scaled['cluster'] 
df3['renta'] = (df3['renta'] - df3['renta'].mean()) / df3['renta'].std()     
df3.columns

del df2
del df_g
del df_z_scaled

'''FIGURA 7'''
df_s = pd.DataFrame(df3.groupby(['cluster','segmento']).ncodpers.nunique())
df_s= df_s.reset_index()
df_s.columns
sns.set_theme(style='white')
g = sns.catplot(x="segmento", y="ncodpers",
                hue="cluster",
                data=df_s, kind="bar",
                height=4, aspect=.7);

'''FIGURA 8'''
sns.set_theme(style='white')
sns.displot(df3[["age","cluster"]], x="age", hue="cluster", stat="density", common_norm=False)

'''FIGURA 9'''
sns.set_theme(style='white')
sns.displot(df3[["antiguedad","cluster"]], x="antiguedad", hue="cluster", stat="density", common_norm=False)

'''FIGURA 10'''
sns.set_theme(style='white')
sns.displot(df3[["renta","cluster"]], x="renta", hue="cluster", stat="density", common_norm=False)
plt.xlim(-.6, 0.9)
plt.ylim(0,8.5)

'''FIGURA 11 ind_cco_fin_ult1'''
df_p = pd.DataFrame(df3.groupby(['cluster','ind_cco_fin_ult1']).ncodpers.nunique())
df_p= df_p.reset_index()
df_p.columns

sns.set_theme(style='white')
g = sns.catplot(x="ind_cco_fin_ult1", y="ncodpers",
                hue="cluster",
                data=df_p[(df_p.ind_cco_fin_ult1 == 1)], kind="bar",
                height=4, aspect=.7);

'''FIGURA 11 ind_ctop_fin_ult1'''
df_p = pd.DataFrame(df3.groupby(['cluster','ind_ctop_fin_ult1']).ncodpers.nunique())
df_p= df_p.reset_index()
df_p.columns

sns.set_theme(style='white')
g = sns.catplot(x="ind_ctop_fin_ult1", y="ncodpers",
                hue="cluster",
                data=df_p[(df_p.ind_ctop_fin_ult1 == 1)], kind="bar",
                height=4, aspect=.7);

'''FIGURA 11 ind_recibo_ult1'''
df_p = pd.DataFrame(df3.groupby(['cluster','ind_recibo_ult1']).ncodpers.nunique())
df_p= df_p.reset_index()
df_p.columns

sns.set_theme(style='white')
g = sns.catplot(x="ind_recibo_ult1", y="ncodpers",
                hue="cluster",
                data=df_p[(df_p.ind_recibo_ult1 == 1)], kind="bar",
                height=4, aspect=.7);

'''DESARROLLO DE MODELOS'''

df = df3.copy()
del df3
del df_p
del df_s

''' TABLA 2 '''

df2 = df.copy()
df2['total_productos'] = df2.loc[:,['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']].sum(axis=1)
df2.groupby(['periodo'])['total_productos'].sum().sort_values(ascending=True)
df.groupby('periodo').ncodpers.nunique()
del df2

'''SE DEJAN FUERA REGISTROS DE CLIENTES SIN ALTAS '''

a = df.iloc[:, -31:-7].sum(axis=1).to_frame(name='suma')
a=a[a['suma']<1]
df = df.loc[~df.index.isin(a.index.tolist())]

df = df.reset_index()
df.columns
df = df.drop('index', axis=1)
df.shape
df.index.tolist()
del a


''' TRANSFORMACIÓN AL FORMATO NECESARIO '''

df_train = pd.melt(df[df["periodo"] == 201508 ], id_vars=['ncodpers','periodo','ind_empleado','sexo', 'age','ind_nuevo',
                  'antiguedad','indrel','indrel_1mes','tiprel_1mes','indext',
                   'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta',
                   'segmento','pais_residencia_ES', 'canal_entrada_KHE',
                    'canal_entrada_KAT','canal_entrada_KFC', 'canal_entrada_OTRO','cluster'])

df_test = pd.melt(df[df["periodo"] == 201509 ], id_vars=['ncodpers','periodo','ind_empleado','sexo', 'age','ind_nuevo',
                  'antiguedad','indrel','indrel_1mes','tiprel_1mes','indext',
                   'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta',
                   'segmento','pais_residencia_ES', 'canal_entrada_KHE',
                    'canal_entrada_KAT','canal_entrada_KFC', 'canal_entrada_OTRO','cluster'])

df_valid = pd.melt(df[df["periodo"] == 201601 ], id_vars=['ncodpers','periodo','ind_empleado','sexo', 'age','ind_nuevo',
                  'antiguedad','indrel','indrel_1mes','tiprel_1mes','indext',
                   'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta',
                   'segmento','pais_residencia_ES', 'canal_entrada_KHE',
                    'canal_entrada_KAT','canal_entrada_KFC', 'canal_entrada_OTRO','cluster'])

df_pred = pd.melt(df[df["periodo"] == 201605 ], id_vars=['ncodpers','periodo','ind_empleado','sexo', 'age','ind_nuevo',
                  'antiguedad','indrel','indrel_1mes','tiprel_1mes','indext',
                   'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta',
                   'segmento','pais_residencia_ES', 'canal_entrada_KHE',
                    'canal_entrada_KAT','canal_entrada_KFC', 'canal_entrada_OTRO','cluster'])

df_train.rename(columns={'variable': 'producto', 'value': 'compro'}, inplace=True)
df_test.rename(columns={'variable': 'producto', 'value': 'compro'}, inplace=True)
df_valid.rename(columns={'variable': 'producto', 'value': 'compro'}, inplace=True)
df_pred.rename(columns={'variable': 'producto', 'value': 'compro'}, inplace=True)

''' TABLA 4 '''

df_train[df_train['ncodpers']==15889][['ncodpers','periodo','producto','compro']]


''' GENERACION GROUPID '''

df_train['groupid'] = df_train.ncodpers*1000+df_train.periodo%1000
df_test['groupid'] = df_test.ncodpers*1000+df_test.periodo%1000
df_valid['groupid'] = df_valid.ncodpers*1000+df_valid.periodo%1000
df_pred['groupid'] = df_pred.ncodpers*1000+df_pred.periodo%1000

#Ordeno por grupo
df_train = df_train.sort_values(by='groupid', ascending=True)
df_test = df_test.sort_values(by='groupid', ascending=True)
df_valid = df_valid.sort_values(by='groupid', ascending=True)
df_pred = df_pred.sort_values(by='groupid', ascending=True)

df_train=df_train.reset_index()
df_test=df_test.reset_index()
df_valid=df_valid.reset_index()
df_pred=df_pred.reset_index()

''' TABLA 5 '''

df_train[df_train['ncodpers']==15889][['ncodpers','periodo','groupid','producto','compro']]

''' PRODUCTOS A COLUMNAS '''
df_train=pd.get_dummies(df_train,columns=['producto'])
df_test=pd.get_dummies(df_test,columns=['producto'])
df_valid=pd.get_dummies(df_valid,columns=['producto'])

''' TAMAÑO DE GRUPOS '''
g_train = df_train.groupby('groupid').size().to_frame('size')['size'].to_numpy()
g_test = df_test.groupby('groupid').size().to_frame('size')['size'].to_numpy()
g_val = df_valid.groupby('groupid').size().to_frame('size')['size'].to_numpy()

''' SEPARACION VARIABLES PREDICTORAS Y TARGET '''
afuera =['level_0','index','ncodpers','periodo','producto','groupid','compro']
predictors = [x for x in df_train.columns if x not in afuera]

X_train = df_train[predictors]
y_train = df_train['compro']

X_test = df_test[predictors]
y_test = df_test['compro']

X_val = df_valid[predictors]
y_val = df_valid['compro']
 
''' MATRIZ XGBOOST '''
train_dmatrix = DMatrix(X_train, y_train)
valid_dmatrix = DMatrix(X_val,y_val)
test_dmatrix = DMatrix(X_test)

''' TAMAÑO MATRIZ'''
train_dmatrix.set_group(g_train)
valid_dmatrix.set_group(g_val)
test_dmatrix.set_group(g_test)

''' MODELO BASELINE '''
lr = LogisticRegression(random_state=123)
lr.fit( X_train, y_train)
print(lr.score(X_test, y_test)) # 0.95

''' MAP VALIDATION ''' #0.6936
pred = lr.predict(X_val)
X = df_valid
X['y_real'] = df_valid.compro
X['pred'] = pred
g=df_valid.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' MAP TEST ''' #.6974
pred = lr.predict(X_test)
X = df_test
X['y_real'] = df_test.compro
X['pred'] = pred
g=df_test.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' MAP TRAIN''' #0.6952
pred = lr.predict(X_train)
X = df_train
X['y_real'] = df_train.compro
X['pred'] = pred
g=df_train.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' OPTIMIZACION DE PARAMETROS (la ejecución demora)'''

sample_df_train = df_train.loc[df_train['groupid'] < 55555555]
sample_df_test = df_test.loc[df_test['groupid'] < 55555555]
sample_df_val = df_valid.loc[df_valid['groupid'] < 55555555]

X_train_s = sample_df_train[predictors]
y_train_s = sample_df_train['compro']

X_test_s = sample_df_test[predictors]
y_test_s  = sample_df_test['compro']

X_val_s = sample_df_val[predictors]
y_val_s  = sample_df_val['compro']

modelo_random=XGBClassifier(seed=123, objective= 'binary:logistic')
 
params_random = { 
        "max_depth": [8, 20, 50],
        "eta": [0.5, 1,5,],
        "min_child_weight": [5,50,200,350]
        }

gs = RandomizedSearchCV(modelo_random, params_random,  n_iter=30, n_jobs=-1,cv=5 )        
bstx=gs.fit(X_train_s, y_train_s,eval_set=[(X_train_s, y_train_s), (X_test_s, y_test_s)],
                eval_metric=["auc"],early_stopping_rounds=10) 
''' RESULTADO RANDOMIZED SEARCH CV (se utilizaron como punto de partida, los parámetros finales son los que se encuentran a continuación en los modelos'''
bstx.best_params_

del sample_df_train
del sample_df_test
del sample_df_val
del X_train_s
del y_train_s
del X_test_s
del y_test_s
del X_val_s
del y_val_s
del gs
del bstx
del modelo_random


''' MODELO BINARIO'''

params = {'seed':123,'max_depth':8, 'eta':1, 'eval_metric':'auc', 
          'min_child_weight':300, 'objective':'binary:logistic'}

lr = xgb.train(
      params = params,
      dtrain = train_dmatrix, 
      num_boost_round = 100,
      evals = [(train_dmatrix, 'tr'), (valid_dmatrix, 'tv')],
      early_stopping_rounds = 10
    )   

''' VALIDACION MAP'''  #0.8642
pred = lr.predict(valid_dmatrix, ntree_limit = lr.best_ntree_limit)
X = df_valid
X['y_real'] = df_valid.compro
X['pred'] = pred
g=df_valid.groupby('groupid')  
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])        
        
'''TEST MAP''' #0.8673
w = df_test
w["rk"] = pred = lr.predict(test_dmatrix, ntree_limit = lr.best_ntree_limit)
g = w.groupby("groupid")
np.mean([average_precision_score(x.compro, x.rk) for k, x in g])


'''TRAIN MAP''' #0.8662
pred = lr.predict(train_dmatrix, ntree_limit = lr.best_ntree_limit)
X = df_train
X['y_real'] = df_train.compro
X['pred'] = pred
g=df_train.groupby('groupid')
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' GRAFICAR CURVA ROC FIGURA 14'''
ns_probs = [0 for _ in range(len(y_train))]
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_train, ns_probs)

pred = lr.predict(train_dmatrix, ntree_limit = lr.best_ntree_limit)
fpr, tpr, _ = metrics.roc_curve(df_train.compro,  pred)
roc_auc = metrics.auc(fpr, tpr) 

plt.plot(ns_fpr,ns_tpr)
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

del w,pred,g,X,fpr,tpr

''' MODELO PAIRWISE '''

train_dmatrix = DMatrix(X_train, y_train)
valid_dmatrix = DMatrix(X_val,y_val)
test_dmatrix = DMatrix(X_test)

train_dmatrix.set_group(g_train)
valid_dmatrix.set_group(g_val)
test_dmatrix.set_group(g_test)

params = {'seed':123,'max_depth':8, 'eta':1, 'eval_metric':'map', 
          'min_child_weight':300, 'objective':'rank:pairwise'}

xgb_pairwise = xgb.train(params, train_dmatrix, num_boost_round=100,
                      early_stopping_rounds= 10,
                      evals=[(train_dmatrix, 'tr'), (valid_dmatrix, 'v')])

''' TEST MAP '''  #0.8696
pred = xgb_pairwise.predict(test_dmatrix, ntree_limit = xgb_pairwise.best_ntree_limit)
X = df_test
X['y_real'] = df_test.compro
X['pred'] = pred
g=df_test.groupby('groupid')
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' TRAIN MAP''' #0.8684
pred = xgb_pairwise.predict(train_dmatrix, ntree_limit = xgb_pairwise.best_ntree_limit)
X = df_train
X['y_real'] = df_train.compro
X['pred'] = pred
g=df_train.groupby('groupid')
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' VALIDATION MAP''' #0.8673
pred = xgb_pairwise.predict(valid_dmatrix, ntree_limit = xgb_pairwise.best_ntree_limit)
X = df_valid
X['y_real'] = df_valid.compro
X['pred'] = pred
g=df_valid.groupby('groupid')
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

'''IMPORTANCIA DE VARIABLES FIGURA 16'''
from xgboost import plot_importance
plot_importance(xgb_pairwise,max_num_features=5)

'''EJEMPLO CLIENTES PERIODO 201605'''

'''Productos a dummies'''
df_pred2 =pd.get_dummies(df_pred,columns=['producto'])

'''Cantidad de cada grupo'''
g_pred = df_pred2.groupby('groupid').size().to_frame('size')['size'].to_numpy()

'''Separo target'''
afuera =['level_0','index','ncodpers','periodo','producto','groupid','compro']
predictors = [x for x in df_pred2.columns if x not in afuera]

X_pred = df_pred2[predictors]
y_pred = df_pred2['compro']

'''Matriz XGBoost'''
pred_dmatrix = DMatrix(X_pred)

'''Seteo tamaño de grupos'''
pred_dmatrix.set_group(g_pred)

'''EJECUTO EL MODELO'''
predic = xgb_pairwise.predict(pred_dmatrix, ntree_limit = xgb_pairwise.best_ntree_limit)

X = df_pred
X['y_real'] = df_pred.compro
X['pred'] = predic
g=df_pred.groupby('groupid')

''' TEST MAP''' # 0.8599
np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

''' TABLA 6 '''
X.rename(columns={'pred': 'ranking'}, inplace=True)
X[['ncodpers','producto','ranking']].loc[(X['ncodpers'] == 15889) & (X['y_real'] == 0) ].sort_values(by='ranking', ascending = False)

''' TABLA 7'''
X[['ncodpers','producto','ranking']].loc[(X['ncodpers'] == 1464736) & (X['y_real'] == 0) ].sort_values(by='ranking', ascending = False)

''' TABLA 8 PRODUCTO PRESTAMOS'''
df_max = X.loc[(X['y_real'] == 0)].sort_values('ranking').groupby(['groupid']).tail(1)
df_max = df_max.reset_index()
df_max['producto'] = df_max['producto'].astype(pd.StringDtype())
df_max.dtypes
df_max[['ncodpers','producto','ranking']].loc[(df_max['producto'].str.contains("ind_pres_fin_ult1"))].sort_values(by='ranking', ascending = False).head(10)
df_max.producto.value_counts()

''' TABLA 9 PRODUCTO CUENTA PARTICULAR PLUS'''
df_max[['ncodpers','producto','ranking']].loc[(df_max['producto'].str.contains("ind_ctpp_fin_ult1"))].sort_values(by='ranking', ascending = False).head(10)

''' Total clientes 696539'''
df_max.ncodpers.value_counts()

'''Próximos productos más preferidos (cantidad de cientes que lo tiene como primero en el ranking)'''
df_max.groupby(['producto'])['ncodpers'].count().sort_values(ascending = False)

'''4 productos que no terminaron primero para ningun cliente: caja de ahorro (“ind_ahor_fin_ult1”), garantías (“ind_aval_fin_ult1”), cuentas derivadas (“ind_cder_fin_ult1”) y depósitos a mediano plazo (“ind_deme_fin_ult1”).'''
df_max.producto.unique()  

del predic,df_pred,df_pred2,df_max

''' MODELO LISTWISE '''

params = {'seed':123,'max_depth':8, 'eta':1, 'eval_metric':'map', 
          'min_child_weight':300, 'objective':'rank:map'}

xgb_listwise_map = xgb.train(params, train_dmatrix, num_boost_round=100,
                      early_stopping_rounds= 10,
                      evals=[(valid_dmatrix, 'validation')])


'''TEST MAP ''' # 0.86813
pred = xgb_listwise_map.predict(test_dmatrix, ntree_limit = xgb_pairwise.best_ntree_limit)
X = df_test
X['y_real'] = df_test.compro
X['pred'] = pred
g=df_test.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

'''TRAIN MAP ''' #0.8666 
pred = xgb_listwise_map.predict(train_dmatrix, ntree_limit = xgb_listwise_map.best_ntree_limit)
X = df_train
X['y_real'] = df_train.compro
X['pred'] = pred
g=df_train.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])

'''VALIDATION MAP''' #0.8657 
pred = xgb_listwise_map.predict(valid_dmatrix, ntree_limit = xgb_listwise_map.best_ntree_limit)
X = df_valid
X['y_real'] = df_valid.compro
X['pred'] = pred
g=df_valid.groupby('groupid')

np.mean([average_precision_score(r.y_real,r.pred) for i, r in g])