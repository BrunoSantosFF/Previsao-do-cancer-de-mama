##================## Importando bibliotecas ##================##
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import mpld3 as mpl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

##================## Lendo o database ##================##
df = pd.read_csv("database/data.csv",header = 0)
#imprimindo uma parte dos dados do DB
#print(df.head())

##================## Limpando e preparando os dados ##================##
#eliminando a coluna de Id euma coluna vazia no CSV
df = df.drop(columns=['id', 'Unnamed: 32'])
#substitui valores categóricos por valores numéricos na coluna diagnosis
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

#fornece informações sobre a distribuição e principais medidas de tendência central e dispersão
#print(df.describe())

#Imprime a quantidade de tumores malignos e benignos
# print("Quantidade de Maligno: ", np.bincount(df['diagnosis'])[0])
# print("Quantidade de Benigno: ", np.bincount(df['diagnosis'])[1])

#Pegando os primeiros 10 atributos do database e adicionando em uma lista
features_mean=list(df.columns[1:11])

#Passando para um novo dataframe, porém separando os tumores malignos dos benignos
dfM=df[df['diagnosis'] ==1]
dfB=df[df['diagnosis'] ==0]
