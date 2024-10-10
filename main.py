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
#retornando um array com valores unicos da coluna "diagnosis"
print(df.diagnosis.unique())