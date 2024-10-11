##================## Importando bibliotecas ##================##
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec 
import mpld3 as mpl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#models
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from function import plot_attribute_graphs,statify_preview,classification_model,classification_model_with_cv,plot_graphic_pie


##================## Lendo o database ##================##
df = pd.read_csv("database/data.csv",header = 0)
#imprimindo uma parte dos dados do DB
#print(df.head())

##================## Limpando e preparando os dados ##================##
#eliminando a coluna de Id euma coluna vazia no CSV
df = df.drop(columns=['id', 'Unnamed: 32'])
#substitui valores categóricos por valores numéricos na coluna diagnosis
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

#verifica se existe alguma coluna com dados vazios
#print(df.isnull().sum())

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
#Quantidade de malignos e benignos
#print("Malignos: ",len(dfM))
#print("Benignos: ",len(dfB))

#plotando graficos para ter noção dos melhores atributos
#plot_attribute_graphs(features_mean,dfM,dfB)

##================## Treinando e Testando ##================##

traindf, testdf = train_test_split(df, test_size=0.3, random_state=42, stratify=df['diagnosis'])
#Plotando gráfico pizza para saber a porcentagem de maligno e benigno
plot_graphic_pie(traindf)

#statify_preview(df,traindf,testdf)

##================## Modelo de Classificação ##================##
#lista com atributos que foram escolhidos para terem relação com previsão
predictor = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']



# # Lista de modelos a serem testados
# models = [
#     LogisticRegression(max_iter=1000, random_state=42),
#     RandomForestClassifier(random_state=42),
#     DecisionTreeClassifier(random_state=42),
#     SVC(probability=True, random_state=42)  
# ]

# #treinamento e teste
# for model in models :
#     classification_model(model, traindf, testdf, predictor, "diagnosis")
#     #classification_model_with_cv(model, traindf, testdf, predictor, "diagnosis")
