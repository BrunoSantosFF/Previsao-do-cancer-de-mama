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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


from imblearn.over_sampling import SMOTE

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

# Definindo as features a serem utilizadas (através dos graficos)
predictor = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean']

traindf, testdf = train_test_split(df, test_size=0.3, random_state=42, stratify=df['diagnosis'])

# Separando as features e o target do conjunto de treino
X_train = traindf[predictor]
y_train = traindf['diagnosis']
X_test = testdf[predictor]
y_test = testdf['diagnosis']

# Balanceando os dados com SMOTE
oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train)

#Plotando gráfico pizza para saber a porcentagem de maligno e benigno
#plot_graphic_pie(y_train,y_train_resh)
#statify_preview(df,traindf,testdf)

##================## Modelo de Classificação ##================##

# Criando pipelines para cada modelo
#Random Forest	
rf_pipeline = Pipeline(steps = [('scale',StandardScaler()),('classifier',RandomForestClassifier(random_state=42))])
#Regressão Logística	
logreg_pipeline = Pipeline(steps = [('scale',StandardScaler()),('classifier',LogisticRegression(random_state=42))])
#Naive Bayes Gaussiano	
gaussianNB_pipeline = Pipeline(steps = [('scale',StandardScaler()),('classifier',GaussianNB())])
#K-Nearest Neighbors (KNN)	
knn_pipeline = Pipeline(steps = [('scale',StandardScaler()),('classifier',KNeighborsClassifier(n_neighbors=5))])
#Árvore de Decisão	
decision_tree_pipeline = Pipeline(steps=[('scale', StandardScaler()),('classifier', DecisionTreeClassifier(random_state=42))])
#Máquina de Vetores de Suporte (SVM)	
svm_pipeline = Pipeline(steps=[('scale', StandardScaler()),('classifier', SVC(probability=True, random_state=42))])

