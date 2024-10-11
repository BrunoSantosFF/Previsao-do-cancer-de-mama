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
import seaborn as sns

#models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

from function import plot_attribute_graphs,statify_preview,plot_graphic_pie,evaluate_models,print_model_evaluation,plot_confusion_matrix,hyperparameter_tuning_and_evaluation


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

##================## Sepando os dados e features ##================##

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

#Array com as metricas
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

#dicionario de modelos
models = {
    'Random Forest': rf_pipeline,
    'Logistic Regression': logreg_pipeline,
    'Gaussian Naive Bayes': gaussianNB_pipeline,
    'K-Nearest Neighbors': knn_pipeline,
    'Decision Tree': decision_tree_pipeline,
    'Support Vector Machine': svm_pipeline
}

##================## Conjunto Treinamento ##================##

#Função para fazer a validação cruzada
#evaluate_models(models, X_train_resh, y_train_resh, scoring_metrics)

##================## Conjunto Teste ##================##

##Foi feito o teste para dados balanceados(SMOTE) e dados desbalanceados(dados originais)
## Os dados balanceados apresentaram melhores resultados
##Todos os modelos foram bons, porém quero dar foco nos que tiveram o melhor recall (que é o objetivo) pois o objetivo é acertar o maximo te tumores malignos, são eles :Random Forest, K-Nearest Neighbors, Support Vector Machine

#Agora iremos usar o conjunto Teste para relacionar
rf_pipeline.fit(X_train_resh,y_train_resh)
knn_pipeline.fit(X_train_resh,y_train_resh)
svm_pipeline.fit(X_train_resh,y_train_resh)

#Fazendo a predição
rf_pred  = rf_pipeline.predict(X_test)
knn_pred = knn_pipeline.predict(X_test)
svm_pred = svm_pipeline.predict(X_test)

#Plotando a matriz de confusão
rf_cm  = confusion_matrix(y_test,rf_pred)
knn_cm = confusion_matrix(y_test,knn_pred)
svm_cm = confusion_matrix(y_test,svm_pred)

#Imprimindo F1_score
rf_f1  = f1_score(y_test,rf_pred)
knn_f1 = f1_score(y_test,knn_pred)
svm_f1 = f1_score(y_test,svm_pred)

# Chame a função com os dados necessários
#print_model_evaluation(rf_f1, knn_f1, svm_f1, rf_cm, knn_cm, svm_cm, y_test, rf_pred, knn_pred, svm_pred)

#plotar matriz de confusão mais interativamente
# plot_confusion_matrix(rf_cm)
# plot_confusion_matrix(knn_cm)
# plot_confusion_matrix(svm_cm)

##================## Ajuste de hiperparametros ##================##
## Ajustando parametro para ver se conseguimos melhorar o modelo
# Hiperparâmetros para Random Forest
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Hiperparâmetros para K-Nearest Neighbors
param_grid_knn = {
    'classifier__n_neighbors': [3, 5, 7, 9],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan']
}

# Hiperparâmetros para Support Vector Machine
param_grid_svm = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__kernel': ['linear', 'rbf']
}

# Ajustando os modelos usando GridSearchCV
rf_pred, grid_rf = hyperparameter_tuning_and_evaluation(rf_pipeline, param_grid_rf, X_train_resh, y_train_resh, X_test, y_test, 'Random Forest')
knn_pred, grid_knn = hyperparameter_tuning_and_evaluation(knn_pipeline, param_grid_knn, X_train_resh, y_train_resh, X_test, y_test, 'K-Nearest Neighbors')
svm_pred, grid_svm = hyperparameter_tuning_and_evaluation(svm_pipeline, param_grid_svm, X_train_resh, y_train_resh, X_test, y_test, 'Support Vector Machine')

# Repetir o cálculo de F1 score e matriz de confusão
rf_f1 = f1_score(y_test, rf_pred)
knn_f1 = f1_score(y_test, knn_pred)
svm_f1 = f1_score(y_test, svm_pred)

# Chame a função com os dados necessários
print_model_evaluation(rf_f1, knn_f1, svm_f1, 
                       confusion_matrix(y_test, rf_pred), 
                       confusion_matrix(y_test, knn_pred), 
                       confusion_matrix(y_test, svm_pred), 
                       y_test, rf_pred, knn_pred, svm_pred)

##================## Comparação de modelos ##================##
