# Importações necessárias
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

def hyperparameter_tuning_and_evaluation(pipeline, param_grid, X_train, y_train, X_test, y_test, model_name):
    """
    Ajusta os hiperparâmetros de um modelo e avalia seu desempenho.

    :param pipeline: Pipeline do modelo a ser ajustado.
    :param param_grid: Grade de hiperparâmetros a serem testados.
    :param X_train: Conjunto de dados de treino.
    :param y_train: Rótulos do conjunto de dados de treino.
    :param X_test: Conjunto de dados de teste.
    :param y_test: Rótulos do conjunto de dados de teste.
    :param model_name: Nome do modelo (para exibir nos resultados).
    :return: Predições do modelo ajustado e o objeto GridSearch.
    """
    # Ajustando o modelo com GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # Fazendo predições
    predictions = grid_search.predict(X_test)
    
    # Retorna as predições e o objeto grid_search
    return predictions, grid_search

def print_model_evaluation(rf_f1, knn_f1, svm_f1, rf_cm, knn_cm, svm_cm, 
                            y_test, rf_pred, knn_pred, svm_pred):
    print('Mean f1 scores:')
    print('RF mean :', rf_f1)
    print('KNN mean :', knn_f1)
    print('SVM mean :', svm_f1)

    print("======== Random Forest ========")
    print(rf_cm)
    print(classification_report(y_test, rf_pred))
    print('Accuracy Score: ', accuracy_score(y_test, rf_pred))

    print("======== KNN ========")
    print(knn_cm)
    print(classification_report(y_test, knn_pred))
    print('Accuracy Score: ', accuracy_score(y_test, knn_pred))

    print("======== SVM ========")
    print(svm_cm)
    print(classification_report(y_test, svm_pred))
    print('Accuracy Score: ', accuracy_score(y_test, svm_pred))


def evaluate_models(models, X_train, y_train, scoring_metrics):
    for name_model, model in models.items():
        print(f"Modelo: {name_model}")
        for metric in scoring_metrics:
            val = cross_val_score(model, X_train, y_train, cv=5, scoring=metric)
            print(f'{metric}: {100 * val.mean():.2f}')
        print("==" * 30)


def plot_confusion_matrix(cm):
    """
    Plota a matriz de confusão para as previsões feitas.
    """
    # Matriz de Confusão com Rótulos
    labels = ['Benigno (0)', 'Maligno (1)']
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    cm_display.plot(cmap=plt.cm.Blues)  # Usando o cmap para uma boa visualização
    plt.title('Matriz de Confusão')
    plt.show()  # Mostrar a matriz de confusão plotada

def plot_attribute_graphs(features_mean, dfM, dfB):
  # Definir o tamanho do gráfico
  plt.figure(figsize=(15, 15))

  # Loop pelos atributos e criar um subplot para cada um
  for i, feature in enumerate(features_mean):
      plt.subplot(5, 2, i+1)  # 5 linhas, 2 colunas de subplots
      plt.hist(dfM[feature], bins=30, alpha=0.5, label='Maligno', color='red')  # Histograma para tumores malignos
      plt.hist(dfB[feature], bins=30, alpha=0.5, label='Benigno', color='blue')  # Histograma para tumores benignos
      plt.title(f'{feature}')
      plt.legend()

  # Ajustar o layout para não sobrepor os gráficos
  plt.tight_layout()

  # Mostrar os gráficos
  plt.show()

#Observe os grafico que é mais envolvido com os tumores malignos
#radius, perimeter, area, compactness, concavity and concave points

def statify_preview(df,traindf,testdf):
  # Verificando a proporção das classes
  print("Proporção no Conjunto Completo:")
  print(df['diagnosis'].value_counts(normalize=True))

  print("\nProporção no Conjunto de Treinamento:")
  print(traindf['diagnosis'].value_counts(normalize=True))

  print("\nProporção no Conjunto de Teste:")
  print(testdf['diagnosis'].value_counts(normalize=True))

def plot_graphic_pie(y_train, y_train_resh):
  # Contando a quantidade de cada classe
    labels = ['Benigno (0)', 'Maligno (1)']
    original_counts = [sum(y_train == 0), sum(y_train == 1)]
    reshaped_counts = [sum(y_train_resh == 0), sum(y_train_resh == 1)]

    # Criando o gráfico de pizza para o conjunto original
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, 1ª posição
    plt.pie(original_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuição Original das Classes')
    plt.axis('equal')  # Para garantir que o gráfico de pizza seja circular

    # Criando o gráfico de pizza para o conjunto balanceado
    plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, 2ª posição
    plt.pie(reshaped_counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuição das Classes Após SMOTE')
    plt.axis('equal')  # Para garantir que o gráfico de pizza seja circular

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()

# Função para calcular métricas de desempenho (sem ROC AUC)
def calculate_metrics(y_true, y_pred):
    return {
        'F1': f1_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred)
    }
