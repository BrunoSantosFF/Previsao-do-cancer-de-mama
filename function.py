import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score

# Importações necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,ConfusionMatrixDisplay


def plot_confusion_matrix(testdf, predictions, target):
    """
    Plota a matriz de confusão para as previsões feitas.
    """
    # Matriz de Confusão com Rótulos
    labels = ['Benigno (0)', 'Maligno (1)']
    cm = confusion_matrix(testdf[target], predictions)
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
