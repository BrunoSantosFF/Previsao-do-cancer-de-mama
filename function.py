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

def classification_model(model, traindf, testdf, predictors, target):
    """
    Treina o modelo, faz previsões e calcula as métricas de avaliação.

    Parameters:
    - model: modelo de classificação a ser treinado.
    - traindf: DataFrame de treinamento.
    - testdf: DataFrame de teste.
    - predictors: lista de nomes das colunas usadas como preditores.
    - target: nome da coluna alvo.

    """
    # Treinamento do modelo
    model.fit(traindf[predictors], traindf[target])
    
    # Previsões no conjunto de teste
    predictions = model.predict(testdf[predictors])
    # Cálculo das métricas
    accuracy = accuracy_score(testdf[target], predictions)
    precision = precision_score(testdf[target], predictions)  
    recall = recall_score(testdf[target], predictions)        
    f1 = f1_score(testdf[target], predictions)              

    # Exibição das métricas
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Acurácia: {100*accuracy:.2f}%")
    print(f"Precisão: {100*precision:.2f}%")
    print(f"Recall: {100*recall:.2f}%")
    print(f"F1 Score: {100*f1:.2f}%")
    #print("\nRelatório de Classificação:\n", classification_report(testdf[target], predictions))
    print("Matriz de Confusão:\n", confusion_matrix(testdf[target], predictions))
    print("=" * 50)
    
def classification_model_with_cv(model, traindf, testdf, predictors, target):
    """
    Avalia o modelo usando validação cruzada com o conjunto de treinamento e também calcula as métricas no conjunto de teste.

    Parameters:
    - model: modelo de classificação a ser avaliado.
    - traindf: DataFrame de treinamento.
    - testdf: DataFrame de teste.
    - predictors: lista de nomes das colunas usadas como preditores.
    - target: nome da coluna alvo.
    """
    # Ajustar o modelo no conjunto de treinamento
    model.fit(traindf[predictors], traindf[target])
    
    # Validação cruzada para o conjunto de treinamento
    cv_accuracy = cross_val_score(model, traindf[predictors], traindf[target], cv=5, scoring='accuracy')
    cv_precision = cross_val_score(model, traindf[predictors], traindf[target], cv=5, scoring='precision')
    cv_recall = cross_val_score(model, traindf[predictors], traindf[target], cv=5, scoring='recall')
    cv_f1 = cross_val_score(model, traindf[predictors], traindf[target], cv=5, scoring='f1')

    # Previsões no conjunto de teste
    predictions = model.predict(testdf[predictors])
    
    # Cálculo das métricas no conjunto de teste
    accuracy_test = accuracy_score(testdf[target], predictions)
    precision_test = precision_score(testdf[target], predictions)
    recall_test = recall_score(testdf[target], predictions)
    f1_test = f1_score(testdf[target], predictions)

    # Exibição das métricas
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Média da Acurácia na Validação Cruzada: {100 * cv_accuracy.mean():.2f}%")
    print(f"Média da Precisão na Validação Cruzada: {100 * cv_precision.mean():.2f}%")
    print(f"Média do Recall na Validação Cruzada: {100 * cv_recall.mean():.2f}%")
    print(f"Média do F1 Score na Validação Cruzada: {100 * cv_f1.mean():.2f}%")
    
    print("\nMétricas no Conjunto de Teste:")
    print(f"Acurácia: {100 * accuracy_test:.2f}%")
    print(f"Precisão: {100 * precision_test:.2f}%")
    print(f"Recall: {100 * recall_test:.2f}%")
    print(f"F1 Score: {100 * f1_test:.2f}%")
    
    print("\nMatriz de Confusão no Conjunto de Teste:\n", confusion_matrix(testdf[target], predictions))
    print("=" * 50)

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