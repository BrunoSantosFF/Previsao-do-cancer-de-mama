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

    Returns:
    - None: imprime as métricas no console.
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
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nRelatório de Classificação:\n", classification_report(testdf[target], predictions))
    print("Matriz de Confusão:\n", confusion_matrix(testdf[target], predictions))
    plot_confusion_matrix(testdf,predictions,target)
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