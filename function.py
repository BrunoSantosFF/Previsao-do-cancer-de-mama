import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score



def classification_model(model, data, predictors, outcome):
    # Treina o modelo
    model.fit(data[predictors], data[outcome])
    
    # Faz previsões no conjunto de treinamento
    predictions = model.predict(data[predictors])
    
    # Imprime a acurácia no conjunto de treinamento
    accuracy = accuracy_score(data[outcome], predictions)
    print("Acurácia no conjunto de treinamento: %s" % "{0:.3%}".format(accuracy))

    # Realiza a validação cruzada k-fold com 5 dobras
    cross_val_scores = cross_val_score(model, data[predictors], data[outcome], cv=5)

    # Imprime as pontuações de validação cruzada
    print("Pontuações de Validação Cruzada: ", cross_val_scores)
    print("Média das Pontuações de Validação Cruzada: %s" % "{0:.3%}".format(cross_val_scores.mean()))

    # Ajusta o modelo novamente para que possa ser referenciado fora da função
    model.fit(data[predictors], data[outcome])



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