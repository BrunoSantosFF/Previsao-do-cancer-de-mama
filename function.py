import matplotlib.pyplot as plt 

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