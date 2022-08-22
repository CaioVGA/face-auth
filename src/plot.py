import matplotlib.pyplot as plt

sample = [0, 200, 400, 600, 800, 1000, 1200]

percent = [0, 75.44, 78.21, 83.68, 86.31, 85.22, 84.81]

plt.xlabel('Amostras')
plt.ylabel('Precisão do Modelo (%)')
plt.title('Gráfico da acurácia pela quantidade de amostras com classificação dos olhos e boca.')
plt.plot(sample, percent)
plt.show()