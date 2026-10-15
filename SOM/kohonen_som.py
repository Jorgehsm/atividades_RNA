# Nona atividade de RNA
# implementar um Mapa Auto-Organizável de Kohonen (SOM - Self-Organizing Map) unidimensional. 

# O objetivo é criar uma rede neural não supervisionada capaz de aprender a topologia de um conjunto de dados bidimensional
# (pontos dentro de um círculo) e representá-lo através de uma "cadeia" de 50 neurônios interligados.

import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Criar dataset 
# (100 pontos de treinamento restritos a uma região circular de raio 0.5)
num_pontos = 5000
X1, X2 = [], []

def dataset():
    while len(X1) < num_pontos:
        x1 = np.random.uniform(-0.5, 0.5)
        x2 = np.random.uniform(-0.5, 0.5)

        if x1**2 + x2**2 < 0.25:
            X1.append(x1)
            X2.append(x2)
    return np.array(X1), np.array(X2)

# Inicializando pesos
W1 = np.random.uniform(-1, 1, size=num_pontos)
W2 = np.random.uniform(-1, 1, size=num_pontos)

# Plot dos resultados
def plot_rede(W1, W2, epoca, final=False):
    plt.clf()
    plt.plot(W1, W2, 'o-', color='blue', label='Neurônios (SOM)')
    plt.title(f'SOM - Self-Organizing Map - Época {epoca}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.7, 0.7)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    if final:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.5)

# Treinamento
def treino(X1, X2, W1, W2, max_epoca, lr):
    epoca = 0
    d = np.zeros(num_pontos)

    for epoca in range(max_epoca):
        for i in range(len(X1)):
            x_sample = X1[i]
            y_sample = X2[i]

            d = np.sqrt((x_sample - W1)**2 + (y_sample- W2)**2)
        
            # Encontra o índice do neurônio vencedor (BMU)
            bmu = np.argmin(d)

            # Atualização dos pesos na casa do bmu e seus vizinhos imediatos
            W1[bmu] += lr * (x_sample - W1[bmu])
            W2[bmu] += lr * (y_sample - W2[bmu])

            if bmu - 1 >= 0:
                W1[bmu-1] += lr * (x_sample - W1[bmu-1])
                W2[bmu-1] += lr * (y_sample - W2[bmu-1])

            if bmu + 1 < num_pontos:
                W1[bmu+1] += lr * (x_sample - W1[bmu+1])
                W2[bmu+1] += lr * (y_sample - W2[bmu+1])
            # Atualizando Learning Rate
            lr = 0.5 - (0.5 - 0.01)/99*epoca

        if epoca % 10 == 0 or epoca == 0:
            plot_rede(W1, W2, epoca, final=False)
    
    plt.ioff()
    plot_rede(W1, W2, epoca+1, final=True)
    return

dataset()
treino(X1, X2, W1, W2, max_epoca=100, lr=0.5)