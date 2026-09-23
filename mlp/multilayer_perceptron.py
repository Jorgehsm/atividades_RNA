# Sexta atividade de RNA
# Treinar uma rede multicamada (MLP) para aproximar uma função a partir dos pontos amostrados

# Para isso utilizaremos uma rede com uma camada oculta e vamos comparar o erro quadrático utilizando 4, 8 e 16 neurônios na camada oculta

# A camada de entrada possui um neurônio, para receber a variável X

import numpy as np
import math
import matplotlib.pyplot as plt

x = np.arange(0, 1.1, 0.1)
y = np.array([-0.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5])

# Reshape para vetor coluna
X = x.reshape(-1, 1)  # Shape: (11, 1)
Y = y.reshape(-1, 1)  # Shape: (11, 1)

learning_rate = 0.08
n_neuronios_ocultos = [4, 8, 16]

def f_active(v):
    return 2.0 / (1.0 + np.exp(-v)) - 1.0

def df_active(v):
    out = f_active(v)
    return 0.5 * (1.0 + out) * (1.0 - out)

def treino(entrada, target, n_neuronios_ocultos, lr):
    epoca = 0
    max_epoca = 10000
    input_size = entrada.shape[1]
    output_size = target.shape[1]

    historico_mse = []

    # Inicia pesos com valor aleatório e bias nulo
    np.random.seed(42) # Permitir reprodução dos resultados
    W1 = np.random.randn(input_size, n_neuronios_ocultos)*0.01
    b1 = np.zeros((1, n_neuronios_ocultos))
    W2 = np.random.randn(n_neuronios_ocultos, output_size)*0.01
    b2 = np.zeros((1, output_size))

    while epoca < max_epoca:
        z_input = np.dot(entrada, W1) + b1 # Input da camada oculta
        z = f_active(z_input) # Output da camada oculta

        y_input = np.dot(z, W2) + b2 # Input da camada de entrada
        y = y_input # Ativação linear 

        # Erro quadrático médio (MSE)
        erro = y - target
        mse = np.mean(erro ** 2)
        historico_mse.append(mse)

        # Retropropagação
        delta_output = erro # Derivada da ativação linear é 1
 
        erro_camada_oculta = np.dot(delta_output, W2.T)
        delta_camada_oculta = erro_camada_oculta * df_active(z_input)

        # Gradients
        dW2 = np.dot(z.T, erro)
        db2 = np.sum(delta_output, axis=0, keepdims=True)
        
        dW1 = np.dot(entrada.T, delta_camada_oculta)
        db1 = np.sum(delta_camada_oculta, axis=0, keepdims=True)

        # Gradient Descent Update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        epoca += 1

        if epoca % 500 == 0:
            print(f"Época {epoca}/{max_epoca} - MSE: {mse:.6f}")

    print(f"Pesos iniciais da camada oculta: {W1}")
    print(f"Pesos iniciais da camada de entrada: {W2.T}")
    return W1, b1, W2, b2, y, historico_mse

resultados = {}
historicos_mse = {}

for n_neuronios in n_neuronios_ocultos:
    print(f"\n--- Treinando com {n_neuronios} neurônios na camada oculta ---")
    W1_res, b1_res, W2_res, b2_res, resultado_aprox, hist = treino(X, Y, n_neuronios, learning_rate)
    resultados[n_neuronios] = resultado_aprox
    historicos_mse[n_neuronios] = hist

print("\nTreinamento completo!")
y_4neuronios = resultados[4].flatten()
y_8neuronios = resultados[8].flatten()
y_16neuronios = resultados[16].flatten()

# --- Aproximação da Função ---
# Dados reais
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='black', label='Dados Originais', zorder=5, s=50)

plt.plot(x, y_4neuronios, label='4 Neurônios', linestyle='--')
plt.plot(x, y_8neuronios, label='8 Neurônios', linestyle='-.')
plt.plot(x, y_16neuronios, label='16 Neurônios', linestyle='-')

plt.title('Comparação do Número de Neurônios na Camada Oculta')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

graph1 = f'result_lr_{learning_rate}.png'
plt.savefig(graph1, dpi=300, bbox_inches='tight')

# --- Evolução do MSE por Época ---
plt.figure(figsize=(10, 5))
plt.plot(historicos_mse[4], label='4 Neurônios', linestyle='--')
plt.plot(historicos_mse[8], label='8 Neurônios', linestyle='-.')
plt.plot(historicos_mse[16], label='16 Neurônios', linestyle='-')
plt.title('Evolução do Erro Quadrático Médio (MSE) durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.yscale('log')

graph2 = f'mse_lr_{learning_rate}.png'
plt.savefig(graph2, dpi=300, bbox_inches='tight')

plt.show()