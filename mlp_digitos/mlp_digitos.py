# Sétima atividade de RNA
# Treinar uma rede multicamada (MLP) a partir do dataset MNIST para detectar dígitos

# Para comparar com resultados de outros trabalhos treinaremos uma rede com uma camada oculta com 784 neurônios de entrada 
# (Um para cada valor de pixel com resolução de 28x28), 800 neurônios na camada oculta e 10 saídas

# A função de ativação utilizada foi sigmóide

# Para treinar a rede, basta baixar mnist_train.csv de https://github.com/phoebetronic/mnist

import numpy as np
import math
import matplotlib.pyplot as plt

# Importa base de dados MNIST
data = np.loadtxt('mnist_train.csv', delimiter=',')

dataset = data[:, 1:]
dataset = dataset / 255.0 # Normalizando dataset em 0 - 1
target_raw = data[:, 0].astype(int)

# Converter o target para One-Hot Encoding (10 classes: 0 a 9)
num_classes = 10
target = np.zeros((target_raw.size, num_classes))
target[np.arange(target_raw.size), target_raw] = 1.0 # Converte, por exemplo, a linha com valor 5 para [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0]

learning_rate = 0.5
n_neuronios = 800

def f_active(v):
  # Sigmóide tradicional [0, 1]
  v = np.clip(v, -500, 500) # Limita os valores de v para evitar overflow no np.exp()
  return 1.0 / (1.0 + np.exp(-v))

def df_active(v):
  # Derivada da sigmóide tradicional: y * (1 - y)
  out = f_active(v)
  return out * (1.0 - out)

def treino(entrada, target, n_neuronios_ocultos, lr):
    epoca = 0
    max_epoca = 500
    input_size = entrada.shape[1]
    output_size = target.shape[1]
    batch_size = 64 # Quebra os dados em mini-lotes

    historico_mse = []

    # Inicia pesos com valor aleatório e bias nulo
    np.random.seed(42) # Permitir reprodução dos resultados
    W1 = np.random.randn(input_size, n_neuronios_ocultos)*np.sqrt(1.0 / input_size)
    b1 = np.zeros((1, n_neuronios_ocultos))
    W2 = np.random.randn(n_neuronios_ocultos, output_size)*np.sqrt(1.0 / input_size)
    b2 = np.zeros((1, output_size))

    n_amostras = entrada.shape[0]

    while epoca < max_epoca:
        # Embaralha os dados a cada época para ajudar no aprendizado
        indices = np.random.permutation(n_amostras)
        entrada_shuf = entrada[indices]
        target_shuf = target[indices]

        mse_epocas = []

        # Loop de Mini-Batches
        for i in range(0, n_amostras, batch_size):
            X_batch = entrada_shuf[i:i+batch_size]
            y_batch = target_shuf[i:i+batch_size]

            # Forward pass
            z_input = np.dot(X_batch, W1) + b1 
            z = f_active(z_input) 

            y_input = np.dot(z, W2) + b2 
            y = f_active(y_input) 

            # Erro do batch
            erro = y - y_batch
            mse_batch = np.mean(erro ** 2)
            mse_epocas.append(mse_batch)

            # Retropropagação
            delta_output = erro * df_active(y_input)
            erro_camada_oculta = np.dot(delta_output, W2.T)
            delta_camada_oculta = erro_camada_oculta * df_active(z_input)

            # Gradientes normalizados pelo tamanho do batch
            dW2 = np.dot(z.T, delta_output) / X_batch.shape[0]
            db2 = np.sum(delta_output, axis=0, keepdims=True) / X_batch.shape[0]
            
            dW1 = np.dot(X_batch.T, delta_camada_oculta) / X_batch.shape[0]
            db1 = np.sum(delta_camada_oculta, axis=0, keepdims=True) / X_batch.shape[0]

            # Atualização dos pesos
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Média do MSE da época inteira
        mse_medio = sum(mse_epocas) / len(mse_epocas)
        historico_mse.append(mse_medio)

        epoca += 1
        if epoca == 1 or epoca % 10 == 0:
            print(f"Época {epoca}/{max_epoca} - MSE: {mse_medio:.6f}")

    return W1, b1, W2, b2, y, historico_mse


print(f"\n--- Treinando com {n_neuronios} neurônios na camada oculta ---")
W1_res, b1_res, W2_res, b2_res, resultado_aprox, hist = treino(dataset, target, n_neuronios, learning_rate)


print("\nTreinamento completo!")
# Salvando pesos resultantes
np.save("W1.npy", W1_res)
np.save("b1.npy", b1_res)
np.save("W2.npy", W2_res)
np.save("b2.npy", b2_res)

# --- Teste da rede neural ---

# --- Evolução do MSE por Época ---
plt.figure(figsize=(10, 5))
plt.plot(hist, label='800 Neurônios', linestyle='--')
plt.title('Evolução do Erro Quadrático Médio (MSE) durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.yscale('log')

graph2 = f'mse_lr_{learning_rate}.png'
plt.savefig(graph2, dpi=300, bbox_inches='tight')

plt.show()