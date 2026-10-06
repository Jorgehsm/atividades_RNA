# Oitava atividade de RNA
# Treinar uma rede multicamada (MLP) a partir do dataset Iris Dataset para detectar dígitos

# O dataset é composto por 3 classes, com 50 instâncias de cada, 4 com características.

# Utilizar função de ativação soft-max e implementar cross-entropy

import numpy as np
import math
import matplotlib.pyplot as plt
import os

os.makedirs("model", exist_ok=True)

# Importa base de dados
data = np.loadtxt('iris_data.csv', delimiter=',', dtype=str)

# Atributos
X = data[:, 0:4].astype(float)
labels = data[:, 4]

# Mapeia os nomes das classes de texto para índices numéricos (0, 1, 2)
classes = np.unique(labels)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
target_raw = np.array([class_to_idx[l] for l in labels])

# Converte o target para One-Hot Encoding
num_classes = len(classes)
target = np.zeros((target_raw.size, num_classes))
target[np.arange(target_raw.size), target_raw] = 1.0

# Hiparâmetros e Arquitetura
learning_rate = 0.05
n_neuronios = [20, 10, 5]
max_epoca = 1000

input_size = X.shape[1]
output_size = target.shape[1]

layer_sizes = [input_size] + n_neuronios + [output_size]

# Funções de Ativação
def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return (z > 0).astype(float)

def softmax(z):
    if z.ndim == 1:
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z)
    else:
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)

def treino(entrada, target, layer_sizes, lr, max_epoca):
    n_amostras = entrada.shape[0]
    historico_loss = []

    # Inicialização de pesos e vieses dinamicamente para N camadas
    np.random.seed(42)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        # Inicialização He/Xavier para ReLU
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]))
        weights.append(w)
        biases.append(b)

    epoca = 0
    while epoca < max_epoca:
        # --- Forward Pass ---
        activations = [entrada]
        zs = []
        
        for l in range(len(weights)):
            z = np.dot(activations[-1], weights[l]) + biases[l]
            zs.append(z)

            # Softmax na camada de saída, ReLU nas camadas ocultas
            if l == len(weights) - 1:
                a = softmax(z)
            else:
                a = relu(z)
            activations.append(a)

        y_pred = activations[-1]
          
        # Cross-Entropy Loss (com proteção contra log(0))
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss_medio = -np.mean(np.sum(target * np.log(y_pred_clipped), axis=1))
        historico_loss.append(loss_medio)

        # --- Backward Pass ---
        deltas = [None] * len(weights)
        # Gradiente da Cross-Entropy combinada com Softmax na camada de saída
        deltas[-1] = y_pred - target # Derivada da Cross-Entropy com a Softmax

        # Propaga o erro de trás para frente pelas camadas ocultas
        for l in range(len(weights) - 2, -1, -1):
            error = np.dot(deltas[l + 1], weights[l + 1].T)
            deltas[l] = error * d_relu(zs[l])

        # Gradientes e Atualização de Pesos (normalizados pelo total de amostras)
        for l in range(len(weights)):
            dW = np.dot(activations[l].T, deltas[l]) / n_amostras
            db = np.sum(deltas[l], axis=0, keepdims=True) / n_amostras

            weights[l] -= lr * dW
            biases[l] -= lr * db

        epoca += 1
        if epoca == 1 or epoca % 200 == 0:
            True
            print(f"Época {epoca}/{max_epoca} - Loss (Cross-Entropy): {loss_medio:.6f}")

    return weights, biases, historico_loss

print(f"\n--- Treinando MLP com arquitetura ocultas: {n_neuronios} ---")
weights_res, biases_res, hist = treino(X, target, layer_sizes, learning_rate, max_epoca)

print("\nTreinamento completo!")

# Salvando pesos resultantes de todas as camadas
for idx, w in enumerate(weights_res):
    np.save(f"model/W_{idx+1}.npy", w)
for idx, b in enumerate(biases_res):
    np.save(f"model/b_{idx+1}.npy", b)

# --- Evolução da Loss por Época ---
plt.figure(figsize=(10, 5))
plt.plot(hist, label=f'Arquitetura {n_neuronios}', color='blue')
plt.title('Evolução da Cross-Entropy Loss durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

graph_filename = f'model/loss_arquitetura_{"_".join(map(str, n_neuronios))}.png'
plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
plt.show()


--- Resultados do Teste ---
Total de dados de teste: 30
Total de acertos: 30
Acurácia do modelo: 100.00%
Porcentagem de erro: 0.00%