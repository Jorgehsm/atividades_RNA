import numpy as np

n_camadas = 4
weights = []
biases = []

# Carrega os pesos salvos
for i in range(1, n_camadas + 1):
    weights.append(np.load(f'model/W_{i}.npy'))
    biases.append(np.load(f'model/b_{i}.npy'))

# Carrega base de dados de teste
data = np.loadtxt('iris_test_data.csv', delimiter=',', dtype=str)

# Atributos
X = data[:, 0:4].astype(float)
labels = data[:, 4]

# Mapeia os nomes das classes de texto para índices numéricos (0, 1, 2)
classes = np.unique(labels)
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
target_raw = np.array([class_to_idx[l] for l in labels])

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

# Função para prever novas imagens (inferência)
def prever(entrada):
    activations = [entrada]
    for l in range(len(weights)):
        z = np.dot(activations[-1], weights[l]) + biases[l]
        
        # ReLU nas camadas ocultas, Softmax apenas na camada de saída
        if l == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)
        activations.append(a)
        
    return activations[-1]

output = prever(X)

# Probabilidades dos dígitos previstos
results = np.argmax(output, axis=1)

# 3. Compara os acertos
acertos = np.sum(results == target_raw)
total = target_raw.size
acuracia = (acertos / total) * 100

print(f'\n--- Resultados do Teste ---')
print(f'Total de dados de teste: {total}')
print(f'Total de acertos: {acertos}')
print(f'Acurácia do modelo: {acuracia:.2f}%')
print(f'Porcentagem de erro: {100 - acuracia:.2f}%')