import numpy as np

# Carrega os pesos salvos
W1 = np.load('W1.npy')
b1 = np.load('b1.npy')
W2 = np.load('W2.npy')
b2 = np.load('b2.npy')

# Carrega dataset de teste
data = np.loadtxt('mnist_test.csv', delimiter=',')

dataset = data[:, 1:]
dataset = dataset / 255.0 # Normalizando dataset em 0 - 1
target_raw = data[:, 0].astype(int)

def f_active(v):
  v = np.clip(v, -500, 500)
  return 1.0 / (1.0 + np.exp(-v))

# Função para prever novas imagens (inferência)
def prever(entrada):
  z = f_active(np.dot(entrada, W1) + b1)
  y = f_active(np.dot(z, W2) + b2)
  return y  # Retorna as probabilidades para as 10 classes

output = prever(dataset)

# Probabilidades dos dígitos previstos
digitos_previstos = np.argmax(output, axis=1)

# 3. Compara os acertos
acertos = np.sum(digitos_previstos == target_raw)
total = target_raw.size
acuracia = (acertos / total) * 100

print(f'Total de imagens de teste: {total}')
print(f'Imagens acertadas: {acertos}')
print(f'Acurácia do modelo: {acuracia:.2f}%')
print(f'Porcentagem de erro: {100 - acuracia:.2f}%')