# Quinta atividade de RNA
# Treinar neurônio Adeline usando a base de dados fornecida 

# Plotar erro quadradático total durante treinamento
# Testar rede neural treinada

import numpy as np
import matplotlib.pyplot as plt

# Importa base de dados
data = np.loadtxt('database.csv', delimiter=',', skiprows=1, unpack=True)

dataset = data[:2, :].T
target = data[2]

dataset_treino = dataset[:-10]
target_treino = target[:-10]

dataset_teste = dataset[-10:]
target_teste = target[-10:]

# Inicializa pesos com valor aleatório pequeno:
rng = np.random.default_rng()
W = rng.random(2)
bias = 0
learning_rate = 0.1

print(W)

# # y: entrada
# # t: target

def treino(x, target, w, bias, lr):
    epoca = 0
    max_epoca = 20
    historico_erro = []

    while epoca < max_epoca:
        erro_na_epoca = 0

        for i in range(len(x)):
            x_i = x[i]
            target_i = target[i]
            y_input = np.dot(x_i, w) + bias
            
            erro = target_i - y_input

            w += lr * x_i * erro
            bias += lr * erro

            erro_na_epoca += 0.5 * (erro ** 2)

        if erro_na_epoca == 0:
            print(f"Treinamento convergido com sucesso em {epoca} épocas!")
            break
        historico_erro.append(erro_na_epoca)
        epoca += 1

    return w, bias, historico_erro # Retorna pesos e bias novo para cada época

 #Executa treinamento
W, bias, hist = treino(dataset_treino, target_treino, W, bias, learning_rate)
plt.plot(hist)
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Total')
plt.title('Evolução do Erro no Treinamento do ADALINE')
plt.grid(True)
plt.savefig('erro_treinamento.png')

print("\nTreinamento finalizado!")

# Teste
print("\n--- Testando a rede neural (Test Set) ---")
for i in range(len(dataset_treino)):
    x_i = dataset_treino[i]
    target_i = target_treino[i]
    
    y_input = np.dot(x_i, W) + bias
    prediction = 1 if y_input >= 0 else -1
    
    print(f"Sample {i}: Target = {int(target_i)} | Predição = {prediction} | Saída linear = {y_input:.3f}")

--- Testando a rede neural (Test Set) ---
Sample 0: Target = -1   | Predição = -1   | Saída linear = -0.900
Sample 1: Target = 1    | Predição = 1     | Saída linear = 0.717
Sample 2: Target = 1    | Predição = 1     | Saída linear = 1.473
Sample 3: Target = -1   | Predição = -1   | Saída linear = -1.711
Sample 4: Target = -1   | Predição = -1   | Saída linear = -1.293
Sample 5: Target = 1    | Predição = 1     | Saída linear = 0.445
Sample 6: Target = -1   | Predição = -1   | Saída linear = -0.641
Sample 7: Target = 1    | Predição = 1     | Saída linear = 0.772
Sample 8: Target = -1   | Predição = -1   | Saída linear = -0.859
Sample 9: Target = 1    | Predição = 1     | Saída linear = 0.847