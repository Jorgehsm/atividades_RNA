# Terceira atividade de RNA
# Utilizar Perceptron para classificação

# Faremos um neurônio capaz de classificar entre as letras T e X (Resolução 5x5)

import numpy as np

# 25 inputs -> 25 pesos

# Ground Truth 
X = np.array([[ 1, -1, -1, -1,  1],
              [-1,  1, -1,  1, -1],
              [-1, -1,  1, -1, -1],
              [-1,  1, -1,  1, -1],
              [ 1, -1, -1, -1,  1]])

T = np.array([[ 1,  1,  1,  1,  1],
              [-1, -1,  1, -1, -1],
              [-1, -1,  1, -1, -1],
              [-1, -1,  1, -1, -1],
              [-1, -1,  1, -1, -1]])

# X = 1, T = -1
dataset = np.array([X.ravel(), T.ravel()])
target = np.array([1, -1])

# Dataset de testes
def addRuido(M, taxa_de_ruido):
    print("Letra original")
    print(M.reshape(X.shape))
    n_afetados = int(len(M) * taxa_de_ruido)
    indices = np.random.choice(len(M), n_afetados, replace = False)

    # Inverte os n pixel selecionados
    M[indices] *= -1

    print("\nLetra com ruído")
    print(M.reshape(X.shape))
    return M

# Inicializa pesos e bias como zero:
w = np.zeros(25);
bias = 0;
learning_rate = 1

def treino(entrada, target, w, bias, lr):
    epoca = 0
    max_epoca = 100

    while epoca < max_epoca:
        erros_na_epoca = 0

        for i in range(len(entrada)):
            entrada_i = entrada[i]
            target_i = target[i]
            y_input = np.dot(entrada_i, w) + bias

            # Função de ativação
            if(y_input >= 0): y = 1
            else: y = -1

            if(y != target_i):
                erro = target_i - y
                w = w + learning_rate * entrada_i * erro
                bias = bias + learning_rate * erro
                erros_na_epoca += 1

        if erros_na_epoca == 0:
            print(f"Treinamento convergido com sucesso em {epoca} épocas!")
            break

        #print(erros_na_epoca)
        epoca += 1

    return w, bias # Retorna pesos e bias novo para cada época

w, bias = treino(dataset, target, w, bias, learning_rate)

print("\nPesos finais:", w)
print("Bias final:", bias)

# Testando o neurônio treinado
print("\n--- Testes ---")
for i, nome in enumerate(["X", "T"]):
    entrada_teste = addRuido(dataset[i], 0.3)
    y_input = np.dot(entrada_teste, w) + bias

    # Função de ativação
    if(y_input >= 0): saida = 1
    else: saida = -1

    print(f"Letra testada: {nome} | Saída do neurônio: {saida} (Esperado: {target[i]})")