# Segunda atividade de RNA
# Jorge Henrique Marques
# Considere as 16 funções lógicas que podem ser construídas a partir de 2 variáveis.

# Mostre que a regra de Hebb pode encontrar os pesos das redes neurais correspondentes
# a 14 destas funções lógicas quando for usada a representação bipolar.

import numpy as np

# Entradas para função lógica [x1, x2, bias]
X = np.array([
    [-1, -1, 1],
    [-1,  1, 1],
    [ 1, -1, 1],
    [ 1,  1, 1]
])

# Nomes e descrições das 16 funções lógicas
funcs = [
    "f0: Contante Falso (-1)",
    "f1: AND (x1 AND x2)",
    "f2: Inibição de x2 (x1 AND NOT x2)",
    "f3: Transferência x1",
    "f4: Inibição de x1 (NOT x1 AND x2)",
    "f5: Transferência x2",
    "f6: XOR (x1 XOR x2)",
    "f7: OR (x1 OR x2)",
    "f8: NOR (NOT (x1 OR x2))",
    "f9: XNOR (Equivalência)",
    "f10: NOT x2",
    "f11: Implicação (x2 -> x1)",
    "f12: NOT x1",
    "f13: Implicação (x1 -> x2)",
    "f14: NAND (NOT (x1 AND x2))",
    "f15: Constante Verdadeiro (+1)"
]

def gerar_tabelas_verdade(): # Gera os vetores de saída target para as 16 funções em formato bipolar (-1, +1).
    tabelas_verdade = []
    for i in range(16):

        # Converte o índice i em uma string binária de 4 bits (0000, 0001, 0010, ...)
        # Isso cria a tabela verdade para cada uma das funções, na ordem da lista
        tabela_verdade = format(i, '04b')

        # Mapeia '0' -> -1 e '1' -> +1 (Representação bipolar)
        tabela_bipolar = np.array([1 if bit == '1' else -1 for bit in tabela_verdade])

        tabelas_verdade.append(tabela_bipolar)

    return np.array(tabelas_verdade)


def regra_hebb(X, y): # Treina os pesos [w1, w2, b] utilizando a Regra de Hebb (1 época).

    w = np.zeros(X.shape[1]) # Inicializa [w1, w2, b] com zeros: [0.0, 0.0, 0.0]
    for x_i, y_i in zip(X, y):
        w += x_i * y_i # Soma do produto da entrada pela saída esperada
    return w # novo peso

def prever(X, w): #Função de ativação degrau bipolar / sinal (sgn(net >= 0) -> +1, sgn(net < 0) -> -1)

    entrada = np.dot(X, w) # Produto escalar: x1*w1 + x2*w2 + 1*b

    # Função de ativação degrau bipolar (sgn)
    # Para cada elemento do array entrada, verifique se ele é maior ou igual a zero
    return np.where(entrada >= 0, 1, -1)  


# Teste para cada função lógica
tabelas = gerar_tabelas_verdade()

print("=== DEMONSTRAÇÃO DA REGRA DE HEBB BIPOLAR NAS 16 FUNÇÕES LÓGICAS ===\n")

sucesso = 0
falha = 0

for idx, (target, nome) in enumerate(zip(tabelas, funcs)):
    w = regra_hebb(X, target)
    y_pred = prever(X, w)
    correto = np.array_equal(y_pred, target)
    
    if correto:
        status = "SUCESSO"
        sucesso += 1
    else:
        status = "FALHOU"
        falha += 1
        
    print(f"Função {idx:2d} | {nome:<36} | Pesos [w1, w2, b]: {w} | Status: {status}")
    print(f"   Alvo:      {target.tolist()}")
    print(f"   Predição:  {y_pred.tolist()}")
    print("-" * 80)

print(f"\nRESUMO:")
print(f"Funções aprendidas com sucesso: {sucesso} / 16")
print(f"Funções que falharam:            {falha} / 16")