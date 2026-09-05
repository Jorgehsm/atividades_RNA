# Quarta atividade de RNA
# Utilizar vários neurônios Perceptron para classificação

# Faremos uma rede capaz de classificar entre os dígitos (0-9) (Resolução 4x7)

# Para rodar use:
# pip install numpy streamlit pandas streamlit-drawable-canvas pillow
# streamlit run perceptron_digitos.py

import numpy as np
import streamlit as st
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw

# 28 inputs -> 28 pesos

# Ground Truth 
zero = np.array([[-1,  1,  1, -1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [-1,  1,  1, -1]])

one = np.array([[-1, -1,  1, -1],
                [-1,  1,  1, -1],
                [-1, -1,  1, -1],
                [-1, -1,  1, -1],
                [-1, -1,  1, -1],
                [-1, -1,  1, -1],
                [-1,  1,  1,  1]])

two = np.array([[-1,  1,  1, -1],
                [ 1, -1, -1,  1],
                [-1, -1, -1,  1],
                [-1,  1,  1, -1],
                [ 1, -1, -1, -1],
                [ 1, -1, -1, -1],
                [ 1,  1,  1,  1]])

three = np.array([[-1,  1,  1, -1],
                  [ 1, -1, -1,  1],
                  [-1, -1, -1,  1],
                  [-1,  1,  1, -1],
                  [-1, -1, -1,  1],
                  [ 1, -1, -1,  1],
                  [-1,  1,  1, -1]])

four = np.array([[ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [ 1,  1,  1,  1],
                 [-1, -1, -1,  1],
                 [-1, -1, -1,  1],
                 [-1, -1, -1,  1]])

five = np.array([[ 1,  1,  1,  1],
                 [ 1, -1, -1, -1],
                 [ 1, -1, -1, -1],
                 [ 1,  1,  1, -1],
                 [-1, -1, -1,  1],
                 [-1, -1, -1,  1],
                 [ 1,  1,  1,  1]])

six = np.array([[-1,  1,  1, -1],
                [ 1, -1, -1, -1],
                [ 1, -1, -1, -1],
                [ 1,  1,  1, -1],
                [ 1, -1, -1,  1],
                [ 1, -1, -1,  1],
                [-1,  1,  1, -1]])

seven = np.array([[ 1,  1,  1,  1],
                  [-1, -1, -1,  1],
                  [-1, -1,  1, -1],
                  [-1, -1,  1, -1],
                  [-1,  1, -1, -1],
                  [-1,  1, -1, -1],
                  [-1,  1, -1, -1]])

eight = np.array([[-1,  1,  1, -1],
                  [ 1, -1, -1,  1],
                  [ 1, -1, -1,  1],
                  [-1,  1,  1, -1],
                  [ 1, -1, -1,  1],
                  [ 1, -1, -1,  1],
                  [-1,  1,  1, -1]])

nine = np.array([[-1,  1,  1, -1],
                 [ 1, -1, -1,  1],
                 [ 1, -1, -1,  1],
                 [-1,  1,  1,  1],
                 [-1, -1, -1,  1],
                 [-1, -1, -1,  1],
                 [ 1,  1,  1,  1]])


dataset = np.array([zero.ravel(), one.ravel(), two.ravel(), three.ravel(), four.ravel(), five.ravel(), six.ravel(), seven.ravel(), eight.ravel(), nine.ravel()])
target = np.eye(10, dtype=int) * 2 - 1 # Cria a matriz target com linhas -1 e casa 1 para o dígito de interesse

# Inicializa pesos e bias como zero:
W = np.zeros((10, 28))
biases = np.zeros(10)
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
                w = w + lr * entrada_i * erro
                bias = bias + lr * erro
                erros_na_epoca += 1

        if erros_na_epoca == 0:
            print(f"Treinamento convergido com sucesso em {epoca} épocas!")
            break

        #print(erros_na_epoca)
        epoca += 1

    return w, bias # Retorna pesos e bias novo para cada época

for i in range(len(target)):
    print(f"--- Treinando neurônio para o dígito {i} ---")
    # Treina o neurônio 'i' usando a coluna correspondente da matriz target
    W[i], biases[i] = treino(dataset, target[i], W[i], biases[i], learning_rate)

print("\nTreinamento finalizado!")
print(f"Dimensões da matriz de pesos W: {W.shape}")
print(f"Dimensões do vetor de biases: {biases.shape}")

# Interface para input do número
st.title("Reconhecedor de Dígitos por Perceptron (4x7)")
st.write("Clique nos blocos para desenhar o dígito e clique em **Send**.")

# Desenha grid
canvas_width = 200
canvas_height = 350

grid_image = Image.new(
    "RGB",
    (canvas_width, canvas_height),
    "black"
)

draw = ImageDraw.Draw(grid_image)

# Linhas verticais
for x in range(0, canvas_width + 1, canvas_width // 4):
    draw.line(
        [(x, 0), (x, canvas_height)],
        fill=(80, 80, 80),
        width=1
    )

# Linhas horizontais
for y in range(0, canvas_height + 1, canvas_height // 7):
    draw.line(
        [(0, y), (canvas_width, y)],
        fill=(80, 80, 80),
        width=1
    )

# Cria o componente de canvas para desenho à mão livre
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=35,
    stroke_color="#FFFFFF",

    background_image=grid_image,

    width=canvas_width,
    height=canvas_height,

    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas_desenho",
)

# Botão para enviar e processar o desenho
if st.button("Send", type="primary"):
  if canvas_result.image_data is not None:
    from PIL import Image

    # 1. Pega a imagem gerada no canvas (array RGBA)
    img_array = canvas_result.image_data.astype(np.uint8)

    # 2. Converte para imagem PIL e redimensiona exatamente para 4x7 pixels
    img = Image.fromarray(img_array).convert("L")  # Converte para escala de cinza
    img_resized = img.resize((4, 7), Image.Resampling.BOX)

    # 3. Transforma em numpy array e binariza para 1 (traço) e -1 (fundo)
    arr = np.array(img_resized)
    # Se o pixel estiver mais claro que o fundo, vira 1, senão -1
    vetor_entrada = np.where(arr > 128, 1, -1).ravel()

    st.write("Matriz 4x7 gerada pelo seu desenho:")
    st.write(vetor_entrada.reshape(7, 4))

    # Faz a predição com os 10 neurônios treinados
    ativacoes = np.dot(W, vetor_entrada) + biases
    digito_previsto = np.argmax(ativacoes)

    st.success(f"O Perceptron identificou o dígito: **{digito_previsto}**")
  else:
    st.warning("Por favor, desenhe um número antes de enviar.")