import cv2
import os
import numpy as np

# Diretório de entrada com as imagens originais
input_dir = 'img/'
# Diretório de saída para as imagens binarizadas
output_dir = 'res/'

# Função para medir a área da sola do pé em uma imagem binarizada
def medir_area(imagem_binaria):
    
    # Encontrar contornos na imagem binarizada
    contornos, _ = cv2.findContours(imagem_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    # Inicializar a área total
    area_total = 0
   
    # Iterar sobre todos os contornos
    for contorno in contornos:       
        # Calcular a área do contorno
        area = cv2.contourArea(contorno)
        # Filtrar contornos com base em uma área mínima
        if 100000 < area:
            area_total = area
    return area_total

# Função para pré-processar a imagem e focar na sola do pé
def preprocessar_imagem(imagem):

    # Assumindo que a régua ocupa os primeiros pixels da imagem
    # Cortamos o topo da imagem para deixar apenas o pé o máximo possível
    altura_corte = 210

    # Cortar a parte superior da imagem para remover a régua
    imagem_cortada = imagem[altura_corte:, :]

    # Converter a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem_cortada, cv2.COLOR_BGR2GRAY)
   
    # Aplicar binarização
    _, imagem_binaria = cv2.threshold(imagem_cinza, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Aplicar operações morfológicas para remover ruídos e pequenos detalhes
    kernel = np.ones((11,11), np.uint8)
    imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_CLOSE, kernel)
    imagem_binaria = cv2.morphologyEx(imagem_binaria, cv2.MORPH_OPEN, kernel)
    return imagem_binaria

# Lista todos os arquivos no diretório de entrada
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):

        # Caminho completo da imagem
        input_path = os.path.join(input_dir, filename)

        # Carregar a imagem
        imagemMatriz = cv2.imread(input_path)

        # Pré-processar a imagem
        imagem_binaria = preprocessar_imagem(imagemMatriz)

        # Medir a área da sola do pé
        area = medir_area(imagem_binaria)
        print(f'Área da sola do pé na imagem {filename} ~= {area} pixels')
        
        # Caminho completo para salvar a imagem binarizada
        output_path = os.path.join(output_dir, filename)
        
        # Salvar a imagem binarizada
        cv2.imwrite(output_path, imagem_binaria)
        print()


