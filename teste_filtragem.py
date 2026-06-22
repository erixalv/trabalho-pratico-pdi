import numpy as np
import os
from modulos.a_filtragem import carregar_filtros, correlacao_2d, rgb_para_cinza
from utils.io_imagens import salvar_imagem, abrir_imagem

def criar_imagem_sintetica(caminho: str) -> np.ndarray:
    # cria uma imagem preta de 150x150 com um quadrado branco e retângulos para testar bordas
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    img[40:110, 40:110, :] = 255  # Quadrado central
    img[10:30, 10:140, :] = 128   # Faixa cinza horizontal
    salvar_imagem(caminho, img)
    return img

def main():
    print("Iniciando testes do Módulo A - Filtragem e IO de Imagens...")
    
    arquivo_teste = "teste_input.png"
    if not os.path.exists(arquivo_teste):
        print(f"[{arquivo_teste}] não encontrada. Criando imagem sintética...")
        criar_imagem_sintetica(arquivo_teste)
    
    print("\nLendo a imagem...")
    imagem_rgb = abrir_imagem(arquivo_teste)
    
    print("Carregando matrizes de filtros JSON...")
    filtros = carregar_filtros("config/filtros.json")
    sobel_h = filtros["sobel_horizontal"]
    sobel_v = filtros["sobel_vertical"]
    gaussiana = filtros["gaussiana_5x5"]
    
    # 1. Testar Correlação RGB com filtro Gaussiano (importante normalizar o peso do filtro para não estourar os brancos)
    soma_gaussiana = sum(sum(linha) for linha in gaussiana)
    gauss_norm = [[v / soma_gaussiana for v in linha] for linha in gaussiana]
    
    print("Aplicando filtro Gaussiano (RGB)...")
    img_suavizada = correlacao_2d(imagem_rgb, gauss_norm)
    salvar_imagem("teste_resultado_gaussiana.png", img_suavizada)
    
    # 2. Testar Conversão para Escala de Cinza
    print("Convertendo imagem para escala de cinza...")
    img_cinza = rgb_para_cinza(imagem_rgb)
    salvar_imagem("teste_resultado_cinza.png", img_cinza)
    
    # 3. Testar Filtro Sobel (Geralmente aplicado na escala de cinza para detectar bordas)
    print("Aplicando filtro Sobel Horizontal e Vertical...")
    img_sobel_h = correlacao_2d(img_cinza, sobel_h)
    img_sobel_v = correlacao_2d(img_cinza, sobel_v)
    
    # A convolução/correlação de Sobel gera valores negativos, aplicamos valor absoluto para visualizar em imagem
    salvar_imagem("teste_resultado_sobel_h.png", np.abs(img_sobel_h))
    salvar_imagem("teste_resultado_sobel_v.png", np.abs(img_sobel_v))
    
    print("\nTestes concluídos com sucesso! Arquivos gerados:")
    print("- teste_input.png")
    print("- teste_resultado_gaussiana.png")
    print("- teste_resultado_cinza.png")
    print("- teste_resultado_sobel_h.png")
    print("- teste_resultado_sobel_v.png")

if __name__ == "__main__":
    main()
