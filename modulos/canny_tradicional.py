from modulos.a_filtragem import correlacao_2d, carregar_filtros, rgb_para_cinza
import numpy as np

def suavizar(imagem_gray, caminho_filtros):
    filtros = carregar_filtros(caminho_filtros)
    mascara_gaussiana = np.array(filtros["gaussiana_5x5"], dtype=np.float64)

    #normaliza a máscara
    mascara_gaussiana = mascara_gaussiana / mascara_gaussiana.sum()

    return correlacao_2d(imagem_gray, mascara_gaussiana)

def calcular_gradiente(imagem_suavizada, caminho_filtros):
    filtros = carregar_filtros(caminho_filtros)

    gradiente_vertical   = correlacao_2d(imagem_suavizada, filtros["sobel_horizontal"])
    gradiente_horizontal = correlacao_2d(imagem_suavizada, filtros["sobel_vertical"])

    magnitude = np.sqrt(gradiente_vertical**2 + gradiente_horizontal**2)
    angulo    = np.rad2deg(np.arctan2(gradiente_vertical, gradiente_horizontal))

    #mapeia o array e cada angulo que for menor que zero, é somado 180
    angulo[angulo < 0] += 180

    angulo_q = np.zeros_like(angulo)
    angulo_q[(angulo < 22.5) | (angulo >= 157.5)] = 0
    angulo_q[(angulo >= 22.5) & (angulo < 67.5)] = 45
    angulo_q[(angulo >= 67.5) & (angulo < 112.5)] = 90
    angulo_q[(angulo >= 127.5) & (angulo  < 157.5)] = 135

    return magnitude, angulo_q

def nms(magnitude: np.ndarray, angulo_quantizado: np.ndarray):
    H, W = magnitude.shape
    saida = np.zeros_like(magnitude)

    for i in range(1, H-1):
        for j in range(1, W-1):
            ang = angulo_quantizado[i, j]

            if ang == 0:
                vizinho_um = magnitude[i, j-1]
                vizinho_dois = magnitude[i, j+1]
            elif ang == 45:
                vizinho_um = magnitude[i-1, j+1]
                vizinho_dois = magnitude[i+1, j-1]
            elif ang == 90:
                vizinho_um = magnitude[i-1, j]
                vizinho_dois = magnitude[i+1, j]
            elif ang == 135:
                vizinho_um = magnitude[i-1, j-1]
                vizinho_dois = magnitude[i+1, j+1]

            elif magnitude[i, j] >= vizinho_um and magnitude[i, j] >= vizinho_dois:
                saida[i, j] = magnitude[i, j]

    return saida

def histerese(magnitude_nms: np.ndarray, t_high: float, t_low: float):  
    forte = (magnitude_nms >= t_high)
    fraca = (magnitude_nms >= t_low) & (magnitude_nms < t_high)

    saida = np.zeros_like(magnitude_nms, dtype=np.uint8)
    saida[forte] = 255

    from collections import deque
    fila = deque(zip(*np.where(forte)))

    while fila:
        i, j = fila.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i+di, j+dj
                if 0 <= ni < saida.shape[0] and 0 <= nj < saida.shape[1]:
                    if fraca[ni, nj] and saida[ni, nj] == 0:
                        saida[ni, nj] = 255
                        fila.append((ni, nj))

    return saida

def canny_tradicional(imagem_rgb, t_high, t_low, caminho_filtros):
    cinza = rgb_para_cinza(imagem_rgb)
    suavizada = suavizar(cinza, caminho_filtros)
    magnitude, angulo_q = calcular_gradiente(suavizada, caminho_filtros)
    mag_nms = nms(magnitude, angulo_q)
    bordas = histerese(mag_nms, t_high, t_low)
    return bordas