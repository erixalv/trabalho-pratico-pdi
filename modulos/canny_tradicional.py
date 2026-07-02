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

    pixels_altos = np.argwhere(magnitude > 100)
    print(f"pixels com magnitude > 100: {len(pixels_altos)}")
    if len(pixels_altos) > 0:
        i, j = pixels_altos[0]
        print(f"  exemplo: ({i},{j}) mag={magnitude[i,j]:.2f} ang={angulo_quantizado[i,j]}")
        print(f"  vizinho cima ({i-1},{j}): {magnitude[i-1,j]:.2f}")
        print(f"  vizinho baixo ({i+1},{j}): {magnitude[i+1,j]:.2f}")
        

    for i in range(H):
        for j in range(W):
            ang = angulo_quantizado[i, j]

            if ang == 0:
                v1 = magnitude[i, j-1] if j > 0 else 0
                v2 = magnitude[i, j+1] if j < W-1 else 0
            elif ang == 45:
                v1 = magnitude[i-1, j+1] if i > 0 and j < W-1 else 0
                v2 = magnitude[i+1, j-1] if i < H-1 and j > 0 else 0
            elif ang == 90:
                v1 = magnitude[i-1, j] if i > 0 else 0
                v2 = magnitude[i+1, j] if i < H-1 else 0
            elif ang == 135:
                v1 = magnitude[i-1, j-1] if i > 0 and j > 0 else 0
                v2 = magnitude[i+1, j+1] if i < H-1 and j < W-1 else 0
            else:
                continue

            if magnitude[i, j] >= v1 and magnitude[i, j] >= v2:
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
    print(f"cinza min: {cinza.min():.2f}, max: {cinza.max():.2f}")

    suavizada = suavizar(cinza, caminho_filtros)
    print(f"suavizada min: {suavizada.min():.2f}, max: {suavizada.max():.2f}")

    magnitude, angulo_q = calcular_gradiente(suavizada, caminho_filtros)
    print(f"magnitude min: {magnitude.min():.2f}, max: {magnitude.max():.2f}")
    print(f"angulos unicos: {np.unique(angulo_q)}")

    mag_nms = nms(magnitude, angulo_q)
    print(f"mag_nms min: {mag_nms.min():.2f}, max: {mag_nms.max():.2f}")
    print(f"pixels nao-zero no nms: {np.count_nonzero(mag_nms)}")

    bordas = histerese(mag_nms, t_high, t_low)
    print(f"pixels de borda: {np.count_nonzero(bordas)}")
    print(f"t_high={t_high}, t_low={t_low}")

    return bordas