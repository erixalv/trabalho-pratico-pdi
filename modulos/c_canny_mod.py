import numpy as np
from modulos.a_filtragem import correlacao_2d
from modulos.b_gabor import gerar_banco


def aplicar_gabor_rgb(imagem_rgb: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    R = correlacao_2d(imagem_rgb[:, :, 0], mascara)
    G = correlacao_2d(imagem_rgb[:, :, 1], mascara)
    B = correlacao_2d(imagem_rgb[:, :, 2], mascara)
    return np.sqrt(R**2 + G**2 + B**2)


def filtrar_por_orientacao(imagem_rgb: np.ndarray, banco: list) -> list[tuple[float, np.ndarray]]:
    return [(theta, aplicar_gabor_rgb(imagem_rgb, mascara)) for theta, mascara in banco]


def reducao_por_maximo(mapas: list[tuple[float, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    thetas = np.array([theta for theta, _ in mapas])
    magnitudes = np.stack([mag for _, mag in mapas], axis=0)  # (N, H, W)

    indice_max = np.argmax(magnitudes, axis=0)    # (H, W) — índice da orientação vencedora
    magnitude_final = np.max(magnitudes, axis=0)  # (H, W)
    orientacao_final = thetas[indice_max]          # (H, W) — ângulo em graus

    return magnitude_final, orientacao_final


def magnitude_e_orientacao(imagem_rgb: np.ndarray, caminho_banco: str) -> tuple[np.ndarray, np.ndarray]:
    banco = gerar_banco(caminho_banco)
    mapas = filtrar_por_orientacao(imagem_rgb, banco)
    return reducao_por_maximo(mapas)


def nms_direcional(magnitude_final: np.ndarray,
                   orientacao_final: np.ndarray) -> np.ndarray:
    """
    Supressão de Não-Máximos adaptada ao caso vetorial (Gabor-Di Zenzo).

    A orientação final (Θ) vinda do banco de Gabor indica a DIREÇÃO DA BORDA
    (paralela a ela). Para o NMS, precisamos comparar os vizinhos na direção
    PERPENDICULAR à borda (direção do gradiente), ou seja, Θ + 90°.

    A perpendicular é quantizada em 4 direções de vizinhança, pois a grade de
    pixels possui apenas 8 vizinhos:
        0°   → vizinhos (i, j-1) e (i, j+1)        — horizontal
        45°  → vizinhos (i-1, j+1) e (i+1, j-1)    — diagonal ↗↙
        90°  → vizinhos (i-1, j) e (i+1, j)         — vertical
        135° → vizinhos (i-1, j-1) e (i+1, j+1)     — diagonal ↖↘

    Garante afinamento para exatamente 1 pixel de largura.
    """
    H, W = magnitude_final.shape
    saida = np.zeros_like(magnitude_final)

    # Calcula a direção do gradiente (perpendicular à borda)
    grad_direcao = orientacao_final % 180.0

    # Quantiza em 4 direções: 0, 45, 90, 135
    angulo_q = np.zeros_like(grad_direcao)
    angulo_q[(grad_direcao >= 0)    & (grad_direcao < 22.5)]  = 0
    angulo_q[(grad_direcao >= 22.5) & (grad_direcao < 67.5)]  = 45
    angulo_q[(grad_direcao >= 67.5) & (grad_direcao < 112.5)] = 90
    angulo_q[(grad_direcao >= 112.5) & (grad_direcao < 157.5)] = 135
    angulo_q[(grad_direcao >= 157.5)] = 0  # 157.5–180° ≈ 0°

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            ang = angulo_q[i, j]

            if ang == 0:
                vizinho_um = magnitude_final[i, j - 1]
                vizinho_dois = magnitude_final[i, j + 1]
            elif ang == 45:
                vizinho_um = magnitude_final[i - 1, j + 1]
                vizinho_dois = magnitude_final[i + 1, j - 1]
            elif ang == 90:
                vizinho_um = magnitude_final[i - 1, j]
                vizinho_dois = magnitude_final[i + 1, j]
            else:  # 135
                vizinho_um = magnitude_final[i - 1, j - 1]
                vizinho_dois = magnitude_final[i + 1, j + 1]

            if magnitude_final[i, j] > vizinho_um and magnitude_final[i, j] > vizinho_dois:
                saida[i, j] = magnitude_final[i, j]

    return saida


def histerese(magnitude_nms: np.ndarray,
              t_high: float,
              t_low: float) -> np.ndarray:
    """
    Limiarização por histerese com análise de conectividade (BFS).

    1. Pixels com magnitude >= t_high → bordas fortes (definitivas).
    2. Pixels com t_low <= magnitude < t_high → bordas fracas (candidatas).
    3. Pixels com magnitude < t_low → suprimidos.

    A partir dos pixels fortes, propaga-se via BFS (vizinhança 8-conectada)
    para promover pixels fracos adjacentes a fortes.

    Retorna mapa binário: 255 (borda) ou 0 (fundo).
    """
    forte = (magnitude_nms >= t_high)
    fraca = (magnitude_nms >= t_low) & (magnitude_nms < t_high)

    saida = np.zeros_like(magnitude_nms, dtype=np.uint8)
    saida[forte] = 255

    # BFS a partir dos pixels fortes para promover vizinhos fracos
    from collections import deque
    fila = deque(zip(*np.where(forte)))

    while fila:
        i, j = fila.popleft()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < saida.shape[0] and 0 <= nj < saida.shape[1]:
                    if fraca[ni, nj] and saida[ni, nj] == 0:
                        saida[ni, nj] = 255
                        fila.append((ni, nj))

    return saida


def canny_modificado(imagem_rgb: np.ndarray,
                     caminho_banco: str,
                     t_high: float,
                     t_low: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pipeline completa do Canny Modificado Gabor-Di Zenzo.

    Encadeia:
        1. Filtragem Gabor por orientação + fusão L2 Di Zenzo (Pessoa 3)
        2. Supressão de Não-Máximos direcional (Pessoa 4)
        3. Limiarização por histerese (Pessoa 4)

    Retorna:
        - bordas: mapa binário final (0 ou 255)
        - magnitude_final: mapa de magnitudes antes do NMS (para visualização)
        - mag_nms: mapa de magnitudes após NMS (para visualização)
    """
    magnitude_final, orientacao_final = magnitude_e_orientacao(imagem_rgb, caminho_banco)

    mag_nms = nms_direcional(magnitude_final, orientacao_final)

    bordas = histerese(mag_nms, t_high, t_low)

    return bordas, magnitude_final, mag_nms
