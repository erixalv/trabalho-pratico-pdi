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
