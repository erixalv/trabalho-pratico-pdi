import numpy as np
import json
from math import *
import matplotlib.pyplot as plt

#função que lê o arquivo de filtros e carrega o JSON
def carregar_parametros(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)

def gerar_mascara_gabor(tamanho, sigma, lmbda, gamma, psi, theta_graus) -> np.ndarray:
    #calcular tamanho da máscara, variando de -half a half, com centro em (0,0)
    half = tamanho // 2
    y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float64)

    #conversão de graus para radianos
    theta = np.deg2rad(theta_graus)

    #cálculo de x_rot e y_rot 
    x_rot =  np.cos(theta) * x + np.sin(theta) * y
    y_rot = -np.sin(theta) * x + np.cos(theta) * y

    #calculo da gaussiana e senoidal para o filtro de gabor
    gaussiana = np.exp(-(x_rot**2 + (gamma**2) * (y_rot**2)) / (2 * sigma**2))
    senoidal  = np.cos(2 * np.pi * x_rot / lmbda + psi)

    #resultado do filtro de gabor
    return gaussiana * senoidal


def gerar_banco(json_path: str) -> list[tuple[float, np.ndarray]]:
    p = carregar_parametros(json_path=json_path)
    return [
        (theta, gerar_mascara_gabor(
        p["tamanho_mascara"],
        p["sigma"],
        p["lmbda"],
        p["gamma"],
        p["psi"],
        theta
    )) for theta in p["orientacoes_graus"]]

def visualizar_banco(banco: str):
    n = len(banco)
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))

    for ax, (theta, mascara) in zip(axes, banco):
        # expansão de histograma para visualização (exigida pela especificação)
        mn, mx = mascara.min(), mascara.max()
        img_vis = (mascara - mn) / (mx - mn) * 255

        ax.imshow(img_vis, cmap='gray')
        ax.set_title(f'{theta}°', fontsize=9)
        ax.axis('off')

    plt.suptitle('Banco de filtros de Gabor', fontsize=11)
    plt.tight_layout()
    plt.savefig('banco_gabor.png', dpi=150)
    plt.show()

banco = gerar_banco('config/gabor_bank.json')
visualizar_banco(banco)