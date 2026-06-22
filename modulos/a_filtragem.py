import numpy as np
import json
from numpy.lib.stride_tricks import sliding_window_view

def carregar_filtros(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def correlacao_2d(imagem: np.ndarray, filtro: list) -> np.ndarray:
    # converte o filtro carregado para array do numpy e define tamanho
    filtro_np = np.array(filtro, dtype=np.float64)
    f_alt, f_larg = filtro_np.shape
    pad_alt, pad_larg = f_alt // 2, f_larg // 2

    if imagem.ndim == 3:
        # padding nas bordas para tratar a perda de dimensão
        img_pad = np.pad(imagem, ((pad_alt, pad_alt), (pad_larg, pad_larg), (0, 0)), mode='edge')
        
        # técnica para janelamento deslizante vetorizado e eficiente no numpy
        janelas = sliding_window_view(img_pad, (f_alt, f_larg), axis=(0, 1))
        
        # multiplica a janela pelo filtro e soma nos dois eixos da janela (-2 e -1)
        img_out = np.sum(janelas * filtro_np, axis=(-2, -1))
        return img_out
    
    elif imagem.ndim == 2:
        img_pad = np.pad(imagem, ((pad_alt, pad_alt), (pad_larg, pad_larg)), mode='edge')
        janelas = sliding_window_view(img_pad, (f_alt, f_larg))
        img_out = np.sum(janelas * filtro_np, axis=(-2, -1))
        return img_out

    raise ValueError("A imagem deve ter 2 ou 3 dimensões.")

def rgb_para_cinza(imagem: np.ndarray) -> np.ndarray:
    return np.dot(imagem[..., :3], [0.2989, 0.5870, 0.1140])
