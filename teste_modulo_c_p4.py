"""
Teste do Módulo C (Parte 2) — Pessoa 4
Executa o pipeline completo do Canny Modificado Gabor-Di Zenzo
e salva imagens intermediárias para validação visual.
"""
import numpy as np
import os
from utils.io_imagens import abrir_imagem, salvar_imagem
from modulos.c_canny_mod import canny_modificado, magnitude_e_orientacao, nms_direcional

# Parâmetros
CAMINHO_BANCO = "config/gabor_bank.json"

# Diretório de saída
os.makedirs("resultados_p4", exist_ok=True)

# Imagens de teste (excluímos as muito grandes para evitar estouro de memória no Módulo A)
imagens = [
    "imagens/GrayAndMagenta.png",
    "imagens/FCBarcelona.png",
    "imagens/VintageCar.png",
]

def expandir_histograma(mapa):
    """Expansão de histograma para visualização de mapas com valores fora de [0,255]."""
    mn, mx = mapa.min(), mapa.max()
    if mx - mn == 0:
        return np.zeros_like(mapa, dtype=np.uint8)
    return ((mapa - mn) / (mx - mn) * 255).astype(np.uint8)

for caminho in imagens:
    if not os.path.exists(caminho):
        print(f"[SKIP] {caminho} não encontrado")
        continue

    nome = os.path.splitext(os.path.basename(caminho))[0]
    print(f"\n{'='*60}")
    print(f"Processando: {nome}")
    print(f"{'='*60}")

    img = abrir_imagem(caminho)
    if img is None:
        print(f"[ERRO] Não foi possível abrir {caminho}")
        continue

    print(f"  Dimensões: {img.shape}")

    # Etapa 1: obter magnitude e orientação (Pessoa 3)
    magnitude_final, orientacao_final = magnitude_e_orientacao(img, CAMINHO_BANCO)

    print(f"  Magnitude — min: {magnitude_final.min():.2f}, max: {magnitude_final.max():.2f}")
    print(f"  Magnitude — média: {magnitude_final.mean():.2f}, mediana: {np.median(magnitude_final):.2f}")

    # Calcular limiares adaptativos baseados no percentil da magnitude
    t_high = np.percentile(magnitude_final, 85)
    t_low = np.percentile(magnitude_final, 60)
    print(f"  Limiares adaptativos — T_high: {t_high:.2f} (p85), T_low: {t_low:.2f} (p60)")

    # Etapa 2: NMS (Pessoa 4)
    mag_nms = nms_direcional(magnitude_final, orientacao_final)

    # Etapa 3: Histerese (Pessoa 4)
    from modulos.c_canny_mod import histerese
    bordas = histerese(mag_nms, t_high, t_low)

    # Salvar resultados intermediários
    salvar_imagem(f"resultados_p4/{nome}_magnitude.png", expandir_histograma(magnitude_final))
    salvar_imagem(f"resultados_p4/{nome}_nms.png", expandir_histograma(mag_nms))
    salvar_imagem(f"resultados_p4/{nome}_bordas.png", bordas)

    # Estatísticas
    print(f"  Pós-NMS   — pixels não-zero: {np.count_nonzero(mag_nms)} / {mag_nms.size}")
    print(f"  Bordas    — pixels de borda: {np.count_nonzero(bordas)} / {bordas.size}")
    print(f"  Resultados salvos em resultados_p4/{nome}_*.png")

print("\n✓ Teste completo!")
