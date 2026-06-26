import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from modulos.c_canny_mod import magnitude_e_orientacao

imagem_rgb = np.array(Image.open("imagens/Zebra.png").convert("RGB"))
print(f"Imagem carregada: {imagem_rgb.shape}")

print("Aplicando banco de Gabor nos 3 canais e calculando magnitude e orientação...")
magnitude, orientacao = magnitude_e_orientacao(imagem_rgb, "config/gabor_bank.json")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(imagem_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(magnitude, cmap="gray")
axes[1].set_title("Magnitude Final")
axes[1].axis("off")

axes[2].imshow(orientacao, cmap="hsv")
axes[2].set_title("Orientacao Final (graus)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("teste_resultado_modulo_c.png", dpi=150)
plt.show()

print(f"Magnitude — min: {magnitude.min():.2f}, max: {magnitude.max():.2f}")
print(f"Orientacoes presentes: {np.unique(orientacao)}")
print("Resultado salvo em teste_resultado_modulo_c.png")
