from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from modulos.canny_tradicional import canny_tradicional

imagem_rgb = np.array(Image.open("imagens/VintageCar.png").convert("RGB"))

bordas = canny_tradicional(
    imagem_rgb,
    t_high=100,
    t_low=50,
    caminho_filtros="config/filtros.json"
)

plt.imshow(bordas, cmap='gray')
plt.title("Canny tradicional — Bear")
plt.savefig("resultado_gray_magenta.png")
plt.show()