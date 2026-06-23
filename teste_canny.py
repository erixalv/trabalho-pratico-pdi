from PIL import Image
import numpy as np
import modulos.canny_tradicional as canny
import matplotlib.pyplot as plt

#testar com GrayAndMagenta -> não detecta bordas!!
imagem_rgb = np.array(Image.open("imagens/Zebra.png"))

bordas = canny.canny_tradicional(
    imagem_rgb,
    t_high=100,
    t_low=50,
    caminho_filtros="config/filtros.json",
)

plt.imshow(bordas, cmap='gray')
plt.savefig("resultado_canny_tradicional.png")
plt.show()