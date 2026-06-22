import cv2
import numpy as np

def abrir_imagem(caminho: str) -> np.ndarray:
    img = cv2.imread(caminho)
    if img is not None:
        # PDI trabalha melhor com RGB em vez do BGR padrão do cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def salvar_imagem(caminho: str, imagem: np.ndarray):
    # restringe os limites para 0-255 antes de salvar
    img_salvar = np.clip(imagem, 0, 255).astype(np.uint8)
    
    if img_salvar.ndim == 3:
        img_salvar = cv2.cvtColor(img_salvar, cv2.COLOR_RGB2BGR)
        
    cv2.imwrite(caminho, img_salvar)
