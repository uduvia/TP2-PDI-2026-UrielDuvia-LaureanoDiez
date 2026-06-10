import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definimos función para mostrar imágenes (conservada de tu estructura)
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)

# ==============================================================================
# --- PARTE A: Detección y Segmentación de la Placa Patente --------------------
# ==============================================================================

# 1. Leemos la imagen original
img_path = 'Patentes/img_6.jpg' # Reemplazar por la ruta correcta
img = cv2.imread(img_path)
if img is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imshow(img_rgb, title="1. Imagen Original", color_img=True)

# 2. Resaltar zonas de alto contraste (donde hay texto) usando Gradiente Morfológico
# Esto nos ayuda a destacar los bordes de los caracteres de la patente.
kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
gradiente = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel_grad)

# Umbralizamos el gradiente para quedarnos con los bordes más fuertes
_, grad_th = cv2.threshold(gradiente, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
imshow(grad_th, title="2. Gradiente Morfológico + Otsu")

# 3. Agrupación (Clausura Morfológica)
# Usamos un kernel rectangular ancho para conectar los caracteres entre sí y 
# formar un bloque sólido que represente la patente completa.
kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
clausura = cv2.morphologyEx(grad_th, cv2.MORPH_CLOSE, kernel_clausura)
imshow(clausura, title="3. Clausura (Agrupación de caracteres)")

# 4. Detección de Contornos y Filtrado Geométrico
# Buscamos los contornos externos del bloque generado
contours, _ = cv2.findContours(clausura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contours = img_rgb.copy()
patente_roi = None
coord_patente = None

# Dimensiones oficiales Mercosur: 400mm ancho x 130mm alto.
# Aspect Ratio teórico = 400/130 ≈ 3.07. 
# Damos un margen de tolerancia por la deformación de la perspectiva.
ratio_min, ratio_max = 2.2, 4.5 
area_min = 1000 # Evitamos ruido pequeño

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    aspect_ratio = float(w) / h
    
    if area > area_min and (ratio_min < aspect_ratio < ratio_max):
        # 5. Verificación por Color (Franja Azul)
        # Extraemos la región candidata en HSV para buscar el azul
        candidato_hsv = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        
        # Rango para el color azul (ajustable según iluminación)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(candidato_hsv, lower_blue, upper_blue)
        
        # Si hay una cantidad razonable de píxeles azules en la parte superior, ¡es la patente!
        blue_pixels = cv2.countNonZero(mask_blue)
        if blue_pixels > (area * 0.05): # Al menos 5% del área debe ser azul
            cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 3)
            patente_roi = img_gray[y:y+h, x:x+w]
            patente_roi_rgb = img_rgb[y:y+h, x:x+w]
            coord_patente = (x, y, w, h)
            break # Encontramos la patente, salimos del loop

imshow(img_contours, title="4. Placa Patente Detectada", color_img=True)

# ==============================================================================
# --- PARTE B: Segmentación de Caracteres --------------------------------------
# ==============================================================================

if patente_roi is not None:
    # 6. Umbralización Adaptativa (Ayuda #2 del TP)
    # Lidia con reflejos o sombras en la chapa calculando umbrales locales.
    # Invertimos para que las letras negras queden en blanco (255) sobre fondo negro (0).
    th_adapt_roi = cv2.adaptiveThreshold(
        patente_roi, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        19, 5
    )
    
    plt.figure()
    ax1 = plt.subplot(121)
    imshow(patente_roi, new_fig=False, title="5. ROI Patente (Grises)")
    plt.subplot(122, sharex=ax1, sharey=ax1)
    imshow(th_adapt_roi, new_fig=False, title="6. ROI Umbral Adaptativo")
    plt.show(block=False)

    # 7. Componentes conectadas para aislar caracteres
    # Usamos stats para filtrar por geometría de letra
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        th_adapt_roi, connectivity=8, ltype=cv2.CV_32S
    )
    
    roi_caracteres = patente_roi_rgb.copy()
    caracteres_detectados = []
    
    # Altura de la patente ROI
    h_placa = patente_roi.shape[0]
    
    for i in range(1, num_labels): # Ignoramos el 0 (fondo)
        x_c = stats[i, cv2.CC_STAT_LEFT]
        y_c = stats[i, cv2.CC_STAT_TOP]
        w_c = stats[i, cv2.CC_STAT_WIDTH]
        h_c = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Filtros geométricos para caracteres oficiales:
        # Altura letras: ~65mm. Altura placa: 130mm. (La letra ocupa ~50% de la altura)
        # Ratio letras (w/h): ~45/65 = 0.69. Ratio números: ~40/65 = 0.61.
        char_aspect_ratio = float(w_c) / h_c
        
        # Condición 1: El caracter debe ser más alto que ancho (0.2 < ratio < 0.9)
        # Condición 2: Debe ocupar una altura significativa de la chapa (> 35% y < 90%)
        if (0.2 < char_aspect_ratio < 0.9) and (h_placa * 0.35 < h_c < h_placa * 0.90):
            caracteres_detectados.append((x_c, y_c, w_c, h_c))
            
    # Ordenamos los caracteres de izquierda a derecha según su coordenada X
    caracteres_detectados = sorted(caracteres_detectados, key=lambda c: c[0])
    
    # Dibujamos y mostramos resultados por consola
    print(f"\n--- Resultados de Segmentación ---")
    print(f"Patente detectada en ROI: x={coord_patente[0]}, y={coord_patente[1]}")
    print(f"Cantidad de caracteres encontrados: {len(caracteres_detectados)}\n")
    
    for idx, (xc, yc, wc, hc) in enumerate(caracteres_detectados):
        print(f"Caracter {idx+1}: Posición X={xc}, Ancho={wc}, Alto={hc}")
        cv2.rectangle(roi_caracteres, (xc, yc), (xc+wc, yc+hc), (255, 0, 0), 2)
        cv2.putText(roi_caracteres, str(idx+1), (xc, yc-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
    imshow(roi_caracteres, title="7. Caracteres Segmentados", color_img=True)

else:
    print("No se logró detectar ninguna patente que cumpla con las proporciones.")