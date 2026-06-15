"""
Trabajo Práctico II - PDI
Laureano Diez, Uriel Duvia

Junio 2026

Ejercicio 2

Se requiere detectar chapas patentes de forma automática a través de un script de python, dadas 
las imágenes de los autos.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================================
# Definimos función para mostrar imágenes
# ===============================================================================
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


for idx_img in range(1, 13): 
    filename = f'imagenes/img_{idx_img}.jpg'
    img = cv2.imread(filename)
    
    if img is None:
        print(f"No se encontró la imagen {filename}. Saltando...")
        continue
        
    # Conversión a distintos espacios de color para el análisis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_result = img_rgb.copy()

    # ---------------------------------------------------------------------------
    # --- 1. Umbralización y Extracción de Características ----------------------
    # ---------------------------------------------------------------------------
    # Se utiliza un umbral adaptativo para mitigar las variaciones de iluminación 
    # y sombras presentes en las fotografías.
    th_adapt = cv2.adaptiveThreshold(img_gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 19, 5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_adapt, connectivity=8, ltype=cv2.CV_32S)
    mask_caracteres = np.zeros_like(img_gray)
    
    # Filtrado inicial por geometría básica de caracteres
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        aspect_ratio = float(w) / h
        solidez = float(area) / (w * h)
        
        # Se conservan únicamente las componentes con proporciones de trazo tipográfico
        if (0.15 < aspect_ratio < 1.0) and (0.2 < solidez < 0.9) and (15 < h < 150):
            mask_caracteres[labels == i] = 255

    # ---------------------------------------------------------------------------
    # --- 2. Agrupación Morfológica ---------------------------------------------
    # ---------------------------------------------------------------------------
    # Clausura con un kernel rectangular horizontal amplio para fusionar los 
    # caracteres contiguos y formar el bloque de la patente.
    kernel_grupo = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 5))
    mask_agrupada = cv2.morphologyEx(mask_caracteres, cv2.MORPH_CLOSE, kernel_grupo)

    # ---------------------------------------------------------------------------
    # --- 3. Filtrado Sintáctico y Validación de Candidatos ---------------------
    # ---------------------------------------------------------------------------
    num_labels_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(mask_agrupada, connectivity=8, ltype=cv2.CV_32S)
    
    patente_roi_gray = None
    caracteres_finales = []
    
    mejor_puntaje = 0
    mejor_bbox = (0, 0, 0, 0)
    
    for i in range(1, num_labels_g):
        gx = stats_g[i, cv2.CC_STAT_LEFT]
        gy = stats_g[i, cv2.CC_STAT_TOP]
        gw = stats_g[i, cv2.CC_STAT_WIDTH]
        gh = stats_g[i, cv2.CC_STAT_HEIGHT]
        
        # Descarte de bloques que no posean aspecto de chapa rectangular
        if gw > 50 and gh > 15 and (gw/gh) > 1.2:
            
            roi_chars = mask_caracteres[gy:gy+gh, gx:gx+gw]
            c_num, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(roi_chars, connectivity=8, ltype=cv2.CV_32S)
            
            chars_candidatos = []
            for j in range(1, c_num):
                cx = c_stats[j, cv2.CC_STAT_LEFT]
                cy = c_stats[j, cv2.CC_STAT_TOP]
                cw = c_stats[j, cv2.CC_STAT_WIDTH]
                ch = c_stats[j, cv2.CC_STAT_HEIGHT]
                
                # Se descartan ruidos internos evaluando la altura relativa al bloque
                if 0.15 < (cw/ch) < 1.0 and ch > (gh * 0.3):
                    chars_candidatos.append({
                        'x': gx + cx, 'y': gy + cy, 'w': cw, 'h': ch, 
                        'img': roi_chars[cy:cy+ch, cx:cx+cw]
                    })
            
            if len(chars_candidatos) < 5: 
                continue 
            
            # Filtro de uniformidad geométrica
            h_mediano = np.median([c['h'] for c in chars_candidatos])
            y_mediano = np.median([c['y'] for c in chars_candidatos])
            
            chars_filtrados = []
            for c in chars_candidatos:
                # Se exige similitud en altura y alineación sobre el eje Y
                if abs(c['h'] - h_mediano) < (h_mediano * 0.35) and abs(c['y'] - y_mediano) < (h_mediano * 0.4):
                    chars_filtrados.append(c)

            # Validación del patrón de caracteres Mercosur
            if 5 <= len(chars_filtrados) <= 8:
                
                # Análisis de color en el espectro HSV para detectar la franja superior azul
                y_franja = max(0, gy - int(gh * 0.7))
                roi_hsv_franja = img_hsv[y_franja : gy + int(gh * 0.4), gx : gx + gw]
                
                lower_blue = np.array([85, 30, 30])
                upper_blue = np.array([140, 255, 255])
                mask_blue = cv2.inRange(roi_hsv_franja, lower_blue, upper_blue)
                
                porcentaje_azul = cv2.countNonZero(mask_blue) / (gw * int(gh * 1.1) + 1)
                
                # Asignación de puntaje para priorizar chapas con el formato exacto
                puntaje = porcentaje_azul * 100 
                if len(chars_filtrados) == 7:
                    puntaje += 50 
                    
                # Selección del mejor candidato en la imagen
                if puntaje > mejor_puntaje and puntaje > 1.0: 
                    mejor_puntaje = puntaje
                    
                    # Generación de Bounding Box Dinámico ajustado a los extremos del texto
                    min_x = min([c['x'] for c in chars_filtrados])
                    min_y = min([c['y'] for c in chars_filtrados])
                    max_x = max([c['x'] + c['w'] for c in chars_filtrados])
                    max_y = max([c['y'] + c['h'] for c in chars_filtrados])
                    
                    pad = 8
                    mejor_bbox = (max(0, min_x - pad), 
                                  max(0, min_y - pad * 2), 
                                  (max_x - max(0, min_x - pad)) + pad, 
                                  (max_y - max(0, min_y - pad * 2)) + pad)
                    
                    caracteres_finales = sorted(chars_filtrados, key=lambda c: c['x'])

    # ---------------------------------------------------------------------------
    # --- 4. Segmentación y Acondicionamiento de Caracteres ---------------------
    # ---------------------------------------------------------------------------
    if mejor_puntaje > 0:
        bx, by, bw, bh = mejor_bbox
        patente_roi_gray = img_gray[by:by+bh, bx:bx+bw]
        
        cv2.rectangle(img_result, (bx, by), (bx+bw, by+bh), (0, 255, 0), 3)

        kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        caracteres_recortados = []
        pad = 2 
        
        for c in caracteres_finales:
            x_char = max(0, c['x'] - pad)
            y_char = max(0, c['y'] - pad)
            w_char = c['w'] + (pad * 2)
            h_char = c['h'] + (pad * 2)
            
            cv2.rectangle(img_result, (x_char, y_char), (x_char + w_char, y_char + h_char), (255, 0, 0), 2)
            
            img_char_bin = th_adapt[y_char : y_char + h_char, x_char : x_char + w_char].copy()
            
            # Aislamiento de la componente principal para suprimir ruido de los bordes
            c_num_char, c_labels_char, c_stats_char, _ = cv2.connectedComponentsWithStats(img_char_bin, connectivity=8, ltype=cv2.CV_32S)
            
            if c_num_char > 1:
                # Se conserva exclusivamente la etiqueta con mayor área descartando el fondo
                etiqueta_mayor = 1 + np.argmax(c_stats_char[1:, cv2.CC_STAT_AREA])
                img_char_bin[c_labels_char != etiqueta_mayor] = 0

            # Restauración y refinamiento morfológico del trazo
            char_clean = cv2.morphologyEx(img_char_bin, cv2.MORPH_CLOSE, kernel_clausura)
            char_clean = cv2.erode(char_clean, kernel_erosion, iterations=1)
            
            caracteres_recortados.append(char_clean)
            
        # -----------------------------------------------------------------------
        # --- 5. Visualización e Informe de Resultados --------------------------
        # -----------------------------------------------------------------------
        print(f"\n[INFO] Imagen {idx_img}: Patente detectada.")
        print(f"       Caracteres segmentados exitosamente: {len(caracteres_recortados)}")
        
        plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(231)
        imshow(img_result, new_fig=False, title=f"Imagen {idx_img} - Detección", color_img=True)
        
        plt.subplot(232)
        imshow(mask_caracteres, new_fig=False, title="1. Contenido")
        
        plt.subplot(233)
        imshow(mask_agrupada, new_fig=False, title="2. Agrupación (Bloques)")
        
        plt.subplot(234)
        imshow(patente_roi_gray, new_fig=False, title="3. ROI Extraída")
        
        plt.subplot(235)
        # Concatenación de caracteres normalizados para visualización final
        max_h = max([img.shape[0] for img in caracteres_recortados])
        chars_concat = []
        for img in caracteres_recortados:
            pad_bottom = max(0, max_h - img.shape[0])
            padded_char = cv2.copyMakeBorder(img, 0, pad_bottom, 0, 4, cv2.BORDER_CONSTANT, value=0)
            chars_concat.append(padded_char)
            
        img_all_chars = np.hstack(chars_concat)
        imshow(img_all_chars, new_fig=False, title=f"4. Caracteres Segmentados ({len(caracteres_recortados)})")
            
        plt.tight_layout()
        plt.show(block=True)
        
    else:
        print(f"\n[INFO] Imagen {idx_img}: No se logró detectar una patente válida.")