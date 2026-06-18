"""
 TUIA - Procesamiento de Imágenes I
    TP2 - Problema 1A: Segmentación del área de la cinta transportadora (ROI)
 
    Estrategia:
 La imagen tiene una estructura vertical clara:
    - Fondo superior / inferior oscuro
    - Borde metálico superior (muy brillante, ~200 de gris)
    - Línea de separación oscura (transición abrupta)
    - CINTA (ROI): fondo oscuro ~60-90 con pastillas encima  <-- lo que queremos
    - Línea de separación oscura (transición abrupta)
    - Borde metálico inferior (brillante, ~165 de gris)
    - Base / suelo

"""
 
import cv2
import numpy as np
 
# ─── 1. CARGA ──────────────────────────────────────────────────────────────────
IMAGE_PATH="imagenes\pills.png"
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {IMAGE_PATH}")
 
gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # unidad: color
h_img, w_img = gray_full.shape
print(f"[INFO] Imagen cargada: {w_img}x{h_img} px")
 
# ─── 2. EXTRAER ROI (cinta transportadora) ────────────────────────────────────
row_means   = np.mean(gray_full, axis=1).astype(np.float32)
profile_2d  = row_means.reshape(1, h_img)
smoothed_2d = cv2.GaussianBlur(profile_2d, (1, 31), sigmaX=0)   # Filtra_frecuencial
smooth      = smoothed_2d.reshape(h_img)
 
BRIGHT_THRESH, DARK_THRESH = 120, 50
belt_top = belt_bottom = None
 
found_bright_top = False
for r in range(h_img):
    if smooth[r] > BRIGHT_THRESH: found_bright_top = True
    if found_bright_top and smooth[r] < DARK_THRESH:
        for r2 in range(r, h_img):
            if smooth[r2] > 30: belt_top = r2; break
        break
 
found_bright_bot = False
for r in range(h_img - 1, -1, -1):
    if smooth[r] > BRIGHT_THRESH: found_bright_bot = True
    if found_bright_bot and smooth[r] < DARK_THRESH:
        for r2 in range(r, -1, -1):
            if smooth[r2] > 30: belt_bottom = r2; break
        break
 
 
roi      = img[belt_top:belt_bottom, 0:w_img]
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)    # Color
roi_hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)     # Color
h_roi, w_roi = roi_gray.shape
print(f"[INFO] ROI extraída: {w_roi}x{h_roi} px  (filas {belt_top}-{belt_bottom})")
 
# ─── 3. PREPROCESAMIENTO ──────────────────────────────────────────────────────
blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)    # Filtra_frecuencial
print("[INFO] Suavizado Gaussiano aplicado (kernel 5x5)")
 
# ─── 4. UMBRALIZACIÓN DE OTSU ─────────────────────────────────────────────────
ret, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Segmentacion
print(f"[INFO] Umbral Otsu calculado: {ret:.0f}")
 
# ─── 5. MORFOLOGÍA ────────────────────────────────────────────────────────────
kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Morfologia
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_morph, iterations=1)
print("[INFO] Operaciones morfológicas aplicadas (CLOSE x2, OPEN x1)")
 
# ─── 6. DETECCIÓN DE CONTORNOS ────────────────────────────────────────────────
contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Segmentacion
MIN_AREA = 100
contours = [c for c in contours_raw if cv2.contourArea(c) > MIN_AREA]
print(f"[INFO] Contornos: {len(contours_raw)} totales → {len(contours)} válidos (área > {MIN_AREA})")

# ─── 7. CLASIFICACIÓN ─────────────────────────────────────────────────────────
# Máscara global de color azul (para CAB)
BLUE_LO = np.array([95,  60,  60])
BLUE_HI = np.array([135, 255, 255])
blue_global = cv2.inRange(roi_hsv, BLUE_LO, BLUE_HI)   # Segmentacion (inRange)

pills = []
 
for c in contours:
    area  = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
 
    # Circularidad: 4π·A / P²  (1.0 = círculo, ~0.785 = cuadrado)
    circ = (4 * np.pi * area / perim**2) if perim > 0 else 0
 
    # Aspect ratio con rectángulo mínimo rotado
    rect   = cv2.minAreaRect(c)
    rw, rh = rect[1]
    ar     = max(rw, rh) / (min(rw, rh) + 1e-5)
 
    # Bounding box para extraer color y contar píxeles azules
    x, y, bw, bh = cv2.boundingRect(c)
 
    # Color HSV promedio DENTRO del contorno (cmask)
    cmask = np.zeros(roi_gray.shape, dtype=np.uint8)
    cv2.drawContours(cmask, [c], -1, 255, -1)
    mean_hsv = cv2.mean(roi_hsv, mask=cmask)
    H, S, V  = mean_hsv[0], mean_hsv[1], mean_hsv[2]
 
    blue_px = cv2.countNonZero(blue_global[y:y+bh, x:x+bw])
 
    # ── Reglas de clasificación ───────────────────────────────────────────
    # Orden de prioridad: CAB → CA → RR → CB → RB
    if blue_px > 100:
        # Muchos píxeles azules en la región → cápsula azul/blanca
        tipo = 'CAB'
    elif ar > 1.8 and area > 2500 and S > 50:
        # Elongada, grande y saturada → cápsula amarilla
        tipo = 'CA'
    elif ar < 1.3 and S > 40:
        # Compacta y saturada (rosa/magenta) → redonda rosa
        tipo = 'RR'
    elif ar < 1.3 and S < 15 and circ < 0.855:
        # Compacta, blanca, baja circularidad → cuadrada blanca
        # Círculo: circ ~0.86-0.91 | Cuadrado: circ ~0.77-0.84
        tipo = 'CB'
    elif ar < 1.3 and S < 15:
        # Compacta, blanca, alta circularidad → redonda blanca
        tipo = 'RB'
    else:
        tipo = '??'
 
    pills.append({
        'contour': c,
        'area':    area,
        'ar':      ar,
        'circ':    circ,
        'H': H, 'S': S, 'V': V,
        'blue_px': blue_px,
        'tipo':    tipo,
        'bbox':    (x, y, bw, bh),
        'rect':    rect,
    })
 
# ─── 8. ASIGNAR IDs ───────────────────────────────────────────────────────────
contadores = {}
for p in pills:
    t = p['tipo']
    contadores[t] = contadores.get(t, 0) + 1
    p['id'] = f"{t}{contadores[t]}"
 
# ─── 9. INFORME POR CONSOLA ───────────────────────────────────────────────────
nombres = {
    'CAB': 'Cápsula Azul/Blanca',
    'CA' : 'Cápsula Amarilla   ',
    'RR' : 'Redonda Rosa       ',
    'CB' : 'Cuadrada Blanca    ',
    'RB' : 'Redonda Blanca     ',
    '??' : 'Sin clasificar     ',
}
 
print("\n" + "="*62)
print("  RESULTADOS - DETECCIÓN Y CLASIFICACIÓN DE PASTILLAS")
print("="*62)
print(f"\n  Total pastillas detectadas: {len(pills)}\n")
for tipo, cant in sorted(contadores.items()):
    nombre = nombres.get(tipo, tipo)
    print(f"  {tipo}  {nombre}  → {cant:3d} unidades")
 
print("\n" + "-"*62)
print(f"  {'ID':<8} {'Tipo':<5} {'Area':>5} {'AR':>5} {'Circ':>5} {'S':>5} {'BlPx':>5}")
print("-"*62)
for p in sorted(pills, key=lambda x: x['id']):
    print(f"  {p['id']:<8} {p['tipo']:<5} {p['area']:>5.0f} "
          f"{p['ar']:>5.2f} {p['circ']:>5.3f} {p['S']:>5.1f} {p['blue_px']:>5}")
print("="*62)
 
# ─── 10. IMAGEN RESULTADO CON ETIQUETAS ───────────────────────────────────────
result = img.copy()
 
COLOR = {
    'CAB': (255, 100,   0),
    'CA' : (  0, 200, 255),
    'RR' : ( 60,  60, 255),
    'CB' : (180, 180, 180),
    'RB' : (220, 220, 220),
    '??' : (  0, 255,   0),
}
 
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1
 
for p in pills:
    c     = p['contour']
    color = COLOR.get(p['tipo'], (0, 255, 0))
    c_offset = c + np.array([[[0, belt_top]]])
    cv2.drawContours(result, [c_offset], -1, color, 2)
    x, y, bw, bh = p['bbox']
    y_orig = y + belt_top
    label  = p['id']
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
    cv2.rectangle(result, (x, y_orig - th - 4), (x + tw + 2, y_orig), color, -1)
    cv2.putText(result, label, (x + 1, y_orig - 3), FONT, FONT_SCALE, (0, 0, 0), THICKNESS)
 
# Leyenda
leyenda_y = 20
cv2.rectangle(result, (5, 5), (230, 130), (30, 30, 30), -1)
for tipo, nombre in nombres.items():
    if tipo in contadores:
        color = COLOR.get(tipo, (255,255,255))
        cv2.rectangle(result, (10, leyenda_y-10), (25, leyenda_y+2), color, -1)
        cv2.putText(result, f"{tipo}: {contadores.get(tipo,0)}  ({nombre.strip()})",
                    (30, leyenda_y), FONT, 0.38, (255,255,255), 1)
        leyenda_y += 22
 

 
cv2.imshow('Resultado - Deteccion y Clasificacion', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
 