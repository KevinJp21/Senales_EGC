#%% Importación de librerías
import wfdb
import matplotlib
matplotlib.use('TkAgg')  # Configurar el backend interactivo
import matplotlib.pyplot as plt
plt.ion()  # Activar modo interactivo
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import json

#%% Configuración inicial y carga de datos
# Ruta del archivo de registro ECG
record_name = r"./QT_Database/sel30/sel30"

# Leer el registro ECG
record = wfdb.rdrecord(record_name)

# Obtener la frecuencia de muestreo
fs = record.fs  

# Calcular cuántas muestras corresponden a 10 segundos
samples_10s = int(fs * 10)

# Extraer los primeros 10 segundos del primer canal de ECG
ecg_signal = record.p_signal[:samples_10s, 0]  

# Crear el eje de tiempo
time_axis = np.arange(samples_10s) / fs

#%% Detección de ondas R
peaks, _ = find_peaks(ecg_signal, height=np.max(ecg_signal) * 0.6, distance=fs*0.6)

#%% Definición de funciones auxiliares
def detectar_limites(signal, picos, min_distancia=10):
    inicios = []
    finales = []

    for pico in picos:
        # Buscar el inicio (antes del pico, donde la pendiente empieza a subir)
        inicio = pico
        while inicio > 0 and signal[inicio] > signal[inicio - 1]:
            inicio -= 1
        
        # Buscar el final (después del pico, donde la pendiente deja de bajar)
        final = pico
        while final < len(signal) - 1 and signal[final] > signal[final + 1]:
            final += 1
        
        # Asegurar que el final no sea el mismo punto que el pico R
        if final - pico < min_distancia:
            final = min(pico + min_distancia, len(signal) - 1)

        inicios.append(inicio)
        finales.append(final)

    return np.array(inicios), np.array(finales)

#%% Procesamiento de ondas R
# Detectar inicios y finales
inicios, finales = detectar_limites(ecg_signal, peaks)

# Extraer las ondas R y almacenarlas en una lista con sus tiempos
ondas_r = [ecg_signal[inicio:final] for inicio, final in zip(inicios, finales)]
tiempos_r = [time_axis[inicio:final] for inicio, final in zip(inicios, finales)]

#%% Visualización de la señal completa
plt.figure(figsize=(12, 4))
plt.plot(time_axis, ecg_signal, label="ECG Signal", color='m')  # Fucsia
plt.scatter(time_axis[peaks], ecg_signal[peaks], color='r', label="Picos R", marker="o")  # Picos R en rojo
plt.scatter(time_axis[inicios], ecg_signal[inicios], color='g', label="Inicio", marker="x")  # Inicios en verde
plt.scatter(time_axis[finales], ecg_signal[finales], color='b', label="Final", marker="x")  # Finales en azul

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("ECG - Inicio y Fin de las Ondas R")
plt.legend()
plt.grid()
plt.show()

#%% Funciones de visualización interactiva de ondas R originales
def visualizar_onda(index):
    if not plt.get_fignums():  # Si no hay figuras abiertas, crear una nueva
        global fig_orig, ax_orig
        fig_orig, ax_orig = plt.subplots(figsize=(10, 6))
        fig_orig.canvas.mpl_connect("key_press_event", on_key)
    
    ax_orig.clear()
    ax_orig.plot(tiempos_r[index], ondas_r[index], color='c', linewidth=2)
    ax_orig.set_title(f'Onda R Original {index+1} de {len(ondas_r)}')
    ax_orig.set_xlabel('Tiempo (s)')
    ax_orig.set_ylabel('Amplitud')
    ax_orig.grid(True)
    plt.draw()
    plt.pause(0.1)  # Pequeña pausa para actualizar la figura

    # Imprimir los datos en la consola con tiempos
    print(f"\n📊 Datos de la Onda R Original {index+1} (Tiempo - Amplitud):")
    for t, amp in zip(tiempos_r[index], ondas_r[index]):
        print(f"{t:.5f} s -> {amp:.5f}")

def on_key(event):
    global index_orig
    if event.key == "right":
        index_orig = (index_orig + 1) % len(ondas_r)  # Avanzar
        visualizar_onda(index_orig)
    elif event.key == "left":
        index_orig = (index_orig - 1) % len(ondas_r)  # Retroceder
        visualizar_onda(index_orig)

#%% Inicialización de la visualización interactiva original
# Asegurarse de que todas las variables necesarias estén definidas
if 'ondas_r' in locals() and 'tiempos_r' in locals():
    index_orig = 0
    visualizar_onda(index_orig)
    plt.show(block=True)  # Bloquear la ejecución para mantener la ventana abierta
else:
    print("Error: Las variables 'ondas_r' y 'tiempos_r' no están definidas. Ejecuta las celdas anteriores primero.")

#%% Estandarización y normalización de ondas R
# Preparar las ventanas centradas en cada pico R (duración de 120 ms)
ventana_ms = 120
ventana_muestras = int((ventana_ms / 1000) * fs)

# Estructuras para almacenar los datos
datos_ondas = {
    'ondas_originales': [],
    'ondas_normalizadas': [],
    'tiempos': [],
    'duraciones': [],
    'etiquetas': [],  # Agregamos las etiquetas binarias
    'metadata': {
        'frecuencia_muestreo': fs,
        'ventana_ms': ventana_ms,
        'muestras_por_ventana': ventana_muestras
    }
}

# Extraer y normalizar cada onda R
for pico in peaks:
    mitad = ventana_muestras // 2
    inicio = max(pico - mitad, 0)
    final = min(pico + mitad, len(ecg_signal))

    if final - inicio < ventana_muestras:
        if inicio == 0:
            final = ventana_muestras
        elif final == len(ecg_signal):
            inicio = len(ecg_signal) - ventana_muestras

    # Extraer la onda original
    onda_original = ecg_signal[inicio:final]
    tiempo_onda = time_axis[inicio:final]
    duracion = time_axis[final - 1] - time_axis[inicio]

    # Normalizar usando la fórmula (X + |B|)/(A-B)
    A = 400  # max valor
    B = -200  # min valor
    onda_normalizada = (onda_original + abs(B)) / (A - B)

    # Generar etiquetas binarias (1 en el pico R, 0 en el resto)
    etiquetas = np.zeros_like(onda_original)
    pos_pico = mitad  # El pico R está en el centro de la ventana
    etiquetas[pos_pico] = 1

    # Almacenar los datos
    datos_ondas['ondas_originales'].append(onda_original)
    datos_ondas['ondas_normalizadas'].append(onda_normalizada)
    datos_ondas['tiempos'].append(tiempo_onda)
    datos_ondas['duraciones'].append(duracion)
    datos_ondas['etiquetas'].append(etiquetas)

# Convertir listas a arrays de numpy para mejor manejo
datos_ondas['ondas_originales'] = np.array(datos_ondas['ondas_originales'])
datos_ondas['ondas_normalizadas'] = np.array(datos_ondas['ondas_normalizadas'])
datos_ondas['tiempos'] = np.array(datos_ondas['tiempos'])
datos_ondas['duraciones'] = np.array(datos_ondas['duraciones'])
datos_ondas['etiquetas'] = np.array(datos_ondas['etiquetas'])

#%% Visualización interactiva de ondas R normalizadas con etiquetas
def visualizar_onda_normalizada(index):
    if not plt.get_fignums():
        global fig_norm, ax_norm
        fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
        fig_norm.canvas.mpl_connect("key_press_event", on_key_normalizada)
    
    ax_norm.clear()
    # Graficar la onda normalizada
    ax_norm.plot(datos_ondas['tiempos'][index], 
                datos_ondas['ondas_normalizadas'][index], 
                color='c', linewidth=2, label='Señal')
    
    # Marcar los puntos etiquetados como 1 (picos R)
    etiquetas = datos_ondas['etiquetas'][index]
    tiempo_pico = datos_ondas['tiempos'][index][etiquetas == 1]
    valor_pico = datos_ondas['ondas_normalizadas'][index][etiquetas == 1]
    ax_norm.scatter(tiempo_pico, valor_pico, color='r', s=100, label='Pico R (1)')
    
    ax_norm.set_title(f'Onda R normalizada {index+1} de {len(datos_ondas["ondas_normalizadas"])}\n'
                     f'Duración: {datos_ondas["duraciones"][index]:.2f} s')
    ax_norm.set_xlabel('Tiempo (s)')
    ax_norm.set_ylabel('Amplitud Normalizada')
    ax_norm.legend()
    ax_norm.grid(True)
    plt.draw()
    plt.pause(0.1)

    print(f"\n📊 Datos de la Onda R normalizada {index+1}:")
    print(f"Duración: {datos_ondas['duraciones'][index]:.2f} s")
    print("Tiempo (s) -> Amplitud normalizada -> Etiqueta")
    for t, amp, etiq in zip(datos_ondas['tiempos'][index], 
                           datos_ondas['ondas_normalizadas'][index],
                           datos_ondas['etiquetas'][index]):
        print(f"{t:.5f} -> {amp:.5f} -> {int(etiq)}")

def on_key_normalizada(event):
    global index_norm
    if event.key == "right":
        index_norm = (index_norm + 1) % len(datos_ondas['ondas_normalizadas'])
        visualizar_onda_normalizada(index_norm)
    elif event.key == "left":
        index_norm = (index_norm - 1) % len(datos_ondas['ondas_normalizadas'])
        visualizar_onda_normalizada(index_norm)

#%% Inicialización de visualización de ondas normalizadas
if len(datos_ondas['ondas_normalizadas']) > 0:
    index_norm = 0
    visualizar_onda_normalizada(index_norm)
    plt.show(block=True)
else:
    print("Error: No se encontraron ondas R para normalizar.")

#%% Guardar los datos procesados para entrenamiento
# Preparar los datos en formato JSON
datos_json = {
    'señales_normalizadas': datos_ondas['ondas_normalizadas'].tolist(),
    'etiquetas': datos_ondas['etiquetas'].tolist(),
    'tiempos': datos_ondas['tiempos'].tolist(),
    'duraciones': datos_ondas['duraciones'].tolist(),
    'metadata': datos_ondas['metadata']
}

# Guardar los datos en un archivo JSON
with open('datos_ondas.json', 'w') as f:
    json.dump(datos_json, f, indent=4)

print("\nResumen de los datos guardados en JSON:")
print(f"Número total de ondas R: {len(datos_json['señales_normalizadas'])}")
print(f"Dimensiones de cada onda: {len(datos_json['señales_normalizadas'][0])}")
print(f"Frecuencia de muestreo: {datos_json['metadata']['frecuencia_muestreo']} Hz")
print(f"Ventana temporal: {datos_json['metadata']['ventana_ms']} ms")
print("\nArchivo JSON creado: datos_ondas.json")

# %%

