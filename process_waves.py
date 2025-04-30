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

# Extraer todo el canal de ECG
ecgsignal = record.p_signal[:, 0]

# Crear el eje de tiempo
timeaxis = np.arange(len(ecgsignal)) / fs

#%% Detección de ondas R
rpeaks, _ = find_peaks(ecgsignal, distance=fs*0.3, prominence=0.1)
refined_peaks = []
window_refine = int(0.05 * fs)  # 50 ms a cada lado
for peak in rpeaks:
    start = max(peak - window_refine, 0)
    end = min(peak + window_refine, len(ecgsignal))
    local_max = np.argmax(ecgsignal[start:end])
    refined_peaks.append(start + local_max)
rpeaks = np.array(refined_peaks)

#%% Definición de funciones auxiliares
def detect_limits(signal, peaks, min_distance=10):
    start_indices = []
    end_indices = []
    for peak in peaks:
        # Buscar el inicio (antes del pico, donde la pendiente empieza a subir)
        start = peak
        while start > 0 and signal[start] > signal[start - 1]:
            start -= 1
        # Buscar el final (después del pico, donde la pendiente deja de bajar)
        end = peak
        while end < len(signal) - 1 and signal[end] > signal[end + 1]:
            end += 1
        # Asegurar que el final no sea el mismo punto que el pico R
        if end - peak < min_distance:
            end = min(peak + min_distance, len(signal) - 1)
        start_indices.append(start)
        end_indices.append(end)
    return np.array(start_indices), np.array(end_indices)

#%% Procesamiento de ondas R
# Detectar inicios y finales
start_indices, end_indices = detect_limits(ecgsignal, rpeaks)

# Extraer las ondas R y almacenarlas en una lista con sus tiempos
r_waves = [ecgsignal[start:end] for start, end in zip(start_indices, end_indices)]
times_r_waves = [timeaxis[start:end] for start, end in zip(start_indices, end_indices)]

#%% Visualización de la señal completa
plt.figure(figsize=(12, 4))
plt.plot(timeaxis, ecgsignal, label="Señal ECG", color='m')
plt.scatter(timeaxis[rpeaks], ecgsignal[rpeaks], color='r', label="Picos R", marker="o")
plt.scatter(timeaxis[start_indices], ecgsignal[start_indices], color='g', label="Inicio", marker="x")
plt.scatter(timeaxis[end_indices], ecgsignal[end_indices], color='b', label="Final", marker="x")

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("ECG - Inicio y Fin de las Ondas R")
plt.legend()
plt.grid()
plt.show()

#%% Funciones de visualización interactiva de ondas R originales
def visualizar_onda(index):
    if not plt.get_fignums(): # Si no hay figuras abiertas, crear una nueva
        global fig_orig, ax_orig
        fig_orig, ax_orig = plt.subplots(figsize=(10, 6))
        fig_orig.canvas.mpl_connect("key_press_event", on_key)
    ax_orig.clear()
    ax_orig.plot(times_r_waves[index], r_waves[index], color='c', linewidth=2)
    ax_orig.set_title(f'Onda R Original {index+1} de {len(r_waves)}')
    ax_orig.set_xlabel('Tiempo (s)')
    ax_orig.set_ylabel('Amplitud')
    ax_orig.grid(True)
    plt.draw()
    plt.pause(0.1) # Pequeña pausa para actualizar la figura
    # Imprimir los datos en la consola con tiempos
    print(f"\n📊 Datos de la Onda R Original {index+1} (Tiempo - Amplitud):")
    for t, amp in zip(times_r_waves[index], r_waves[index]):
        print(f"{t:.5f} s -> {amp:.5f}")

def on_key(event):
    global index_orig
    if event.key == "right":
        index_orig = (index_orig + 1) % len(r_waves) # Avanzar
        visualizar_onda(index_orig)
    elif event.key == "left":
        index_orig = (index_orig - 1) % len(r_waves) # Retroceder
        visualizar_onda(index_orig)

#%% Inicialización de la visualización interactiva original
# Asegurarse de que todas las variables necesarias estén definidas
if 'r_waves' in locals() and 'times_r_waves' in locals():
    index_orig = 0
    visualizar_onda(index_orig)
    plt.show(block=True) # Bloquear la ejecución para mantener la ventana abierta
else:
    print("Error: Las variables 'r_waves' y 'times_r_waves' no están definidas. Ejecuta las celdas anteriores primero.")

#%% Estandarización y normalización de ondas R
# Preparar las ventanas centradas en cada pico R (duración de 120 ms)
window_ms = 120
window_samples = int((window_ms / 1000) * fs)
# Estructuras para almacenar los datos
data_waves = {
    'original_waves': [],
    'normalized_waves': [],
    'times': [],
    'durations': [],
    'metadata': {
        'frecuencia_muestreo': fs,
        'ventana_ms': window_ms,
        'muestras_por_ventana': window_samples
    }
}
# Extraer y normalizar cada onda R
for peak in rpeaks:
    half = window_samples // 2
    start = max(peak - half, 0)
    end = min(peak + half, len(ecgsignal))
    if end - start < window_samples:
        if start == 0:
            end = window_samples
        elif end == len(ecgsignal):
            start = len(ecgsignal) - window_samples
    # Extraer la onda original
    original_wave = ecgsignal[start:end] 
    time_wave = timeaxis[start:end]
    duration = timeaxis[end - 1] - timeaxis[start]
    # Normalizar usando la fórmula (X - B)/(A-B)
    A = np.max(original_wave) # max valor
    B = np.min(original_wave) # min valor
    normalized_wave = (original_wave - B) / (A - B)
    # Almacenar los datos
    data_waves['original_waves'].append(original_wave)
    data_waves['normalized_waves'].append(normalized_wave)
    data_waves['times'].append(time_wave)
    data_waves['durations'].append(duration)
# Convertir listas a arrays de numpy para mejor manejo
data_waves['original_waves'] = np.array(data_waves['original_waves'])
data_waves['normalized_waves'] = np.array(data_waves['normalized_waves'])
data_waves['times'] = np.array(data_waves['times'])
data_waves['durations'] = np.array(data_waves['durations'])

#%% Visualización interactiva de ondas R normalizadas
def visualizar_onda_normalizada(index):
    if not plt.get_fignums():
        global fig_norm, ax_norm
        fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
        fig_norm.canvas.mpl_connect("key_press_event", on_key_normalizada)
    ax_norm.clear()
    # Graficar la onda normalizada
    ax_norm.plot(data_waves['times'][index], 
                data_waves['normalized_waves'][index], 
                color='c', linewidth=2, label='Señal')
    ax_norm.set_title(f'Onda R normalizada {index+1} de {len(data_waves["normalized_waves"])}\n'
                     f'Duración: {data_waves["durations"][index]:.2f} s')
    ax_norm.set_xlabel('Tiempo (s)')
    ax_norm.set_ylabel('Amplitud Normalizada')
    ax_norm.legend()
    ax_norm.grid(True)
    plt.draw()
    plt.pause(0.1)
    print(f"\n📊 Datos de la Onda R normalizada {index+1}:")
    print(f"Duración: {data_waves['durations'][index]:.2f} s")
    print("Tiempo (s) -> Amplitud normalizada")
    for t, amp in zip(data_waves['times'][index], 
                       data_waves['normalized_waves'][index]):
        print(f"{t:.5f} -> {amp:.5f}")

def on_key_normalizada(event):
    global index_norm
    if event.key == "right":
        index_norm = (index_norm + 1) % len(data_waves['normalized_waves'])
        visualizar_onda_normalizada(index_norm)
    elif event.key == "left":
        index_norm = (index_norm - 1) % len(data_waves['normalized_waves'])
        visualizar_onda_normalizada(index_norm)

#%% Inicialización de visualización de ondas normalizadas
if len(data_waves['normalized_waves']) > 0:
    index_norm = 0
    visualizar_onda_normalizada(index_norm)
    plt.show(block=True)
else:
    print("Error: No se encontraron ondas R para normalizar.")

#%% Visualización de todas las ondas normalizadas concatenadas
plt.figure(figsize=(15, 6))
full_signal = np.concatenate(data_waves['normalized_waves'])
time_full = np.arange(len(full_signal)) / fs
plt.plot(time_full, full_signal, 'b-', linewidth=1)
plt.title('Señal ECG Concatenada de Ondas R Normalizadas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud Normalizada')
plt.grid(True)
plt.show()

#%% Guardar los datos procesados para entrenamiento
# Preparar los datos en formato JSON
data_json = {
    'normalized_waves': data_waves['normalized_waves'].tolist(),
    'metadata': data_waves['metadata']
}
with open('wave_data.json', 'w') as f:
    json.dump(data_json, f, indent=4)
print("\nResumen de los datos guardados en JSON:")
print(f"Número total de ondas R: {len(data_json['normalized_waves'])}")
print(f"Dimensiones de cada onda: {len(data_json['normalized_waves'][0])}")
print(f"Frecuencia de muestreo: {data_json['metadata']['frecuencia_muestreo']} Hz")
print(f"Ventana temporal: {data_json['metadata']['ventana_ms']} ms")
print("\nArchivo JSON creado: wave_data.json")

# %%
