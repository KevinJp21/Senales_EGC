#%% Importación de librerías
import wfdb
import matplotlib
matplotlib.use('TkAgg')  # Configurar el backend interactivo
import matplotlib.pyplot as plt
plt.ion()  # Activar modo interactivo
import numpy as np
import pandas as pd
import json
from datetime import datetime

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

#%% Cargar y procesar el archivo CSV con anotaciones de ondas R
# Cargar el archivo CSV con las anotaciones
annotations_df = pd.read_csv('ondas_R.csv')

# Filtrar las anotaciones por tipo
start_points = annotations_df[annotations_df['Type'] == '(']
r_points = annotations_df[annotations_df['Type'] == 'N']
end_points = annotations_df[annotations_df['Type'] == ')']

# Asegurarse de que tenemos el mismo número de cada tipo de anotación
min_length = min(len(start_points), len(r_points), len(end_points))
print(f"Total de ondas R encontradas en el CSV: {min_length}")

# Extraer los tiempos y convertirlos a índices de muestra (sin redondear)
start_indices = [time * fs for time in start_points['Time'][:min_length]]
end_indices = [time * fs for time in end_points['Time'][:min_length]]

# Convertir a arrays de numpy y redondear a enteros
start_indices = np.round(start_indices).astype(int)
end_indices = np.round(end_indices).astype(int)

# Verificar que todos los índices están dentro del rango de la señal
valid_indices = []
for i in range(min_length):
    if (start_indices[i] < len(ecgsignal) and end_indices[i] < len(ecgsignal)):
        valid_indices.append(i)

start_indices = start_indices[valid_indices]
end_indices = end_indices[valid_indices]

# Calcular los picos R como el máximo local en cada segmento
r_peaks_calculados = []
for start, end in zip(start_indices, end_indices):
    segmento = ecgsignal[start:end]
    if len(segmento) > 0:
        max_idx = np.argmax(segmento)
        r_peaks_calculados.append(start + max_idx)
    else:
        r_peaks_calculados.append(start)
r_peaks_calculados = np.array(r_peaks_calculados)

#%% Extraer las ondas R basadas en los índices del CSV
# Extraer las ondas R y almacenarlas en una lista con sus tiempos
r_waves = [ecgsignal[start:end] for start, end in zip(start_indices, end_indices)]
times_r_waves = [timeaxis[start:end] for start, end in zip(start_indices, end_indices)]

#%% Visualización de la señal completa con las anotaciones del CSV
plt.figure(figsize=(12, 4))
plt.plot(timeaxis, ecgsignal, label="Señal ECG", color='m')
plt.scatter(timeaxis[r_peaks_calculados], ecgsignal[r_peaks_calculados], color='r', label="Picos R (calculados)", marker="o")
plt.scatter(timeaxis[start_indices], ecgsignal[start_indices], color='g', label="Inicio (()", marker="x")
plt.scatter(timeaxis[end_indices], ecgsignal[end_indices], color='b', label="Final ())", marker="x")

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("ECG - Ondas R desde archivo CSV")
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
if 'r_waves' in locals() and 'times_r_waves' in locals() and len(r_waves) > 0:
    index_orig = 0
    visualizar_onda(index_orig)
    plt.show(block=True) # Bloquear la ejecución para mantener la ventana abierta
else:
    print("Error: No se encontraron ondas R válidas para visualizar.")

#%% Estandarización y normalización de ondas R
# Preparar las estructuras para almacenar los datos
data_waves = {
    'original_waves': [],
    'normalized_waves': [],
    'times': [],
    'durations': [],
    'metadata': {
        'frecuencia_muestreo': fs,
        'source': 'CSV annotations'
    }
}

# Duración estándar en segundos
duracion_estandar = 0.120
longitud_estandar = int(duracion_estandar * fs)

ondas_estandarizadas = []
ondas_normalizadas = []
tiempos_estandarizados = []

for start, end in zip(start_indices, end_indices):
    final = start + longitud_estandar
    if final <= len(ecgsignal):
        onda = ecgsignal[start:final]
        tiempo = timeaxis[start:final]
    else:
        onda = ecgsignal[start:]
        tiempo = timeaxis[start:]
    ondas_estandarizadas.append(onda)
    tiempos_estandarizados.append(tiempo)
    
    # Normalización
    A = np.max(onda)
    B = np.min(onda)
    if A == B:
        normalized_wave = np.zeros_like(onda)
    else:
        normalized_wave = (onda - B) / (A - B)
    ondas_normalizadas.append(normalized_wave)

ondas_estandarizadas = np.array(ondas_estandarizadas, dtype=object)
ondas_normalizadas = np.array(ondas_normalizadas, dtype=object)
tiempos_estandarizados = np.array(tiempos_estandarizados, dtype=object)

# Crear array de tags (1 para ondas con anotación 'N', 0 para otras)
tags = np.zeros(len(ondas_normalizadas))
for i, (start, end) in enumerate(zip(start_indices, end_indices)):
    # Buscar si hay alguna anotación 'N' en este rango
    r_peaks_en_rango = r_points[(r_points['Time'] * fs >= start) & (r_points['Time'] * fs <= end)]
    if len(r_peaks_en_rango) > 0:
        tags[i] = 1

print(f"\nDistribución de tags:")
print(f"Total de ondas: {len(tags)}")
print(f"Ondas con anotación 'N': {np.sum(tags)}")
print(f"Ondas sin anotación 'N': {len(tags) - np.sum(tags)}")

#%% Visualización interactiva de ondas R normalizadas
def visualizar_onda_normalizada(index):
    if not plt.get_fignums():
        global fig_norm, ax_norm
        fig_norm, ax_norm = plt.subplots(figsize=(10, 6))
        fig_norm.canvas.mpl_connect("key_press_event", on_key_normalizada)
    ax_norm.clear()
    # Graficar la onda normalizada
    ax_norm.plot(tiempos_estandarizados[index], 
                ondas_normalizadas[index], 
                color='c', linewidth=2, label='Señal')
    ax_norm.set_title(f'Onda R normalizada {index+1} de {len(ondas_normalizadas)}\n'
                     f'Duración: {tiempos_estandarizados[index][-1] - tiempos_estandarizados[index][0]:.2f} s')
    ax_norm.set_xlabel('Tiempo (s)')
    ax_norm.set_ylabel('Amplitud Normalizada')
    ax_norm.legend()
    ax_norm.grid(True)
    plt.draw()
    plt.pause(0.1)
    print(f"\n📊 Datos de la Onda R normalizada {index+1}:")
    print(f"Duración: {tiempos_estandarizados[index][-1] - tiempos_estandarizados[index][0]:.2f} s")
    print("Tiempo (s) -> Amplitud normalizada")
    for t, amp in zip(tiempos_estandarizados[index], 
                       ondas_normalizadas[index]):
        print(f"{t:.5f} -> {amp:.5f}")

def on_key_normalizada(event):
    global index_norm
    if event.key == "right":
        index_norm = (index_norm + 1) % len(ondas_normalizadas)
        visualizar_onda_normalizada(index_norm)
    elif event.key == "left":
        index_norm = (index_norm - 1) % len(ondas_normalizadas)
        visualizar_onda_normalizada(index_norm)

#%% Inicialización de visualización de ondas normalizadas
if len(ondas_normalizadas) > 0:
    index_norm = 0
    visualizar_onda_normalizada(index_norm)
    plt.show(block=True)
else:
    print("Error: No se encontraron ondas R para normalizar.")

#%% Visualización de todas las ondas normalizadas concatenadas
if len(ondas_normalizadas) > 0:
    plt.figure(figsize=(15, 6))
    
    # Determinar la longitud máxima para el padding
    max_len = max(len(wave) for wave in ondas_normalizadas)
    
    # Crear un array para todas las ondas con padding
    all_waves = []
    for wave in ondas_normalizadas:
        # Padding si es necesario
        padded_wave = np.pad(wave, (0, max_len - len(wave)), 'constant', constant_values=(0, 0))
        all_waves.append(padded_wave)
    
    # Concatenar todas las ondas
    full_signal = np.concatenate(all_waves)
    time_full = np.arange(len(full_signal)) / fs
    
    plt.plot(time_full, full_signal, 'b-', linewidth=1)
    plt.title('Señal ECG Concatenada de Ondas R Normalizadas (desde CSV)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud Normalizada')
    plt.grid(True)
    plt.show()
else:
    print("Error: No hay ondas normalizadas para visualizar.")

#%% Guardar los datos procesados para entrenamiento
if len(ondas_normalizadas) > 0:
    # Preparar los datos en formato JSON
    # Convertimos todo a listas para JSON
    data_json = {
        'normalized_waves': [wave.tolist() for wave in ondas_normalizadas],
        'tags': tags.tolist(),
        'metadata': {
            'frecuencia_muestreo': float(fs),
            'source': 'CSV annotations',
            'total_waves': len(ondas_normalizadas),
            'start_indices': start_indices.tolist(),
            'end_indices': end_indices.tolist(),
            'r_peaks_calculados': r_peaks_calculados.tolist()
        }
    }
    
    with open('wave_data.json', 'w') as f:
        json.dump(data_json, f, indent=4)
    
    print("\nResumen de los datos guardados en JSON:")
    print(f"Número total de ondas R: {len(data_json['normalized_waves'])}")
    if len(data_json['normalized_waves']) > 0:
        print(f"Dimensiones de la primera onda: {len(data_json['normalized_waves'][0])}")
    print(f"Frecuencia de muestreo: {data_json['metadata']['frecuencia_muestreo']} Hz")
    print("\nArchivo JSON creado: wave_data.json")
else:
    print("Error: No hay datos para guardar en JSON.")

# %%