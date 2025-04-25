#%% Importación de librerías
import wfdb
import matplotlib
matplotlib.use('TkAgg')  # Configurar el backend interactivo
import matplotlib.pyplot as plt
plt.ion()  # Activar modo interactivo
import numpy as np
from scipy.signal import find_peaks

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

#%% Funciones de visualización interactiva
def visualizar_onda(index):
    if not plt.get_fignums():  # Si no hay figuras abiertas, crear una nueva
        global fig, ax
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.canvas.mpl_connect("key_press_event", on_key)
    
    ax.clear()
    ax.plot(tiempos_r[index], ondas_r[index], color='c', linewidth=2)
    ax.set_title(f'Onda R {index+1} de {len(ondas_r)}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    ax.grid(True)
    plt.draw()
    plt.pause(0.1)  # Pequeña pausa para actualizar la figura

    # Imprimir los datos en la consola con tiempos
    print(f"\n📊 Datos de la Onda R {index+1} (Tiempo - Amplitud):")
    for t, amp in zip(tiempos_r[index], ondas_r[index]):
        print(f"{t:.5f} s -> {amp:.5f}")

def on_key(event):
    global index
    if event.key == "right":
        index = (index + 1) % len(ondas_r)  # Avanzar
        visualizar_onda(index)
    elif event.key == "left":
        index = (index - 1) % len(ondas_r)  # Retroceder
        visualizar_onda(index)

#%% Inicialización de la visualización interactiva
# Asegurarse de que todas las variables necesarias estén definidas
if 'ondas_r' in locals() and 'tiempos_r' in locals():
    index = 0
    visualizar_onda(index)
    plt.show(block=True)  # Bloquear la ejecución para mantener la ventana abierta
else:
    print("Error: Las variables 'ondas_r' y 'tiempos_r' no están definidas. Ejecuta las celdas anteriores primero.")

#%% Estandarización de ondas R a 120ms
# Preparar las ventanas centradas en cada pico R (duración de 120 ms)
ventana_ms = 120
ventana_muestras = int((ventana_ms / 1000) * fs)

ondas_r_estandarizadas = []
tiempos_r_estandarizados = []
duraciones_r = []

for pico in peaks:
    mitad = ventana_muestras // 2
    inicio = max(pico - mitad, 0)
    final = min(pico + mitad, len(ecg_signal))

    if final - inicio < ventana_muestras:
        if inicio == 0:
            final = ventana_muestras
        elif final == len(ecg_signal):
            inicio = len(ecg_signal) - ventana_muestras

    ondas_r_estandarizadas.append(ecg_signal[inicio:final])
    tiempos_r_estandarizados.append(time_axis[inicio:final])
    duraciones_r.append(time_axis[final - 1] - time_axis[inicio])

#%% Visualización interactiva de ondas R estandarizadas
def visualizar_onda_estandarizada(index):
    if not plt.get_fignums():  # Si no hay figuras abiertas, crear una nueva
        global fig_est, ax_est
        fig_est, ax_est = plt.subplots(figsize=(10, 6))
        fig_est.canvas.mpl_connect("key_press_event", on_key_estandarizada)
    
    ax_est.clear()
    ax_est.plot(tiempos_r_estandarizados[index], ondas_r_estandarizadas[index], color='c', linewidth=2)
    ax_est.set_title(f'Onda R estandarizada {index+1} de {len(ondas_r_estandarizadas)}\nDuración: {duraciones_r[index]:.2f} ms')
    ax_est.set_xlabel('Tiempo (s)')
    ax_est.set_ylabel('Amplitud')
    ax_est.grid(True)
    plt.draw()
    plt.pause(0.1)

    print(f"\n📊 Datos de la Onda R estandarizada {index+1}:")
    print(f"Duración: {duraciones_r[index]:.2f} ms")
    print("Tiempo (s) -> Amplitud")
    for t, amp in zip(tiempos_r_estandarizados[index], ondas_r_estandarizadas[index]):
        print(f"{t:.5f} -> {amp:.5f}")

def on_key_estandarizada(event):
    global index_est
    if event.key == "right":
        index_est = (index_est + 1) % len(ondas_r_estandarizadas)
        visualizar_onda_estandarizada(index_est)
    elif event.key == "left":
        index_est = (index_est - 1) % len(ondas_r_estandarizadas)
        visualizar_onda_estandarizada(index_est)

# Inicializar visualización de ondas R estandarizadas
if 'ondas_r_estandarizadas' in locals():
    index_est = 0
    visualizar_onda_estandarizada(index_est)
    plt.show(block=True)
else:
    print("Error: Las ondas R estandarizadas no están definidas. Ejecuta las celdas anteriores primero.")

#%% Visualización de ondas R estandarizadas concatenadas
# Concatenar todas las ondas R estandarizadas
ecg_reconstruido = np.concatenate(ondas_r_estandarizadas)

# Normalizar amplitud al rango [0, 1]
min_val = np.min(ecg_reconstruido)
max_val = np.max(ecg_reconstruido)
ecg_normalizada = (ecg_reconstruido - min_val) / (max_val - min_val)

# Reconstruir el eje de tiempo desde 0
tiempo_total = []
tiempo_actual = 0
for duracion, tiempos in zip(duraciones_r, tiempos_r_estandarizados):
    delta_tiempo = tiempos - tiempos[0]
    tiempo_segmento = tiempo_actual + delta_tiempo
    tiempo_total.extend(tiempo_segmento)
    tiempo_actual = tiempo_segmento[-1] + 1  # dejar 1 ms de separación

# Graficar señal reconstruida normalizada
plt.figure(figsize=(14, 4))
plt.plot(tiempo_total, ecg_normalizada, color='darkblue', linewidth=1.5)
plt.title('Señal ECG reconstruida con ondas R estandarizadas (normalizada a 0–1)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud Normalizada (0–1)')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

