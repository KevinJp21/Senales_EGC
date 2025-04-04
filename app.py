import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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

# Detección de ondas R
peaks, _ = find_peaks(ecg_signal, height=np.max(ecg_signal) * 0.6, distance=fs*0.6)

# Función para detectar el inicio y el final de cada onda R
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

# Detectar inicios y finales
inicios, finales = detectar_limites(ecg_signal, peaks)

# Extraer las ondas R y almacenarlas en una lista con sus tiempos
ondas_r = [ecg_signal[inicio:final] for inicio, final in zip(inicios, finales)]
tiempos_r = [time_axis[inicio:final] for inicio, final in zip(inicios, finales)]

# Mostrar la señal ECG completa con ondas R resaltadas
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

# Función para visualizar cada onda R individualmente
def visualizar_onda(index):
    plt.clf()
    plt.plot(tiempos_r[index], ondas_r[index], color='c', linewidth=2)
    plt.title(f'Onda R {index+1} de {len(ondas_r)}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.draw()

    # Imprimir los datos en la consola con tiempos
    print(f"\n📊 Datos de la Onda R {index+1} (Tiempo - Amplitud):")
    for t, amp in zip(tiempos_r[index], ondas_r[index]):
        print(f"{t:.5f} s -> {amp:.5f}")

# Función de control para cambiar entre ondas
def on_key(event):
    global index
    if event.key == "right":
        index = (index + 1) % len(ondas_r)  # Avanzar
    elif event.key == "left":
        index = (index - 1) % len(ondas_r)  # Retroceder
    visualizar_onda(index)

# Inicializar visualización de ondas R individuales
index = 0
fig, ax = plt.subplots()
fig.canvas.mpl_connect("key_press_event", on_key)
visualizar_onda(index)
plt.show()
