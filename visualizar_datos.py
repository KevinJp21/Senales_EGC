#%% Importación de librerías
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Configurar el backend interactivo
import matplotlib.pyplot as plt
plt.ion()  # Activar modo interactivo
import json

#%% Cargar los datos desde JSON
with open('datos_ondas.json', 'r') as f:
    datos = json.load(f)

# Convertir los datos a arrays de numpy
X = np.array(datos['senales_normalizadas'])
y = np.array(datos['etiquetas'])
tiempos = np.array(datos['tiempos'])
duraciones = np.array(datos['duraciones'])
metadata = datos['metadata']

#%% Mostrar información general
print("\n📊 Información general de los datos:")
print("=" * 50)
print(f"Frecuencia de muestreo: {metadata['frecuencia_muestreo']} Hz")
print(f"  → Tiempo entre muestras: {1000/metadata['frecuencia_muestreo']:.2f} ms")
print(f"  → Muestras por segundo: {metadata['frecuencia_muestreo']} muestras")
print("-" * 50)
print(f"Ventana de análisis: {metadata['ventana_ms']} ms")
print(f"  → Muestras por ventana: {metadata['muestras_por_ventana']}")
print(f"  → Tiempo entre muestras en ventana: {metadata['ventana_ms']/metadata['muestras_por_ventana']:.2f} ms")
print("-" * 50)
print(f"Número total de ondas R detectadas: {len(X)}")
print(f"Dimensiones de cada onda: {X[0].shape}")
print("=" * 50)

#%% Definición de funciones de visualización
def visualizar_onda_interactiva(index):
    """Visualiza una onda específica con sus etiquetas de forma interactiva"""
    if not plt.get_fignums():
        global fig, ax1, ax2
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Limpiar los ejes
    ax1.clear()
    ax2.clear()
    
    # Subplot para la señal normalizada
    ax1.plot(X[index], 'c-', label='Señal normalizada')
    puntos_pico = np.where(y[index] == 1)[0]
    ax1.scatter(puntos_pico, X[index][puntos_pico], color='r', s=100, label='Pico R (1)')
    ax1.set_title(f'Onda R #{index+1} de {len(X)}')
    ax1.set_ylabel('Amplitud Normalizada')
    ax1.grid(True)
    ax1.legend()
    
    # Subplot para las etiquetas
    ax2.plot(y[index], 'g-', label='Etiquetas')
    ax2.set_ylabel('Etiqueta (0/1)')
    ax2.set_xlabel('Muestras')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    
    # Calcular tiempos
    ms_por_muestra = metadata['ventana_ms'] / metadata['muestras_por_ventana']
    tiempo_pico_ms = puntos_pico[0] * ms_por_muestra
    
    # Imprimir información en consola
    print(f"\n📊 Información de la Onda R #{index+1}:")
    print("=" * 50)
    print(f"Frecuencia de muestreo: {metadata['frecuencia_muestreo']} Hz")
    print(f"  → {1000/metadata['frecuencia_muestreo']:.2f} ms entre muestras")
    print("-" * 50)
    print(f"Ventana de análisis: {metadata['ventana_ms']} ms ({metadata['muestras_por_ventana']} muestras)")
    print(f"Posición del pico R: Muestra {puntos_pico[0]} de {metadata['muestras_por_ventana']}")
    print(f"  → Tiempo del pico: {tiempo_pico_ms:.2f} ms desde inicio de ventana")
    print(f"  → Valor normalizado en el pico: {X[index][puntos_pico[0]]:.5f}")
    print(f"  → Duración de la onda: {duraciones[index]:.2f} s")
    print("=" * 50)

def on_key(event):
    """Maneja los eventos de teclado para la navegación"""
    global current_index
    if event.key == 'right':
        current_index = (current_index + 1) % len(X)
        visualizar_onda_interactiva(current_index)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(X)
        visualizar_onda_interactiva(current_index)
    elif event.key == 'escape':
        plt.close()

def visualizar_todas_ondas():
    """Visualiza todas las ondas superpuestas"""
    plt.figure(figsize=(12, 6))
    
    # Graficar todas las ondas
    for i in range(len(X)):
        plt.plot(X[i], 'c-', alpha=0.3)
    
    plt.title('Todas las ondas R superpuestas')
    plt.xlabel('Muestras')
    plt.ylabel('Amplitud Normalizada')
    plt.grid(True)
    plt.show()

#%% Visualización interactiva de ondas individuales
# Ejecutar esta celda para ver las ondas una por una
# Use las flechas ← → para navegar y ESC para salir
current_index = 0
visualizar_onda_interactiva(current_index)
plt.show(block=True)

#%% Visualización de todas las ondas superpuestas
# Ejecutar esta celda para ver todas las ondas juntas
visualizar_todas_ondas()

# %%
