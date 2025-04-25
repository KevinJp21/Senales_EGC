# Procesamiento de Señales ECG

Este proyecto realiza un procesamiento de señales ECG usando Python. El objetivo es detectar ondas R en un registro de 10 segundos, visualizarlas individualmente, y almacenar los datos procesados para su análisis. A continuación, se describe brevemente el flujo de trabajo:

## Flujo de Trabajo

1. **Lectura del archivo ECG** (`appCeldas.py`)
   - Se carga una señal ECG del conjunto de datos QT Database mediante la librería wfdb
   - Se extraen los primeros 10 segundos del primer canal

2. **Procesamiento de la señal**
   - Se construye un eje de tiempo para la señal
   - Se detectan los picos R usando find_peaks
   - Se extraen ventanas de 120 ms centradas en cada pico R
   - Se normalizan las amplitudes de las ondas
   - Se generan etiquetas binarias (1 en el pico R, 0 en el resto)

3. **Almacenamiento de datos** (`datos_ondas.json`)
   - Los datos procesados se guardan en formato JSON con la siguiente estructura:
     - `senales_normalizadas`: Amplitudes normalizadas de cada onda
     - `etiquetas`: Etiquetas binarias para cada muestra
     - `tiempos`: Tiempos correspondientes a cada muestra
     - `duraciones`: Duración de cada ventana
     - `metadata`: Información sobre frecuencia de muestreo y parámetros

4. **Visualización** (`visualizar_datos.py`)
   - Visualización interactiva de ondas individuales
   - Visualización de todas las ondas superpuestas
   - Información detallada de cada onda:
     - Tiempo absoluto y relativo del pico R
     - Valor normalizado en el pico
     - Duración de la ventana

## Estructura del Proyecto

```
.
├── appCeldas.py        # Script de procesamiento principal
├── visualizar_datos.py # Script de visualización
├── datos_ondas.json   # Datos procesados
├── QT_Database/       # Carpeta con los archivos de señales ECG
│   └── sel30/        # Ejemplo de registro ECG
└── README.md         # Este archivo
```

## Uso

1. Asegúrate de tener los archivos de señales ECG en la carpeta `QT_Database`

2. Procesa los datos:
   ```bash
   python appCeldas.py
   ```
   Esto generará el archivo `datos_ondas.json`

3. Visualiza los resultados:
   ```bash
   python visualizar_datos.py
   ```

4. Interactúa con la visualización:
   - Ejecuta las celdas individualmente en tu IDE
   - Para ver ondas individuales:
     - Usa la flecha derecha (→) para la siguiente onda
     - Usa la flecha izquierda (←) para la onda anterior
     - Presiona ESC para cerrar la ventana
   - Para ver todas las ondas superpuestas, ejecuta la celda correspondiente

## Notas
- La señal ECG debe estar en formato compatible con wfdb
- Los datos procesados se normalizan para facilitar su análisis
- Cada onda R se estandariza a una ventana de 120 ms
- Los tiempos se almacenan tanto en absolutos como relativos
- La visualización permite analizar cada onda en detalle

# Creación de Entorno Virtual e Instalación de Dependencias

Este documento explica cómo crear un entorno virtual en Python e instalar las dependencias necesarias para ejecutar el código correctamente.

## 1. Crear un Entorno Virtual

Python proporciona la herramienta `venv` para crear entornos virtuales y evitar conflictos con dependencias instaladas de forma global en el dispositivo. Para configurarlo, sigue estos pasos:

### En Windows (CMD o PowerShell):
```sh
python -m venv venv
```

Para activar el entorno:
```sh
.\.venv\Scripts\activate
```

### En macOS y Linux:
```sh
python3 -m venv venv
```

Para activar el entorno:
```sh
source ./venv/bin/activate
```

---

## 2. Instalar Dependencias

Dentro del entorno virtual activado, instala las dependencias necesarias ejecutando:

```sh
pip install -r requirements.txt
```

---

## 3. Ejecutar

Para procesar los datos:
```sh
python appCeldas.py
```

Para visualizar los resultados:
```sh
python visualizar_datos.py
```

Usa tu IDE favorito para ejecutar las celdas individualmente y explorar los datos interactivamente.

---

## 4. Desactivar el Entorno Virtual

Cuando termines de trabajar en el entorno virtual, puedes desactivarlo con:

```sh
deactivate
```

Esto restaurará el entorno global de Python.

---

