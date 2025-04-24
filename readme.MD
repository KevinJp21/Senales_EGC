# Procesamiento de Señales ECG

Este script realiza un procesamiento básico de una señal ECG usando Python. El objetivo es detectar ondas R en un registro de 10 segundos, visualizarlas individualmente, y reconstruir una señal a partir de ellas. A continuación, se describe brevemente el flujo de trabajo:

## Flujo de Trabajo

1. **Lectura del archivo ECG**
   - Se carga una señal ECG del conjunto de datos QT Database (Carpeta que contiene los archivos sel30 por ejemplo) mediante la librería wfdb, extrayendo solo los primeros 10 segundos del primer canal.

2. **Eje de tiempo en milisegundos**
   - Se construye un eje de tiempo para graficar la señal con precisión temporal.

3. **Detección de ondas R**
   - Se identifican los picos R (los más representativos del ciclo cardíaco) utilizando find_peaks, con un umbral basado en la amplitud máxima de la señal.

4. **Extracción de segmentos centrados en picos R**
   - Se extraen ventanas de 120 ms centradas en cada pico R. Esto permite estandarizar la duración de cada onda para que todas las muestras tengan la misma longitud.

5. **Visualización**
   - Se muestra la señal ECG original con los picos R resaltados.
   - Se permite al usuario navegar entre ondas individuales usando las flechas del teclado, observando los valores de amplitud y tiempo de cada segmento.

6. **Gráfica final**
   - Se presentan las ondas R de forma concatenada. Esto permite visualizar cómo sería una señal ECG compuesta únicamente por los segmentos estandarizados de cada latido.

## Requisitos

Para ejecutar este script, necesitarás las siguientes librerías de Python:

```bash
pip install wfdb matplotlib numpy scipy
```

## Estructura del Proyecto

```
.
├── app.py              # Script principal
├── QT_Database/        # Carpeta con los archivos de señales ECG
│   └── sel30/         # Ejemplo de registro ECG
└── README.md          # Este archivo
```

## Uso

1. Asegúrate de tener los archivos de señales ECG en la carpeta `QT_Database`
2. Ejecuta el script:
   ```bash
   python app.py
   ```
3. Interactúa con la visualización:
   - Usa la flecha derecha (→) para ver la siguiente onda R
   - Usa la flecha izquierda (←) para ver la onda R anterior
   - Cierra la ventana para terminar la visualización

## Notas
- La señal ECG debe estar en formato compatible con wfdb
- El script está configurado para analizar los primeros 10 segundos de la señal
- Los parámetros de detección de picos pueden necesitar ajustes según la señal específica

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
.\.venv\Scripts\activate
```

---

## 2. Instalar Dependencias

Dentro del entorno virtual activado, instala las dependencias necesarias ejecutando:

```sh
pip install -r requirements.txt
```

---

## 3. Ejecutar

Para ejecutar escribe el siguiente comando
```sh
py app.py
```

A continuación, se abrirá una ventana con las señales. Al cerrar esa ventana, se abrirá otra mostrando cada onda R. Usa las direccionales del teclado para navegar hacia la izquierda o derecha.

---

## 4. Desactivar el Entorno Virtual

Cuando termines de trabajar en el entorno virtual, puedes desactivarlo con:

```sh
deactivate
```

Esto restaurará el entorno global de Python.

---

