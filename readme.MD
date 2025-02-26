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

