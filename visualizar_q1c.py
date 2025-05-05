import wfdb
import os
import csv

# Ruta al archivo base (sin extensión)
base = './QT_Database/sel30/sel30'

# Verificar si existe el archivo de anotaciones .q1c
ruta_q1c = base + '.q1c'
if not os.path.exists(ruta_q1c):
    print(f'No se encontró el archivo: {ruta_q1c}')
    exit(1)

# Leer las anotaciones usando wfdb.rdann
anotaciones = wfdb.rdann(base, 'q1c')

# Preparar archivo CSV
csv_filename = 'ondas_R.csv'
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Sample #', 'Type'])  # encabezado

    print(f"\nOndas tipo 'N' (agrupadas con paréntesis):")
    print("Time\t\tSample #\tType")

    # Contador para el total de ondas R (grupos completos)
    total_ondas_r = 0

    # Recorrer anotaciones buscando patrones ( N )
    i = 0
    while i < len(anotaciones.sample) - 2:
        simbolo1 = anotaciones.symbol[i]
        simbolo2 = anotaciones.symbol[i + 1]
        simbolo3 = anotaciones.symbol[i + 2]
        
        if simbolo1 == '(' and simbolo2 == 'N' and simbolo3 == ')':
            total_ondas_r += 1  # Incrementar contador por cada grupo completo
            
            for j in range(3):  # imprimir y guardar los tres juntos: ( N )
                idx = i + j
                tiempo = anotaciones.sample[idx] / anotaciones.fs
                time_str = f"{tiempo:.3f}"  # Tiempo en segundos con 3 decimales
                sample = anotaciones.sample[idx]
                tipo = anotaciones.symbol[idx]
                print(f"{time_str}\t{sample}\t{tipo}")
                writer.writerow([time_str, sample, tipo])
            print()
            i += 3
        else:
            i += 1

print(f"\nCSV guardado como '{csv_filename}'")
print(f"Total de ondas R (grupos '(' 'N' ')') encontradas: {total_ondas_r}")

# Verificar el contenido del archivo guardado
print("\nVerificando datos del archivo CSV guardado:")
try:
    with open(csv_filename, mode='r') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)  # Leer encabezados
        
        # Contadores para verificación
        total_registros = 0
        tipo_inicio = 0
        tipo_pico = 0 
        tipo_fin = 0
        
        for row in csv_reader:
            total_registros += 1
            if row[2] == '(':
                tipo_inicio += 1
            elif row[2] == 'N':
                tipo_pico += 1
            elif row[2] == ')':
                tipo_fin += 1
        
        # Calcular ondas completas
        ondas_completas = min(tipo_inicio, tipo_pico, tipo_fin)
        
        print(f"Total de registros en CSV: {total_registros}")
        print(f"Anotaciones de inicio '(': {tipo_inicio}")
        print(f"Anotaciones de pico 'N': {tipo_pico}")
        print(f"Anotaciones de fin ')': {tipo_fin}")
        print(f"Ondas R completas (tríos '(', 'N', ')'): {ondas_completas}")
        
        # Verificar coherencia
        if ondas_completas * 3 != total_registros:
            print("ADVERTENCIA: El número de registros no es múltiplo de 3")
        if ondas_completas != total_ondas_r:
            print("ADVERTENCIA: Inconsistencia en el conteo de ondas R")
            
except Exception as e:
    print(f"Error al verificar el archivo CSV: {e}")

# Mostrar tamaño del archivo
try:
    file_size = os.path.getsize(csv_filename)
    print(f"\nTamaño del archivo CSV: {file_size} bytes")
except Exception as e:
    print(f"Error al obtener el tamaño del archivo: {e}")
