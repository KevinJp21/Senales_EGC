import wfdb
import matplotlib.pyplot as plt
record_name = r"D:\Proyectos\Señales_EGS\QT_Database\sel102/sel102"

record = wfdb.rdrecord(record_name)

fs = record.fs  # Muestras por segundo

samples_10s = int(fs * 10)

ecg_signal = record.p_signal[:samples_10s, 0]

time_axis = [i / fs for i in range(samples_10s)]
plt.figure(figsize=(12, 4))
plt.plot(time_axis, ecg_signal, label="ECG Signal", color='m')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("ECG - Primeros 10 segundos")
plt.legend()
plt.grid()
plt.show()