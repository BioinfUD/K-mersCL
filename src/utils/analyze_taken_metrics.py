# Monday, June 12, 2017 9:32:09 AM
import datetime
import sys
import pandas as pd

def extract_time_to_datetime(str_time, str_date="2015-01-01", time_format="%H:%M:%S"):
    str_datetime = str_date + " " +str_time
    str_datetime_format = "%Y-%m-%d " + time_format
    fecha = datetime.datetime.strptime(str_datetime, str_datetime_format)
    return fecha

"""
process_performance_data.py sar_cpu_file.log sar_mem_io_file.log nvidia_gpu.log
"""

# Fecha en que se ejecuta el benchmark
# Used to complement sar data
fecha_datos = datetime.datetime(2017, 4, 15, 20, 0)
str_date = fecha_datos.strftime("%Y-%m-%d")

n_cores = 56
archivo_cpu = sys.argv[1] # File from sar log contaning hour
archivo_io_mem = sys.argv[2] # File from sar log contaning hour

if len(sys.argv)>=4:
    archivo_gpu = sys.argv[3]

# Process cpu file
with open(archivo_cpu, "r") as file_cpu:
    fechas = []
    valores = []
    for line in file_cpu:
        if "all" in line:
            splits = line.split()
            cpu_value = float(splits[-1])
            used_total = (100 - cpu_value) * n_cores
            hora = extract_time_to_datetime(splits[0], str_date=str_date)
            fechas.append(hora)
            valores.append(used_total)
series_cpu = pd.Series(valores, fechas)

# Process memio file
with open(archivo_io_mem, "r") as file_mem_io:
    fechas_io = []
    fechas_mem = []
    bts_written = []
    bts_read = []
    vals_mem = []
    while True:
        line = file_mem_io.readline()
        if not line: break
        if "bwrtn" in line:
            line = file_mem_io.readline() # Skip header line and read line with data
            splits = line.split()
            bt_written = float(splits[-1])
            bt_read = float(splits[-2])
            fecha = extract_time_to_datetime(splits[0], str_date=str_date)
            bts_written.append(bt_written)
            bts_read.append(bt_read)
            fechas_io.append(fecha)
        if "kbmemused" in line:
            line = file_mem_io.readline() # Skip header line and read line with data
            splits = line.split()
            fecha = extract_time_to_datetime(splits[0], str_date=str_date)
            memused = float(splits[2])
            vals_mem.append(memused)
            fechas_mem.append(fecha)

series_written = pd.Series(bts_written, fechas_io)
#print series_written
series_read = pd.Series(bts_read, fechas_io)
#print series_read
series_mem = pd.Series(vals_mem, fechas_mem)
#print series_mem
#print series_cpu
# Join series
df = pd.concat([series_cpu, series_mem, series_read, series_written], axis=1)
df.columns =["CPU", "MEM", "READ", "WRITE"]
