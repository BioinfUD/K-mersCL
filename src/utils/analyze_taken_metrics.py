# Monday, June 12, 2017 9:32:09 AM
import sys
import pandas as pd
import argparse
import datetime

import os

def extract_time_to_datetime(str_time, str_date="2015-01-01", time_format="%H:%M:%S"):
    str_datetime = str_date + " " +str_time
    str_datetime_format = "%Y-%m-%d " + time_format
    fecha = datetime.datetime.strptime(str_datetime, str_datetime_format)
    return fecha

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Merges taken metrics by sar (CPU,IO) and nvidia-smi. Three files expected on input file: sar_cpu_file.log sar_mem_io_file.log nvidia_gpu.log")
    parser.add_argument('--input_path', dest="input_path",
                        help="Input path were the metrics are stored (this will be the same output path for plots and merged metrics)")
    parser.add_argument('--used_cores', dest="used_cores",
                        help="Number of cores used when taking the sar metrics")
    parser.add_argument('--date_of_metrics', dest="str_date",
                        help="Date when the execution was done, this is usefule since sar output files does not contains the date, just the hour")

    args = parser.parse_args()
    input_path = args.input_path
    n_cores = int(args.used_cores)
    str_date = int(args.str_date)
    return input_path, n_cores, str_date

def process_cpu_metrics(input_file, n_cores, str_date):
    # Process cpu file
    with open(input_file, "r") as file_cpu:
        fechas = []
        valores = []
        for line in file_cpu:
            if "all" in line:
                splits = line.split()
                cpu_value = float(splits[-1].replace(",", "."))
                used_total = (100 - cpu_value) * n_cores
                hora = extract_time_to_datetime(splits[0], str_date=str_date)
                fechas.append(hora)
                valores.append(used_total)
    series_cpu = pd.Series(valores, fechas)
    return series_cpu

def proces_memio_metrics(mem_io_file, str_date):
    # Process memio file
    with open(mem_io_file, "r") as file_mem_io:
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
                bt_written = float(splits[-1].replace(",", "."))
                bt_read = float(splits[-2].replace(",", "."))
                fecha = extract_time_to_datetime(splits[0], str_date=str_date)
                bts_written.append(bt_written)
                bts_read.append(bt_read)
                fechas_io.append(fecha)
            if "kbmemused" in line:
                line = file_mem_io.readline() # Skip header line and read line with data
                splits = line.split()
                fecha = extract_time_to_datetime(splits[0], str_date=str_date)
                memused = float(splits[2].replace(",", "."))
                vals_mem.append(memused)
                fechas_mem.append(fecha)
    series_written = pd.Series(bts_written, fechas_io)
    series_read = pd.Series(bts_read, fechas_io)
    series_mem = pd.Series(vals_mem, fechas_mem)
    return series_mem, series_read, series_written

def process_gpu_metrics(gpu_file):
    pass

def merge_metrics(input_path, n_cores, str_date):
    cpu_metrics_file = os.path.join(input_path, "sar_cpu_file.log")
    series_cpu = process_cpu_metrics(cpu_metrics_file, n_cores, str_date)

    mem_io_file = os.path.join(input_path, "sar_mem_io_file.log")
    series_mem, series_read, series_written = proces_memio_metrics(mem_io_file, str_date)

    gpu_file = os.path.join(input_path, "nvidia_gpu.log")
    process_gpu_metrics(gpu_file)

    # Join series
    df = pd.concat([series_cpu, series_mem, series_read, series_written], axis=1)
    df.columns =["CPU", "MEM", "READ", "WRITE"]

    # Output file
    merged_xlsx_file = os.path.join(input_path, "merged_metrics.xlsx")
    writer = pd.ExcelWriter(merged_xlsx_file)
    df.to_excel(writer, "Metrics")
    writer.save()

if __name__ == "__main__":
    input_path, n_cores, str_date = parse_arguments()
    merge_metrics(input_path, n_cores, str_date)


