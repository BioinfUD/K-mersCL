import pandas as pd
import numpy as np
from sys import argv
import re


def extract_nvidia_max_memory(merged_file):
    return np.nanmax(merged_file['GPU'])

def extract_anytool_max_memory(merged_file):
    return (np.nanmax(merged_file['MEM']) - np.nanmin(merged_file['MEM']))/1000.0


def extract_kmc_time(log_file):
    time = 0.0
    for line in log_file:
        if "Time read processing : " in line:
            line = line.split(",")[1]
            content = line.replace("*", "").replace("Time read processing : ", "").replace("ns.", "").strip()
            time += float(content)
    return time/(10**9)

def extract_mspk_time(log_file):
    for line in log_file:
        if "Total algorithm time" in line:
            line = line.split(",")[1]
            time = line.replace("Total algorithm time:","").replace("s", "").strip()
            if "min" in time:
		time = float(time.replace("min", "").strip())
		return time * 60
            return float(time)

def extract_kmercl_signature_time(log_file):
    for line in log_file:
        if "Total algorithm time" in line:
            line = line.split(",")[1]
            time = line.replace("Total algorithm time:","").replace("s", "").strip()
            return float(time)

def extract_kmercl_time(log_file):
    for line in log_file:
        if "Kernel execution took " in line:
            line = line.split(",")[1]
            return float(line.replace("Kernel execution took ", "").strip())


def extract_transfer_time(log_file):
    total = 0.0
    for line in log_file:
        if "Copying data took " in line:
            line = line.split(",")[1]
            total += float(line.replace(" Copying data took ", "").replace("seconds","").strip())
    return total


def extract_metrics(path):
    merged_metrics_path = "{}/metrics/merged_metrics.xlsx".format(path)
    tool_log_path = "{}/metrics/tool_log.csv".format(path)
    log_file = open(tool_log_path)
    merged_file = pd.read_excel(merged_metrics_path)
    transfer_time = None
    if "kmerscl-" in path:
        transfer_time = extract_transfer_time(log_file)
        log_file.seek(0)
        time = extract_kmercl_time(log_file)
        memory = extract_nvidia_max_memory(merged_file)
        log_file = open(tool_log_path)
    elif "kmerscl_signature" in path:
        transfer_time = extract_transfer_time(log_file)
        log_file.seek(0)
        time = extract_kmercl_time(log_file)
        memory = extract_nvidia_max_memory(merged_file)
        log_file = open(tool_log_path)
    elif "kmc" in path:
        time = extract_kmc_time(log_file)
        memory = extract_anytool_max_memory(merged_file)
    elif "mspk" in path:
        time = extract_mspk_time(log_file)
        memory = extract_anytool_max_memory(merged_file)
    else:
        print "Not valid tool"
        exit()
    print path
    m = re.match(r".*/(?P<tool>\w+)-k(?P<kmer>\w+)-m(?P<mmer>\w+)-r(?P<read_lenght>\w+)-(?P<seq>\w+)_(?P<read_millions>\w+)m_.*", path)
    row = m.groupdict()
    row['mem'] = memory
    row['time'] = time


    print "Tool and params: {}".format(path)
    print "Used memory {} mb".format(memory)
    print "Time {} seconds".format(time)
    if transfer_time:
        row['tt'] = transfer_time
        print "Transfer time {} seconds".format(transfer_time)


    return values_to_list(row)


def values_to_list(d):
    nd = {}
    for k, v in d.iteritems():
        nd[k] = [v]
    return nd

# df =  pd.DataFrame(columns=['tool', 'kmer', 'mmer', 'read_lenght', 'seq', 'time', 'mem','read_millions'])
writer = pd.ExcelWriter('output.xlsx')
paths = argv[1:]
# print df
print extract_metrics(paths[0])
frames = []
for path in paths[:]:
    df2 = pd.DataFrame.from_dict(extract_metrics(path))
    frames.append(df2)

df = pd.concat(frames)

df.to_excel(writer,'KmerCL_benchmark')
writer.save()
