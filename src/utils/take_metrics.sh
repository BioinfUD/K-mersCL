LC_TIME='C' sar -P ALL 2 99999 | compress > $1-cpu.gz &
LC_TIME='C' sar -b -r 2 99999 | compress > $1-mem-io.gz &
while true; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --id=05:00.0 --format=csv | ts %.s >> $1-nvidia-log; sleep 1;  done
