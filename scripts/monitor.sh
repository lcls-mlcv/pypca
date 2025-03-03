#!/bin/bash

OUTPUT_FILE="../memory_logs/memory_usage_${SLURM_JOB_ID}.log"
INTERVAL=600 #every 10min

CPU_THRESHOLD=200 
GPU_THRESHOLD=30   

echo "Begin monitoring memory consumption for job ${SLURM_JOB_ID}" > $OUTPUT_FILE

while true; do
    echo "-------------------------------------------" >> $OUTPUT_FILE
    date >> $OUTPUT_FILE
    
    # CPU (GB)
    echo "Top 3 activities with most CPU memory consumption (GB):" >> $OUTPUT_FILE
    ps -u $USER -o pid,rss,command --sort=-rss | head -n 3 | awk '{printf "%s %.2f %s\n", $1, $2/1024/1024, $3}' >> $OUTPUT_FILE
    
    # Total CPU (GB)
    total_mem=$(ps -u $USER -o rss | awk '{sum+=$1} END {printf "%.2f", sum/1024/1024}')
    echo "Total CPU memory consumption: ${total_mem} GB" >> $OUTPUT_FILE
    
    # Vérification du seuil CPU
    if (( $(echo "$total_mem > $CPU_THRESHOLD" | bc -l) )); then
        echo "WARNING : HIGH TOTAL MEMORY CONSUMPTION" >> $OUTPUT_FILE
    fi
    
    # GPU memory consumption (GB)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU memory consumption (GB):" >> $OUTPUT_FILE
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{
            used=$2/1024
            total=$3/1024
            printf "GPU %s: %.2f / %.2f GB\n", $1, used, total
            if (used > '"$GPU_THRESHOLD"') {
                print "WARNING : HIGH GPU MEMORY CONSUMPTION (GPU " $1 ")"
            }
        }' >> $OUTPUT_FILE
    fi
    
    sleep $INTERVAL
done
