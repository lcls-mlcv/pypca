#!/bin/bash

OUTPUT_FILE="../memory_logs/memory_usage_${SLURM_JOB_ID}.log"
INTERVAL=900 #every 15min

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
    
    # GPU memory consumption (GB)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU memory consumption (GB):" >> $OUTPUT_FILE
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "GPU %s: %.2f / %.2f GB\n", $1, $2/1024, $3/1024}' >> $OUTPUT_FILE
    fi
    
    sleep $INTERVAL
done
