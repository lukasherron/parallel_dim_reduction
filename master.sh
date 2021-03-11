#!/bin/bash
for l in {0..5}
do
   for k in {6..28}
   do
      sed -i "108s/.*/num_L_components = ${l}/" parallelized_PDR.py
      sleep 10s
      sed -i "109s/.*/num_K_components = ${k}/" parallelized_PDR.py
      sleep 10s
      sed -i "2s/.*/#SBATCH --job-name=k=${k}_L=${l}_PDR/" /blue/pdixit/lukasherron/parallel_dim_reduction/batch_scripts/parallel_dim_reduction.sh
      sleep 30s
      sbatch /blue/pdixit/lukasherron/parallel_dim_reduction/batch_scripts/parallel_dim_reduction.sh
      sleep 90s
   done
done
