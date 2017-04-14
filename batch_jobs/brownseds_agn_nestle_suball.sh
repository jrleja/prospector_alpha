#!/bin/bash
### Name of the job
### Requested number of cores
#SBATCH -n 32
### Requested number of nodes
#SBATCH -N 1
### Requested computing time in minutes
#SBATCH -t 1440
### Partition or queue name
#SBATCH -p conroy,general,conroy-intel
### memory per cpu, in MB
#SBATCH --mem-per-cpu=3000
### Job name
#SBATCH -J 'bseds_nest'
### output and error logs
#SBATCH -o brownseds_agn_nestle_%a.out
#SBATCH -e brownseds_agn_nestle_%a.err
### mail
#SBATCH --mail-type=END
#SBATCH --mail-user=joel.leja@gmail.com
IDFILE=$APPS"/threedhst_bsfh/data/brownseds_agn.ids"
SLURM_ARRAY_TASK_ID=2 # kill me
OBJID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDFILE")
srun -n $SLURM_NTASKS --mpi=pmi2 \
python $APPS/bsfh/scripts/prospector_nest.py \ 
--param_file=$APPS/threedhst_bsfh/parameter_files/brownseds_agn_nestle/brownseds_agn_nestle_params.py \
--objname=$OBJID \
--outfile=$APPS/threedhst_bsfh/results/brownseds_agn_nestle/brownseds_agn_nestle_$OBJID