#!/bin/bash
#SBATCH -J Kairo_Swarm
#SBATCH -e /work/scratch/kurse/kurs00093/%u/kairo/logs/swarm_%A.out
#SBATCH -o /work/scratch/kurse/kurs00093/%u/kairo/logs/swarm_%A.err
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH -A kurs00093           
#SBATCH -p kurs00093    
#SBATCH --reservation=kurs00093      
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=1024
#SBATCH --time=10

# --- Environment ---
module purge
module load gcc/13.1     # <--- MUST MATCH what you used to create the venv
module load python/3.12.8
source venv/bin/activate

cd $SLURM_SUBMIT_DIR
source venv/bin/activate

# Setup SUMO Paths
export SUMO_HOME=$(python -c "import sumo; print(sumo.SUMO_HOME)")
export PATH=$PATH:$SUMO_HOME/bin

# --- Directories ---
# Update scratch path to the 00093 folder
CENTRAL_SCRATCH="/work/scratch/kurse/kurs00093/$USER/kairo/sumo_data"
mkdir -p $CENTRAL_SCRATCH

# Ensure log directory exists (Slurm fails if this is missing)
mkdir -p /work/scratch/kurse/kurs00093/$USER/kairo/logs
echo "🚀 Starting Array Task $SLURM_ARRAY_TASK_ID on $(hostname)"

# --- Run Simulation ---
python backend/scripts/generate_traffic_data.py \
    --duration 3600 \
    --episodes 50 \
    --workers $SLURM_CPUS_PER_TASK \
    --job-id "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --output-dir $CENTRAL_SCRATCH \
    --skip-setup \
    --no-merge

echo "✅ Task $SLURM_ARRAY_TASK_ID finished."