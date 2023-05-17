#!/bin/bash
#SBATCH --cpus-per-task=2 
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out

# ======== Module, Virtualenv and Other Dependencies ======
source ../../env/cluster_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH

# ======== Configuration ========
PROGRAM="main_segmentation.py"
pushd ../../../src
PYTHON_ARGS=$@

# ======== Execution ========
CMD="python ${PROGRAM} ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd
