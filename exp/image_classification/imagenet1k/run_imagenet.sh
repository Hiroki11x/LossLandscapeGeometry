#!/bin/bash
#SBATCH --cpus-per-task=8 
#SBATCH --gres=gpu:4 
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH -o slurm-%j.out

# ======== Module, Virtualenv and Other Dependencies ======
source ../../env/cluster_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH


# # ======== Data Copy [Val] ========

mkdir $SLURM_TMPDIR/val
t0=$(date +%s)
tar fx /PATH_TO_IMAGENET/ILSVRC2012_img_val.tar -C $SLURM_TMPDIR/val
cp valprep.sh $SLURM_TMPDIR/val
echo val
echo "ls val | wc -l"
ls $SLURM_TMPDIR/val | wc -l
t1=$(date +%s)

echo "................................"
echo "[Val Data] Time for stage-in: $((t1 - t0)) sec"
echo "................................"

t0=$(date +%s)
pushd $SLURM_TMPDIR/val
sh valprep.sh
echo val
echo "ls val | wc -l"
ls $SLURM_TMPDIR/val | wc -l
popd
t1=$(date +%s)

echo "................................"
echo "[Val Data] Time for move: $((t1 - t0)) sec"
echo "................................"


# # ======== Data Copy [Train] ========
mkdir $SLURM_TMPDIR/train

t0=$(date +%s)
tar fx /PATH_TO_IMAGENET/ILSVRC2012_img_train.tar -C $SLURM_TMPDIR/train
cp untarimage.sh $SLURM_TMPDIR/train
echo train
echo "ls train | wc -l"
ls $SLURM_TMPDIR/train | wc -l
t1=$(date +%s)

echo "................................"
echo "[Train Data] Time for stage-in: $((t1 - t0)) sec"
echo "................................"


t0=$(date +%s)
pushd $SLURM_TMPDIR/train
sh untarimage.sh
popd
t1=$(date +%s)

echo "................................"
echo "[Train Data] Time for untar: $((t1 - t0)) sec"
echo "................................"




# ======== Configuration ========
PROGRAM="main_image_classification.py"
pushd ../../../src
PYTHON_ARGS=$@


# ======== Execution ========
CMD="python ${PROGRAM} ${PYTHON_ARGS}"
echo $CMD
eval $CMD

popd