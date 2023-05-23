# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='batchsize-imagenet'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="8"
NUM_GPU="4"

# ======== Configuration ========
DATASET="imagenet"
DATA_ROOT="None"

# ======== Determined Hyperparam ========
EPOCHS="180"
NUM_CLASS="1000"
MOMENTUM="0.9"
OPT="momentum_sgd"
BASE_LR="0.0005"
MODEL="resnet50_1"

WEIGHT_DECAY="1e-6"
LR_SCHEDULE="True"
WARMUP_EPOCHS="3"

# ======== Hyperparam for Sweep ========
BATCH_SIZE_LIST=(64 128 256 512)


# ======== Hyper Parameter Search Loop  ========
for batch_size in "${BATCH_SIZE_LIST[@]}" ; do

    DIV_B=$(python -c "print($batch_size/${BATCH_SIZE_LIST[0]})")
    MULTIPLIER=$(echo "scale=20;sqrt($DIV_B)" | bc)
    LR=$(python -c "print($BASE_LR*$MULTIPLIER)")

    SHELL_ARGS="--seed ${SEED} \
                --num_worker ${NUM_WORKER} \
                --num_gpu ${NUM_GPU} \

                --epochs ${EPOCHS} \
                --model ${MODEL} \
                --dataset ${DATASET} \
                --data_root ${DATA_ROOT} \
                --num_classes ${NUM_CLASS} \

                --batch_size ${batch_size} \
                --opt ${OPT} \
                --momentum ${MOMENTUM} \
                --lr ${LR} \

                --weight_decay ${WEIGHT_DECAY} \
                --use_scheduler ${LR_SCHEDULE} \
                --warmup_epochs ${WARMUP_EPOCHS} \

                --wandb_exp_id ${count} \
                --wandb_entity ${WANDB_ENTITY} \
                --wandb_project_name ${WANDB_PROJECT_NAME} \
                "

    CMD="sbatch run_imagenet.sh ${SHELL_ARGS}"
    echo $CMD
    eval $CMD

done