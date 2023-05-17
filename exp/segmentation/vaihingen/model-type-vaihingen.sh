# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='model-type-vaihingen'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="1"
NUM_GPU="1"

# ======== Configuration ========
DATASET="vaihingen"
DATA_ROOT="../data/vaihingen"

# ======== Determined Hyperparam ========
OPT="momentum_sgd"
NUM_CLASS="6"
MOMENTUM="0.9"
BATCH_SIZE="10"
LR="0.01"
WEIGHT_DECAY="1e-5"

# ======== Hyperparam for Sweep ========
MODEL_LIST=('unet' 'segnet')
EPOCH_LIST=(26 50)

# ======== Hyper Parameter Search Loop  ========
for model in "${MODEL_LIST[@]}" ; do
    for epoch in "${EPOCH_LIST[@]}" ; do

        SHELL_ARGS="--seed ${SEED} \
                    --num_worker ${NUM_WORKER} \
                    --num_gpu ${NUM_GPU} \

                    --epochs_budget ${epoch} \
                    --model ${model} \
                    --dataset ${DATASET} \
                    --data_root ${DATA_ROOT} \
                    --num_classes ${NUM_CLASS} \

                    --batch_size ${BATCH_SIZE} \
                    --opt ${OPT} \
                    --momentum ${MOMENTUM} \
                    --lr ${LR} \
                    --weight_decay ${WEIGHT_DECAY} \

                    --wandb_exp_id ${count} \
                    --wandb_entity ${WANDB_ENTITY} \
                    --wandb_project_name ${WANDB_PROJECT_NAME} \
                    "

        CMD="sbatch run_vaihingen.sh ${SHELL_ARGS}"
        echo $CMD
        eval $CMD

    done
done
