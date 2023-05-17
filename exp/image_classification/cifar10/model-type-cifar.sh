# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='model-type-cifar'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="4"
NUM_GPU="2"

# ======== Configuration ========
DATASET="cifar10"
DATA_ROOT="../data/cifar10"

# ======== Determined Hyperparam ========
OPT="momentum_sgd"
PATIENCE="400"
NUM_CLASS="10"
MOMENTUM="0.9"
BATCH_SIZE="256"
LR="0.01"
WEIGHT_DECAY="1e-6"

# ======== Hyperparam for Sweep ========
MODEL_LIST=('resnet18_2_cifar' 'medium_mlp')
EPOCH_LIST=(150 400)


# Num of Param Information
# resnet18_2_cifar  : 44662922
# medium_mlp        : 46190602


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
                    --patience ${PATIENCE}

                    --wandb_exp_id ${count} \
                    --wandb_entity ${WANDB_ENTITY} \
                    --wandb_project_name ${WANDB_PROJECT_NAME} \
                    "
        CMD="sbatch run_cifar.sh ${SHELL_ARGS}"
        echo $CMD
        eval $CMD

    done
done
