# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='seed-cifar'

# ======== Environmental Setting ========
NUM_WORKER="4"
NUM_GPU="2"

# ======== Configuration ========
DATASET="cifar10"
DATA_ROOT="../data/cifar10"

# ======== Determined Hyperparam ========
MODEL='resnet18_1_cifar'
OPT="momentum_sgd"
EPOCHS="190" 
NUM_CLASS="10"
MOMENTUM="0.9"
BATCH_SIZE="256"
LR="0.01"
WEIGHT_DECAY="1e-6"

# ======== Hyperparam for Sweep ========
SEED_LIST=("2019" "2020" "2021" "2022" "2023")


# ======== Hyper Parameter Search Loop  ========
for seed in "${SEED_LIST[@]}" ; do

    SHELL_ARGS="--seed ${seed} \
                --num_worker ${NUM_WORKER} \
                --num_gpu ${NUM_GPU} \

                --epochs_budget ${EPOCHS} \
                --model ${MODEL} \
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
    CMD="sbatch run_cifar.sh ${SHELL_ARGS}"
    echo $CMD
    eval $CMD

done

