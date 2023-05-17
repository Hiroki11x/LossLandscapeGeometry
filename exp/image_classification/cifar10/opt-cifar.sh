# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='opt-cifar'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="4"
NUM_GPU="2"

# ======== Configuration ========
MODEL="resnet18_1_cifar"
DATASET="cifar10"
DATA_ROOT="../data/cifar10"

# ======== Determined Hyperparam ========
NUM_CLASS="10"
MOMENTUM="0.9"
BATCH_SIZE="256"
WEIGHT_DECAY="1e-6"

# ======== Hyperparam for Sweep ========
OPT_LIST=('vanilla_sgd' 'momentum_sgd' 'adam')
LR_LIST=(0.0001 0.0005 0.001)
EPOCH_LIST=(100 190 280)


# ======== Hyper Parameter Search Loop  ========
for opt in "#{OPT_LIST[@]}" ; do
    for lr in "${LR_LIST[@]}" ; do
        for epoch in "${EPOCH_LIST[@]}" ; do

            SHELL_ARGS="--seed ${SEED} \
                        --num_worker ${NUM_WORKER} \
                        --num_gpu ${NUM_GPU} \

                        --epochs_budget ${epoch} \
                        --model ${MODEL} \
                        --dataset ${DATASET} \
                        --data_root ${DATA_ROOT} \
                        --num_classes ${NUM_CLASS} \

                        --batch_size ${BATCH_SIZE} \
                        --opt ${opt} \
                        --momentum ${MOMENTUM} \
                        --lr ${lr} \
                        --weight_decay ${WEIGHT_DECAY} \
                        
                        --wandb_exp_id ${count} \
                        --wandb_entity ${WANDB_ENTITY} \
                        --wandb_project_name ${WANDB_PROJECT_NAME} \
                        "
            CMD="sbatch run_cifar.sh ${SHELL_ARGS}"
            echo $CMD
            eval $CMD

        done
    done
done