# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='batchsize-cifar'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="4"
NUM_GPU="2"

# ======== Configuration ========
MODEL="resnet18_1_cifar"
DATASET="cifar10"
DATA_ROOT="../data/cifar10"

# ======== Determined Hyperparam ========
OPT="momentum_sgd"
EPOCHS="190" # for debug
NUM_CLASS="10"
WEIGHT_DECAY="1e-6"

# ======== Hyperparam for Sweep ========
MOMENTUM_LIST=(0.9)
BATCH_SIZE_LIST=(64 128 256 512)
BASE_LR=0.005

count=0
lower_bound=0
upper_bound=100


# ======== Hyper Parameter Search Loop  ========
for momentum in "${MOMENTUM_LIST[@]}" ; do
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
                    --momentum ${momentum} \
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
done


