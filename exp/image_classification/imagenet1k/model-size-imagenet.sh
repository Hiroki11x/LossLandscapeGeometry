# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='model-size-imagenet'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="8"
NUM_GPU="4"

# ======== Configuration ========
DATASET="imagenet"
DATA_ROOT="None"

# ======== Determined Hyperparam ========
EPOCHS="90" 
NUM_CLASS="1000"
MOMENTUM="0.9"
BATCH_SIZE="256"
OPT="momentum_sgd"

WEIGHT_DECAY="1e-6"
LR_SCHEDULE="True"
WARMUP_EPOCHS="3"

# ======== Hyperparam for Sweep ========
MODEL_LIST=("resnet18_1" "resnet18_2" "resnet50_1" "resnet50_2" "resnet152_1" "resnet152_05" "resnet152_1")
LR_LIST=(0.1 0.075)


# ======== Hyper Parameter Search Loop  ========
for model in "${MODEL_LIST[@]}" ; do
    for lr in "${LR_LIST[@]}" ; do

        SHELL_ARGS="--seed ${SEED} \
                    --num_worker ${NUM_WORKER} \
                    --num_gpu ${NUM_GPU} \

                    --epochs_budget ${EPOCHS} \
                    --model ${model} \
                    --dataset ${DATASET} \
                    --data_root ${DATA_ROOT} \
                    --num_classes ${NUM_CLASS} \

                    --batch_size ${BATCH_SIZE} \
                    --opt ${OPT} \
                    --momentum ${MOMENTUM} \
                    --lr ${lr} \
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
done
