# ======== Wandb Configuration ========
WANDB_ENTITY='YOUR_ENTITY'
WANDB_PROJECT_NAME='batchsize-wikitext2'

# ======== Environmental Setting ========
SEED="1234"
NUM_WORKER="4"
NUM_GPU="2"

# ======== Configuration ========
DATASET="wikitext-2"
DATA_ROOT="./data/wikitext-2"

# ======== Determined Hyperparam ========
OPT="adam"
EPOCHS="20"
WEIGHT_DECAY=0.00001
BASE_LR=0.0001


# ======== Hyperparam for Sweep ========
MODEL_LIST=('Transformer')
BATCH_SIZE_LIST=(32 64 128 256)


# ======== Hyper Parameter Search Loop  ========
for model in "${MODEL_LIST[@]}" ; do
    for batch_size in "${BATCH_SIZE_LIST[@]}" ; do

        DIV_B=$(python -c "print($batch_size/${BATCH_SIZE_LIST[0]})")
        MULTIPLIER=$(echo "scale=20;sqrt($DIV_B)" | bc)
        LR=$(python -c "print($BASE_LR*$MULTIPLIER)")

        SHELL_ARGS="--seed ${SEED} \
                    --num_worker ${NUM_WORKER} \
                    --num_gpu ${NUM_GPU} \

                    --epochs_budget ${EPOCHS} \
                    --model ${model} \
                    --dataset ${DATASET} \
                    --data_root ${DATA_ROOT} \

                    --batch_size ${batch_size} \
                    --opt ${OPT} \
                    --lr ${LR} \
                    --weight_decay ${WEIGHT_DECAY} \

                    --wandb_exp_id ${count} \
                    --wandb_entity ${WANDB_ENTITY} \
                    --wandb_project_name ${WANDB_PROJECT_NAME} \
                    "
        CMD="sbatch run_wikitext2.sh ${SHELL_ARGS}"
        echo $CMD
        eval $CMD

    done
done
