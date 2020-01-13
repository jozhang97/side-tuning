#!/usr/bin/env bash
# ./scripts/run_lifelong_taskonomy.sh curvature normal few100 sidetune

# This script only supports the following settings (others can be set up):
SOURCE_TASK='normal curvature class_object'
TARGET_TASK='normal curvature class_object'
DATA_SIZE='few100 fullplus'
MODEL='sidetune'

run() {
    SOURCE=$1
    TARGET=$2
    DATA_SIZE=$3
    MODEL=$4

    if [ "$TARGET" == "normal" ]; then
        TARGET_TXT="cfg.training.loss_fn='weighted_l1_loss'"
    elif [ "$TARGET" == "curvature" ]; then
        TARGET_TXT="cfg.training.loss_fn='weighted_l2_loss'"
    elif [ "$TARGET" == "class_object" ]; then
        TARGET_TXT="cfg.training.loss_fn='softmax_cross_entropy'"
    else
        echo 'Not set up for current target task'
        exit
    fi

    CMD="
        python -m scripts.train_transfer \
            /mnt/logdir/vision_transfers/arxiv_code/${SOURCE}_to_${TARGET}_${DATA_SIZE}_${MODEL} \
            train with \
                gsn_transfer_residual_prenorm \
                taskonomy_hp \
                cfg.training.data_dir=/mnt/data \
                cfg.learner.max_grad_norm=1 \
                model_sidetune_encoding gsn_side_resnet50 \
                cfg.learner.model_kwargs.base_weights_path='/mnt/models/${SOURCE}_encoder.dat' \
                cfg.learner.model_kwargs.side_weights_path='/mnt/models/${SOURCE}_encoder.dat' \
                cfg.learner.model_kwargs.use_baked_encoding=False \
                cfg.training.sources=\[\'rgb\'\] \
                cfg.training.targets=\[\'${TARGET}\'\] \
                ${TARGET_TXT} \
                data_size_${DATA_SIZE}
        "
    echo $CMD
    bash -c "$CMD"
}
export -f run


run ${1} ${2} ${3} ${4}

