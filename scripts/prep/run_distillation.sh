#!/usr/bin/env bash
# ./scripts/run_distillation.sh zero tiny curvature fcn8

run() {
    ALGO=$1
    SIZE=$2
    TASK=$3
    ARCH=$4
    echo "Distillation experiment:" $ALGO $SIZE $TASK $ARCH

    if [ "$TASK" == "curvature" ]; then
        EXTRA="loss_perceptual_l2"
    elif [ "$TASK" == "denoising" ]; then
        EXTRA="loss_perceptual"
    elif [ "$TASK" == "class_object" ]; then
        EXTRA="loss_perceptual_cross_entropy"
    else
        echo "BAD TASK"
        exit
    fi

    CMD="python -m tlkit.transfer \
            /mnt/logdir/distillation/${ALGO}/${TASK}_${SIZE}_${ARCH} train with \
            taskonomy_hp model_${ARCH} scheduler_reduce_on_plateau ${EXTRA} \
            uuid=distil \
            cfg.training.data_dir=/mnt/data \
            cfg.training.split_to_use=splits.taskonomy_no_midlevel\[\'${SIZE}\'\] \
            cfg.training.sources=\[\'rgb\'\] \
            cfg.training.targets=\[\'${TASK}_encoding\'\] \
            cfg.training.loss_kwargs.decoder_path='/mnt/models/${TASK}_decoder.dat' \
            cfg.training.annotator_weights_path='/mnt/models/${TASK}_encoder.dat' \
            cfg.training.seed=42 \
            cfg.training.num_epochs=10 \
            cfg.training.loss_kwargs.bake_decodings=False \
            cfg.training.suppress_target_and_use_annotator=True \
            cfg.training.resume_from_checkpoint_path=/mnt/logdir/distillation/${ALGO}/${TASK}_${SIZE}/checkpoints/ckpt-latest.dat \
            cfg.training.resume_training=True \
            cfg.training.algo=${ALGO}"

    echo $CMD
    bash -c "$CMD"
}
export -f run

ALGOS='student zero'
SIZES='debug tiny small medium large full fullplus'
TASKS='curvature denoising class_object'
ARCHS='fcn5_skip fcn8'

run ${1} ${2} ${3} ${4}
