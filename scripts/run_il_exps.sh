#!/usr/bin/env bash
# CANNOT RUN SCRIPT WITH SCRATCH AND SPECIAL ARCHITECTURES

run() {
    SIZE=$1
    MODEL=$2
    TASK=$3
    SPECIAL=$4
    echo "Imitation learning experiment:" $SIZE $MODEL $TASK $SPECIAL

    if [ "$MODEL" == "feat" ]; then
        TRAIN_EXTRA="il_sidetune ilgsn_no_side \
            cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder.dat"
        TEST_EXTRA="taskonomy_features \
            cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder.dat---,normalize_outputs=False)'"
    elif [ "$MODEL" == "sidetune" ]; then
        TRAIN_EXTRA="il_sidetune \
            cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.side_weights_path=/mnt/models/${TASK}_encoder_student.dat \
            cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder.dat"
        TEST_EXTRA="transform_rgb256 taskonomy_features \
            cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder.dat---,normalize_outputs=False)'"
    elif [ "$MODEL" == "finetune" ]; then
        TRAIN_EXTRA="ilgsn_base_resnet50 ilgsn_base_learned cfg.training.dataloader_fn_kwargs.batch_size=16 \
            cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder.dat \
            cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.side_weights_path=/mnt/models/${TASK}_encoder_student.dat"
        TEST_EXTRA="transform_rgb256"
    elif [ "$MODEL" == "scratch" ]; then
        TRAIN_EXTRA="ilgsn_base_resnet50 ilgsn_base_learned cfg.training.dataloader_fn_kwargs.batch_size=16"
        TEST_EXTRA="transform_rgb256"
    else
        echo "BAD MODEL"
        exit
    fi

    if   [ "$SPECIAL" == "ilgsn_base_fcn5" ]; then
        TRAIN_SPECIAL="cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder_student.dat"
        TEST_SPECIAL="cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder_student.dat---,normalize_outputs=False,model=---FCN5Skip---)'"
    elif [ "$SPECIAL" == "ilgsn_side_resnet50" ]; then
        TRAIN_SPECIAL="cfg.learner.model_kwargs.base_kwargs.perception_unit_kwargs.extra_kwargs.sidetune_kwargs.side_weights_path=/mnt/models/${TASK}_encoder.dat \
                       cfg.training.dataloader_fn_kwargs.batch_size=16"
    fi

    TRAIN_CMD="python -m scripts.train_transfer \
        /mnt/logdir/imitation_learning/train/${SIZE}_${TASK}_${MODEL}_${SPECIAL} \
        train with imitation_learning expert_data ${TRAIN_EXTRA} il_${SIZE} \
        cfg.training.seed=42 \
        cfg.training.resume_from_checkpoint_path=/mnt/logdir/imitation_learning/train/${SIZE}_${TASK}_${MODEL}_${SPECIAL}/checkpoints/ckpt-latest.dat \
        cfg.training.resume_training=True cfg.training.resume_w_no_add_epochs=True \
        il_source_rmt ${SPECIAL} ${TRAIN_SPECIAL}"

    echo $TRAIN_CMD
    bash -c "$TRAIN_CMD"

    TEST_CMD="python -m scripts.train_rl \
        /mnt/logdir/imitation_learning/test/${SIZE}_${TASK}_${MODEL}_${SPECIAL} \
        run_training \
        with cfg_habitat planning cfg_test \
        uuid=iclr_imitation_learning_${SIZE}_${TASK}_${MODEL}_${SPECIAL} \
        cfg.learner.algo='imitation_learning' \
        cfg.saving.checkpoint=/mnt/logdir/imitation_learning/train/${SIZE}_${TASK}_${MODEL}_${SPECIAL} \
        cfg.saving.checkpoint_configs=False cfg.training.resumable=True \
        override.saving.log_interval=50 override.saving.vis_interval=50 \
        override.env.num_processes=8 override.env.num_val_processes=8 \
        ${TEST_EXTRA} ${SPECIAL} ${TEST_SPECIAL}"

    echo $TEST_CMD
    bash -c "$TEST_CMD"
}
export -f run

SIZES='debug tiny small medium large largeplus'
MODELS='feat sidetune finetune scratch'
TASKS='curvature denoising'
SPECIALS='ilgsn_base_fcn5 ilgsn_side_resnet50'

if [ "$3" == "" ]; then
    echo 'no cmd input, run all'
    for SIZE in $SIZES; do
        export SIZE
        for MODEL in $MODELS; do
            export MODEL
            for TASK in $TASKS; do
                export TASK
                bash -c 'run ${SIZE} ${MODEL} ${TASK}'
            done
        done
    done
elif [ "$3" == "alltasks" ]; then
    for TASK in $TASKS; do
        export TASK
        run ${1} ${2} ${TASK}
    done
elif [ "$3" == "allmodels" ]; then
    for MODEL in $MODELS; do
        export MODEL
        run ${1} ${MODEL} ${2}
    done
elif [ "$3" == "allsizes" ]; then
    for SIZE in $SIZES; do
        export SIZE
        run ${SIZE} ${1} ${2}
    done
else
    run ${1} ${2} ${3} ${4}
fi
