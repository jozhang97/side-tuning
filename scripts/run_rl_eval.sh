#!/usr/bin/env bash
# ./scripts/run_rl_eval.sh sidetune curvature
# ./scripts/run_rl_eval.sh sidetune curvature 3700

MODELS='sidetune finetune feat scratch'

run() {
    MODEL=$1
    TASK=$2
    CKPTNUM=${3:-None}
    if [ "$MODEL" == "feat" ]; then
        MODELTXT="taskonomy_features \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.encoder_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder_student.dat---,model=---FCN5Skip---)'"
    elif [ "$MODEL" == "sidetune" ]; then
        MODELTXT="taskonomy_features sidetune \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.encoder_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.sidetuner_network_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder_student.dat---,model=---FCN5Skip---)'"
    elif [ "$MODEL" == "finetune" ]; then
        MODELTXT="finetune rlgsn_encoder_learned \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.encoder_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.sidetuner_network_weights_path=/mnt/models/${TASK}_encoder_student.dat"
    elif [ "$MODEL" == "scratch" ]; then
        MODELTXT="finetune rlgsn_encoder_learned"
        TASK=''
    else
        echo 'BAD MODEL'
        exit
    fi

    CMD="python -m scripts.train_rl /mnt/logdir/rl/planning/iclr_eval/${TASK}_${MODEL}_2fcn5s \
        run_training with cfg_habitat planning cfg_test \
        uuid=iclr_${TASK}_${MODEL}_eval \
        rlgsn_encoder_fcn5s rlgsn_side_fcn5s ${MODELTXT} \
        cfg.env.env_specific_kwargs.gpu_devices=\[1\] cfg.training.gpu_devices=\[0\] \
        cfg.saving.checkpoint=/mnt/logdir/rl/planning/iclr/${TASK}_${MODEL}_2fcn5s cfg.training.resumable=True \
        cfg.saving.checkpoint_num=${CKPTNUM}"
        # cfg.saving.checkpoint_configs=False
    echo $CMD
    bash -c "$CMD"
}
export -f run

run ${1} ${2} ${3}