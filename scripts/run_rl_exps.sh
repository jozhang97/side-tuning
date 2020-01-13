#!/usr/bin/env bash
# ./scripts/run_rl_exps.sh small_settings5 sidetune curvature

MODELS='sidetune finetune feat scratch'
SETTING='small_settings5 corl_settings'
#MODELSIZES='std 2fcn5s'
#N_GPUS=4
#EXTRAS='radam dreg'

run() {
    SETTING=$1
    MODEL=$2
    TASK=$3
    if [ "$SETTING" != "small_settings5" ]; then
        echo 'BAD SETTING'
        exit
    fi
    if [ "$MODEL" == "feat" ]; then
        MODELTXT="taskonomy_features \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder_student.dat---,model=---FCN5---)'"
    elif [ "$MODEL" == "sidetune" ]; then
        MODELTXT="taskonomy_features sidetune \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.side_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.env.transform_fn_post_aggregation_kwargs.names_to_transforms.taskonomy='taskonomy_features_transform(---/mnt/models/${TASK}_encoder_student.dat---,model=---FCN5---)'"
    elif [ "$MODEL" == "finetune" ]; then
        MODELTXT="finetune rlgsn_base_learned \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.base_weights_path=/mnt/models/${TASK}_encoder_student.dat \
                  cfg.learner.perception_network_kwargs.extra_kwargs.sidetune_kwargs.side_weights_path=/mnt/models/${TASK}_encoder_student.dat"
    elif [ "$MODEL" == "scratch" ]; then
        MODELTXT="finetune rlgsn_base_learned"
        TASK=''
    else
        echo 'BAD MODEL'
        exit
    fi

    CMD="python -m scripts.train_rl /mnt/logdir/rl/planning/iclr/${TASK}_${MODEL}_2fcn5s \
        run_training with cfg_habitat planning ${SETTING} \
        uuid=iclr_${TASK}_${MODEL} \
        ${MODELTXT} rlgsn_base_fcn5s rlgsn_side_fcn5s \
        cfg.env.env_specific_kwargs.gpu_devices=\[1\] cfg.training.gpu_devices=\[0\]"
    echo $CMD
    bash -c "$CMD"
}
export -f run

run ${1} ${2} ${3}