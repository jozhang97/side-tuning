#!/usr/bin/env bash
# ./scripts/run_lifelong_cifar.sh sidetune model_boosted_cifar

MODELS='sidetune_reverse finetune independent features'
EXTRAS='ewc bsp_norecurse_cifar init_xavier init_lowenergy_cifar'

MODEL=$1
EXTRA=${2:-}
HP=${3:-}

if [ "$MODEL" == "all" ]; then
    for MODEL in $MODELS; do
        export MODEL
        python -m feature_selector.lifelong \
            /mnt/logdir/lifelong/icifar/${MODEL} \
            train with cifar_hp icifar_data \
                model_lifelong_${MODEL}_cifar
    done

    HP='4'
    for EXTRA in $EXTRAS; do
        export EXTRA
        python -m feature_selector.lifelong \
            /mnt/logdir/lifelong/icifar/finetune_${EXTRA}${HP} \
            train with cifar_hp icifar_data \
                model_lifelong_finetune_cifar ${EXTRA}\
            cfg.training.regularizer_kwargs.coef=1e${HP}
    done
else
    if [ "$HP" == "" ]; then
        HP_TXT=""
    else
        HP_TXT="cfg.training.regularizer_kwargs.coef=1e${HP}"
    fi
    CMD="python -m scripts.train_lifelong \
        /mnt/logdir/lifelong/icifar/${MODEL}_${EXTRA}${HP} \
        train with cifar_hp icifar_data \
            model_lifelong_${MODEL}_cifar ${EXTRA}\
        ${HP_TXT}"

    echo $CMD
    bash -c "$CMD"
fi

