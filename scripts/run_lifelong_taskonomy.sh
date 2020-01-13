#!/usr/bin/env bash
# MODEL='finetune' && SIZE='std' && SPLIT='12' && EXTRA='bsp' && HP=''
# ./scripts/run_lifelong_taskonomy.sh finetune std ewc 2

run() {
    MODEL=$1
    SIZE=$2
    SPLIT=$3
    EXTRA=$4
    HP=$5
    if [ "$MODEL" == "independent" ]; then
        BATCHSIZE=16
    else
        BATCHSIZE=32
    fi
    if [ "$HP" == "" ]; then
        HPTXT=""
    else
        HPTXT="cfg.training.regularizer_kwargs.coef=1e${HP}"
    fi
    CMD="python -m scripts.train_lifelong \
        /mnt/logdir/lifelong/taskonomy/arxiv/${SPLIT}_${MODEL}_${SIZE}_${EXTRA}${HP} train with \
            taskonomy_base_data taskonomy_${SPLIT}_data ${EXTRA} \
            model_lifelong_${MODEL}_${SIZE}_taskonomy ${EXTRA} \
            model_learned_decoder \
        ${HPTXT} cfg.training.dataloader_fn_kwargs.batch_size=${BATCHSIZE}"
    echo $CMD
    bash -c "$CMD"
}
export -f run

#SIZES="std resnet50 fcn5s"
#SPLITS="3 12 shuffle12"
#EXTRAS="bsp ewc init_xavier init_lowenergy debug pnn_v4 "
MODELS="finetune sidetune independent"

if [ "$1" == "all" ]; then
    for MODEL in $MODELS; do
        export MODEL
        bash -c "run ${MODEL} std 12"
    done

    bash -c "run finetune std 12 ewc 2"
    bash -c "run finetune std 12 bsp"
else
    run ${1} ${2} ${3} ${4} ${5}
fi

#OLD COMMAND
#MODEL='finetune' && SIZE='std' && SPLIT='big' && EXTRA='bsp' && HP='' && \
#python -m tlkit.lifelong \
#    /mnt/logdir/lifelong/taskonomy/${SPLIT}_${MODEL}_${SIZE}_${EXTRA}${HP} train with \
#        taskonomy_${SPLIT}_data ${EXTRA} \
#        model_lifelong_${MODEL}_${SIZE}_taskonomy \
#        model_learned_decoder \
#    cfg.training.regularizer_kwargs.coef=1e${HP}
