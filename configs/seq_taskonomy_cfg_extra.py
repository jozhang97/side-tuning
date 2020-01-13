##################
# Feedback
##################
@ex.named_config
def full_feedback1():
    cfg = {
        'learner': {
            'use_feedback': True,
            'feedback_kwargs': {
                'num_feedback_iter': 5,
            },
            'model_kwargs': {
                'side_class': 'FCN5MidFeedback',
                'merge_method': 'side_only',
                'side_kwargs': {
                    'kernel_size': 1,
                } } } }

@ex.named_config
def with_feedback1():
    cfg = {
        'learner': {
            'model_kwargs': {
                'merge_method': 'side_only',
                'side_class': 'FCN5LateFeedback',
                'side_kwargs': {
                    'kernel_size': 1,
            } } } }

@ex.named_config
def with_feedback3():
    cfg = {
        'learner': {
            'model_kwargs': {
                'merge_method': 'side_only',
                'side_class': 'FCN5LateFeedback',
                'side_kwargs': {
                    'kernel_size': 3,
            } } } }


##################
# Single task
##################
@ex.named_config
def taskonomy_single_data():
    cfg = {
        'training': {
            'dataloader_fn': 'taskonomy_dataset.get_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/', # for docker
                'train_folders': 'debug',
                'val_folders':   'debug',
                'test_folders':  'debug',
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': True,
            }
        }
    }

@ex.named_config
def model_taskonomy():
    cfg = { 'learner': {
        'model': 'TaskonomyNetwork',
        'model_kwargs': {
            'out_channels': 3,
            'eval_only': False,
        } } }

@ex.named_config
def model_taskonomy_class():
    cfg = {
        'learner': {
            'model': 'TaskonomyNetwork',
            'model_kwargs': {
                'out_channels': 1000,
                'trainable': True,
                'is_decoder_mlp': True
            }
        },
        'training': {
            'sources': ['rgb'],
            'targets': ['class_object'],
        }
    }


##################
# Additional Initializations
##################
@ex.named_config
def init_lowenergy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'FCN5',
            'side_weights_path': '/mnt/models/curvature_encoder_student_lowenergy.dat',
        } },
    }

##################
# Additional Architectures
##################
@ex.named_config
def fcn8():
    # for a smaller independent to match # of parameters of sidetuning
    # (NUM_BASE + NUM_SIDES)/12 = (23655504 + 11070528)/12=2893836
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'side_class': 'GenericSidetuneNetwork',
                'side_kwargs': {
                    'base_class': None,
                    'base_weights_path': None,

                    'side_class': 'FCN8',
                    'side_weights_path': '/mnt/models/curvature_base_fcn8.dat',
                },
            } },
        'training': {
            'dataloader_fn_kwargs': {
                'batch_size': 32,
    } } }

@ex.named_config
def model_frozen_decoder():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'transfer_class': 'PreTransferedDecoder',
            'transfer_kwargs': {
                'transfer_class': 'TransferConv3',
                'transfer_weights_path': None,
                'transfer_kwargs': {'n_channels': 8, 'residual': True},

                'decoder_class': 'TaskonomyDecoder',
                'decoder_weights_path': None,  #  When using, should add diff weights_path
                'decoder_kwargs': {'eval_only': True, 'train': False},
            },
        } } }

@ex.named_config
def model_lifelong_independent_resnet_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'TaskonomyEncoder',
            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
            'side_weights_path': '/mnt/models/curvature_encoder.dat',

            'normalize_pre_transfer': True,
        } } }

@ex.named_config
def model_lifelong_independent_fcn5s_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'FCN5',
            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
            'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
            'normalize_pre_transfer': True,
        } } }

@ex.named_config
def ensemble_side():
    cfg = {
        'learner': {
            'model_kwargs': {
                'side_class': 'EnsembleNet',
                'side_weights_path': None,
                'side_kwargs': {
                    'n_models': 3,
                    'model_class': 'FCN5',
                    'model_weights_path': '/mnt/models/curvature_encoder_student.dat',
                },
            }
        },
    }


@ex.named_config
def model_lifelong_sidetune_double_resnet_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'TaskonomyEncoder',
            'base_weights_path': '/mnt/models/curvature_encoder.dat',  # user needs to input
            'base_kwargs': {'eval_only': True, 'train': False, 'normalize_outputs': False},
            'use_baked_encoding': True,

            'side_class': 'TaskonomyEncoder',
            'side_weights_path': '/mnt/models/curvature_encoder.dat',
            'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},

            'normalize_pre_transfer': True,
        } },
        'training': {
            'sources': [['rgb', 'curvature_encoding']] * N_TASKONOMY_TASKS,
        }
    }

@ex.named_config
def model_lifelong_sidetune_double_fcn5s_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'FCN5',
            'base_weights_path': '/mnt/models/curvature_encoder_student.dat',  # user needs to input
            'base_kwargs': {'eval_only': True, 'train': False, 'normalize_outputs': False},
            'use_baked_encoding': False,

            'side_class': 'FCN5',
            'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
            'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},

            'normalize_pre_transfer': True,
        } } }


@ex.named_config
def model_lifelong_sidetune_double_open_resnet_taskonomy():
    # a variant of finetune with sidetune architecture
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'TaskonomyEncoder',
            'base_weights_path': '/mnt/models/curvature_encoder.dat',  # user needs to input
            'base_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
            'use_baked_encoding': False,

            'side_class': 'TaskonomyEncoder',
            'side_weights_path': '/mnt/models/curvature_encoder.dat',
            'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},

            'normalize_pre_transfer': True,
        } },
    }

@ex.named_config
def model_lifelong_sidetune_double_open_fcn5s_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'FCN5',
            'base_weights_path': '/mnt/models/curvature_encoder_student.dat',  # user needs to input
            'base_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
            'use_baked_encoding': False,

            'side_class': 'FCN5',
            'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
            'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},

            'normalize_pre_transfer': True,
        } } }

@ex.named_config
def model_lifelong_finetune_double_fcn5s_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'n_channels_in': 3,
                'n_channels_out': 8,

                'base_class': 'FCN5',
                'base_weights_path': '/mnt/models/curvature_encoder_student.dat',  # user needs to input
                'base_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
                'use_baked_encoding': False,
                'normalize_pre_transfer': False,

                'side_class': 'FCN5',
                'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
                'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
            },
            'use_baked_encoding': False,
            'normalize_pre_transfer': True,
        } } }

@ex.named_config
def model_lifelong_finetune_double_resnet_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'n_channels_in': 3,
                'n_channels_out': 8,

                'base_class': 'TaskonomyEncoder',
                'base_weights_path': '/mnt/models/curvature_encoder.dat',  # user needs to input
                'base_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
                'use_baked_encoding': False,
                'normalize_pre_transfer': False,

                'side_class': 'TaskonomyEncoder',
                'side_weights_path': '/mnt/models/curvature_encoder.dat',
                'side_kwargs': {'eval_only': False, 'train': True, 'normalize_outputs': False},
            },
            'use_baked_encoding': False,
            'normalize_pre_transfer': True,
        } } }


##################
# Baselines
##################
@ex.named_config
def pnn_v1():
    # naively add lateral connections from Base Network to Side Networks, no new params
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5ProgressiveNoNewParam',
            'pnn': True,
        } },
    }

@ex.named_config
def pnn_v2a():
    # use linear adapter to connect activations from Base Network to activations in Side Network
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5Progressive',
            'side_kwargs': {
                'dense': False,
                'k': 3,
                'adapter': 'linear',
            },
            'pnn': True,
        } },
    }

@ex.named_config
def pnn_v2b():
    # PNN v2 with 1x1 kernel
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5Progressive',
            'side_kwargs': {
                'dense': False,
                'k': 1,
                'adapter': 'linear',
            },
            'pnn': True,
        } },
    }

@ex.named_config
def pnn_v2c():
    # PNN v2 with 1x1 kernel and MLP
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5Progressive',
            'side_kwargs': {
                'dense': False,
                'k': 1,
                'adapter': 'mlp',
            },
            'pnn': True,
        } },
    }

@ex.named_config
def pnn_v2d():
    # PNN v2 with 1x1 kernel and MLP and extra adapter (to compare vs rigid)
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5Progressive',
            'side_kwargs': {
                'dense': False,
                'k': 1,
                'adapter': 'mlp',
                'extra_adapter': True,
            },
            'pnn': True,
        } },
    }

@ex.named_config
def pnn_v3():
    # use linear adapter to connect activations from Base Network AND previous Side Networks
    # to activations in Side Network
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5Progressive',
            'side_kwargs': {
                'dense': True,
                'k': 3,
                'adapter': 'linear',
            },
            'pnn': True,
            'dense': True,
        } },
    }

@ex.named_config
def pnn_v4():
    # Follows the paper as closely as possible
    # Lateral connections from base and other side networks
    # MLP Adapters
    # Merge hidden states and apply nonlinearity after merging
    # Uses 1x1 conv when possible
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5ProgressiveH',
            'pnn': True,
            'dense': True,
        } },
    }

@ex.named_config
def pnn_early_fusion():
    # Note: This is not pnn the same way sidetuning is not pnn
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5ProgressiveNoNewParam',
            'side_kwargs': { 'early_fusion': True },
            'pnn': True,
            'merge_method': 'merge_operators.SideOnly',
        } },
    }


@ex.named_config
def pnn_rigidity():
    # What would happen if task N was trained as task 1 (test for rigidity)
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'use_baked_encoding': False,
            'base_class': 'TaskonomyEncoderWithCache',
            'side_class': 'FCN5ProgressiveH',
            'pnn': True,
            'dense': False,
        } },
    }

@ex.named_config
def bsp_simple_base():
    # use binary superposition from https://arxiv.org/pdf/1902.05522
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_kwargs': {'bsp': True},
            'side_kwargs': {'bsp': True}
        } } }

@ex.named_config
def bsp():
    # use binary superposition from https://arxiv.org/pdf/1902.05522
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'base_kwargs': {'bsp': True, 'period': 12},
                'side_kwargs': {'bsp': True, 'period': 12},
            },
        } } }

@ex.named_config
def bsp_small():
    # use binary superposition from https://arxiv.org/pdf/1902.05522
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'base_kwargs': {'bsp': True, 'period': 3},
                'side_kwargs': {'bsp': True, 'period': 3},
            },
        } } }

@ex.named_config
def untrack_bn_simple_base():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_kwargs': {'track_running_stats': False},
        } } }

@ex.named_config
def untrack_bn():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'base_kwargs': {'track_running_stats': False},
            },
        } } }

@ex.named_config
def ewc():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'EWC',
        'regularizer_kwargs': {
            'coef': 1e5,
            'n_samples_fisher': 5000,
            'avg_tasks': True,
        },
        'loss_list': ['total', 'weight_tying'],
    }

@ex.named_config
def ewc_n_terms():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'EWC',
        'regularizer_kwargs': {
            'coef': 1e4,
            'n_samples_fisher': 5000,
            'avg_tasks': False,
        },
        'loss_list': ['total', 'weight_tying'],
    }

##################
# Losses
##################
@ex.named_config
def loss_perceptual():
    # only for target encodings
    cfg = {}
    cfg['training'] = {
        'loss_fn': 'perceptual_l1',
        'targets': ['normal_encoding'],
        'loss_kwargs' : {
            'decoder_path': '/mnt/models/normal_decoder.dat',
            'bake_decodings': True,  # for now, must bake decodings, since we do not have target encoding data
        },
        # if we have decodings, we do not need annotator
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'annotator_class': 'TaskonomyEncoder',
        'annotator_weights_path': '/mnt/models/normal_encoder.dat',
        'annotator_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': False},
    }

@ex.named_config
def loss_perceptual_l2():
    # only for target encodings
    cfg = {}
    cfg['training'] = {
        'loss_fn': 'perceptual_l2',
        'targets': ['normal_encoding'],
        'loss_kwargs' : {
            'decoder_path': '/mnt/models/normal_decoder.dat',
            'bake_decodings': True,  # for now, must bake decodings, since we do not have target encoding data
        },
        # if we have decodings, we do not need annotator
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'annotator_class': 'TaskonomyEncoder',
        'annotator_weights_path': '/mnt/models/normal_encoder.dat',
        'annotator_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': False},
    }

@ex.named_config
def loss_perceptual_cross_entropy():
    # only for target encodings
    cfg = {}
    cfg['training'] = {
        'loss_fn': 'perceptual_cross_entropy',
        'targets': ['class_object_encoding'],
        'loss_kwargs' : {
            'decoder_path': '/mnt/models/class_scene_decoder.dat',
            'bake_decodings': True,
        },
        # if we have decodings, we do not need annotator
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'annotator_class': 'TaskonomyEncoder',
        'annotator_weights_path': '/mnt/models/class_scene_encoder.dat',
        'annotator_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': False},
    }

##################
# Regularizers
# (wraps around loss function)
##################
@ex.named_config
def transfer_reg():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'transfer_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
        },
        'loss_list': ['l1', 'total', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }

@ex.named_config
def perceptual_reg():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'perceptual_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'decoder_path': '/mnt/models/curvature_decoder.dat',  # user needs to input
        },
        'loss_list': ['l1', 'total', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }


@ex.named_config
def perceptual_reg_no_transfer():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'perceptual_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'decoder_path': '/mnt/models/curvature_decoder.dat',  # user needs to input
            'use_transfer': False,
        },
        'loss_list': ['l1', 'total', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }

##################
# Learning Rates
##################
@ex.named_config
def taskonomy_hp():
    uuid = 'no_uuid'
    cfg = {}
    cfg['learner'] = {
        'lr': 1e-4,                  # Learning rate for algorithm
        'optimizer_kwargs' : {
            'weight_decay': 2e-6
        },
    }

@ex.named_config
def scheduler_reduce_on_plateau():
    cfg = { 'learner': {
        'lr_scheduler_method': 'lr_scheduler.ReduceLROnPlateau',
        'lr_scheduler_method_kwargs': {
            'factor': 0.1,
            'patience': 5
        } } }



@ex.named_config
def scheduler_step_lr():
    cfg = { 'learner': {
        'lr_scheduler_method': 'lr_scheduler.StepLR',
        'lr_scheduler_method_kwargs': {
            'lr_decay_epochs': 30,       # number of epochs before the LR drops by factor of 10
            'gamma': 0.1
        } } }
