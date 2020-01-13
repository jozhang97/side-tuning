## SACRED THING: Make sure to never set kwargs (or any field that can possibly be a dictionary) to None
## SACRED THING: Make sure not to put any comments in the line after `def config():`
import tlkit.data.splits as splits


@ex.config
def cfg_base():
    uuid = 'no_uuid'
    cfg = {}
    cfg['learner'] = {
        'model': 'atari_residual',
        'model_kwargs' : {},
        'eps': 1e-5,                 # Small epsilon to prevent divide-by-zero
        'lr': 1e-3,                  # Learning rate for algorithm
        'optimizer_class': 'optim.Adam',
        'optimizer_kwargs' : {
            'weight_decay': 0.0001
        },
        'lr_scheduler_method': None,
        'lr_scheduler_method_kwargs': {},
        'max_grad_norm': 1.0,        # Clip grads
        'scheduler': 'plateau',     # Automatically reduce LR once loss plateaus (plateau, step)
    }
    cfg['training'] = {
        'algo': 'student',  # 'student zero'
        'post_aggregation_transform_fn': None,
        'post_aggregation_transform_fn_kwargs': {},
        'batch_size': 32,
        'batch_size_val': 32,
        'cuda': True,
        'loss_fn': 'L1',
        'loss_kwargs': {},
        'loss_list': ['total'],      # ones to report, 'standard' and 'final' are already represented
        'regularizer_fn': None,
        'regularizer_kwargs': {},
        'epochs': 100,
        'num_epochs': 100,
        'resume_from_checkpoint_path': None,
        'resume_training': False,
        'resume_w_no_add_epochs': False,
        'seed': random.randint(0,1000),
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'sources': ['rgb'],
        'sources_as_dict': False,
        'targets': ['normal_decoding'],
        'train': True,
        'test': False,
        'use_masks': True,
        'dataloader_fn': None,
        'dataloader_fn_kwargs': {},
        #------- Legacy dataloading (Taskonomy-only) --------
        'data_dir': '/mnt/hdd2/taskonomy_reps', # for docker
        'load_to_mem': False,  # if dataset small enough, can load activations to memory
        'num_workers': 8,
        'pin_memory': True,
        'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
        #------- END --------
    }
    cfg['saving'] = {
        'obliterate_logs': False,
        'log_dir': LOG_DIR,
        'log_interval': 0.25,  # num of epochs... only support < 1
        'ticks_per_epoch': 100,  # one epoch will go thru 100 numbers, if we want to log in between an epoch, will need this to be a larger number. If log_interval >=1, this could be set to 1
        'logging_type': 'tensorboard',
        'results_log_file': os.path.join(LOG_DIR, 'result_log.pkl'),
        'reward_log_file': os.path.join(LOG_DIR, 'rewards.pkl'),
        'save_interval': 1,
        'save_dir': 'checkpoints',
        'visdom_log_file': os.path.join(LOG_DIR, 'visdom_logs.json'),
        'visdom_server': 'localhost',
        'visdom_port': '8097',
        'in_background': False,
    }


####################################
# Datasets
####################################
@ex.named_config
def taskonomy_data():
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
def imagenet_data():
    cfg = {
        'training': {
            'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
            'dataloader_fn': 'imagenet_dataset.get_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/ILSVRC2012', # for docker
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': False,
            },
            'loss_fn': 'softmax_cross_entropy',
            'loss_kwargs': {},
            'use_masks': False,
        }
    }

@ex.named_config
def rotating_data():
    cfg = {
        'training': {
            'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
            'sources': ['rgb', 'rotation'],
            'post_aggregation_transform_fn': 'imagenet_dataset.RotateBatch()',
            'post_aggregation_transform_fn_kwargs': {},
        },
        'learner': {
            'model': 'DummyLifelongTaskonomyNetwork',
            'model_kwargs': {
                'out_channels': 1000,
                'trainable': True,
                'is_decoder_mlp': True,
            }
        }
    }

@ex.named_config
def icifar0_data():
    cfg = {
        'training': {
            # 'split_to_use': 'splits.taskonomy_no_midlevel["debug"]',
            # 'dataloader_fn': 'icifar_dataset.get_limited_dataloaders',get_cifar_dataloaders
            'dataloader_fn': 'icifar_dataset.get_cifar_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/', # for docker
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': True,
                'epochlength': 2000,
                'sources': ['rgb'],
                'targets': [['cifar0-9']],
                'masks':   None,
            },
            'use_masks': False,
            'targets': ['class_object'],
            # 'task_is_classification': [True] * len(chunked_classes),
        }
    }

####################################
# Dataset Sizes
####################################
@ex.named_config
def data_size_few100():
    cfg = {
        'training': {
            'split_to_use': "splits.taskonomy_no_midlevel['few100']",
            'num_epochs': 10000,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 250,
            'save_interval': 1000,
        }
    }

@ex.named_config
def data_size_fullplus():
    cfg = {
        'training': {
            'split_to_use': splits.taskonomy_no_midlevel['fullplus'],
            'num_epochs': 10,
        },
        'saving': {
            'ticks_per_epoch': 0.25,
            'log_interval': 1,
            'save_interval': 1,
        }
    }

####################################
# Models
####################################
@ex.named_config
def model_resnet_cifar():
    cfg = {
        'learner': {
            'model': 'FCN5SkipCifar',
            'model_kwargs': {
                'num_groups': 2,
                'use_residual': False,
                'normalize_outputs': False,
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

@ex.named_config
def model_fcn8():
    cfg = {'learner': {
        'model': 'FCN8',
        'model_kwargs': {
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_fcn5():
    cfg = {'learner': {
        'model': 'FCN5Residual',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': False,
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_fcn5_residual():
    cfg = {'learner': {
        'model': 'FCN5Residual',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': True,
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_fcn5_skip():
    cfg = {'learner': {
        'model': 'FCN5',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': False,
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_fcn5_skip_residual():
    cfg = {'learner': {
        'model': 'FCN5',
        'model_kwargs': {
            'num_groups': 2,
            'use_residual': True,
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_fcn3():
    cfg = { 'learner': {
        'model': 'FCN3',
        'model_kwargs': {
            'num_groups': 2,
            'normalize_outputs': False,
        } } }

@ex.named_config
def model_taskonomy_net():
    cfg = { 'learner': {
        'model': 'TaskonomyNetwork',
        'model_kwargs': {
            'out_channels': 3,
            'eval_only': False,
        } } }

@ex.named_config
def model_sidetune_encoding():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'eval_only': True, 'normalize_outputs': True},
            'use_baked_encoding': True,

            'side_class': 'FCN5',
            'side_kwargs': {'normalize_outputs': True},
            'side_weights_path': None,  # user can input for smart/student initialization

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': False},
        } } }
    del n_channels_out

@ex.named_config
def model_nosidetune_encoding():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'eval_only': True, 'normalize_outputs': True},
            'use_baked_encoding': True,

            'side_class': None,
            'side_kwargs': {},
            'side_weights_path': None,

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': False},
        } } }
    del n_channels_out

@ex.named_config
def model_pix_only_side():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': None,
            'base_weights_path': None,
            'base_kwargs': {},
            'use_baked_encoding': False,

            'side_class': 'FCN5',
            'side_kwargs': {'normalize_outputs': True},
            'side_weights_path': None,  # user can input for smart/student initialization

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': False},
        } } }
    del n_channels_out

@ex.named_config
def model_pix_only_encoder():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user can input for smart initialization
            'base_kwargs': {'eval_only': False, 'normalize_outputs': True},
            'use_baked_encoding': False,

            'side_class': None,
            'side_kwargs': {},
            'side_weights_path': None,

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user can input for smart initialization
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': False},
        } } }
    del n_channels_out

@ex.named_config
def model_transfer_sidetune_encoding():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'eval_only': True, 'normalize_outputs': True},
            'use_baked_encoding': True,

            'side_class': 'FCN5',
            'side_kwargs': {'normalize_outputs': True},
            'side_weights_path': None,  # user can input for smart/student initialization

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user needs to input
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': True},

            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8},
        } } }
    del n_channels_out

@ex.named_config
def model_transfer_nosidetune_encoding():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'eval_only': True, 'normalize_outputs': True},
            'use_baked_encoding': True,

            'side_class': None,
            'side_kwargs': {},
            'side_weights_path': None,

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user needs to input
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': True},

            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8},
        } } }
    del n_channels_out

@ex.named_config
def model_transfer_pix_only_encoder():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user can input for smart initialization
            'base_kwargs': {'eval_only': False, 'normalize_outputs': True},
            'use_baked_encoding': False,

            'side_class': None,
            'side_kwargs': {},
            'side_weights_path': None,

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user needs to input
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': True},

            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8},
        } } }
    del n_channels_out

@ex.named_config
def model_transfer_pix_only_side():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': None,
            'base_weights_path': None,
            'base_kwargs': None,
            'use_baked_encoding': False,

            'side_class': 'FCN5',
            'side_kwargs': {'normalize_outputs': True},
            'side_weights_path': None,  # user can input for smart/student initialization

            'decoder_class': 'TaskonomyDecoder',
            'decoder_weights_path': None,  # user needs to input
            'decoder_kwargs': {'out_channels': n_channels_out, 'eval_only': True},

            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8},
        } } }
    del n_channels_out

@ex.named_config
def model_taskonomy_decoder():
    cfg = { 'learner': {
        'model': 'TaskonomyDecoder',
        'model_kwargs': {
            'out_channels': 3,
            'eval_only': False,
        } } }

@ex.named_config
def model_blind():
    cfg = {
        'learner': {
            'model': 'ConstantModel',
            'model_kwargs': {
                'data': '/mnt/data/normal/median_tiny.png'
            }
        },
        'training': {
            'sources': ['rgb'],
        }
    }

@ex.named_config
def model_unet():
    cfg = { 'learner': {
        'model': 'UNet',
        'model_kwargs': {
            'dcwnsample': 6
        } } }

@ex.named_config
def model_unet_heteroscedastic():
    cfg = { 'learner': {
        'model': 'UNetHeteroscedastic',
        'model_kwargs': {
            'downsample': 6
        } } }

@ex.named_config
def model_unet_hetero_pooled():
    cfg = { 'learner': {
        'model': 'UNetHeteroscedasticPooled',
        'model_kwargs': {
            'downsample': 6
        } } }

####################################
# Generic Sidetune Network models
####################################
@ex.named_config
def gsn_base_resnet50():
    # base is frozen by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
            'use_baked_encoding': True,
        } } }

@ex.named_config
def gsn_base_fcn5s():
    # base is frozen by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'FCN5',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'img_channels': 3, 'eval_only': True, 'normalize_outputs': False},
            'use_baked_encoding': True,
        } } }

@ex.named_config
def gsn_base_learned():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'base_kwargs': {'eval_only': False},
            'use_baked_encoding': False,
        } } }

@ex.named_config
def gsn_side_resnet50():
    # side is learned by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'TaskonomyEncoder',
            'side_weights_path': None,  # user needs to input
            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
        } } }

@ex.named_config
def gsn_side_fcn5s():
    # side is learned by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'FCN5',
            'side_weights_path': None,  # user needs to input
            'side_kwargs': {'img_channels':3, 'eval_only': False, 'normalize_outputs': False},
        } } }

@ex.named_config
def gsn_side_frozen():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'side_kwargs': { 'eval_only': True },
        } } }

@ex.named_config
def gsn_transfer_residual_prenorm():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8, 'residual': True},
            'normalize_pre_transfer': True,
        }
    }}

@ex.named_config
def gsn_merge_concat():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'transfer_class': 'TransferConv3',
            'transfer_weights_path': None,
            'transfer_kwargs': {'n_channels': 8, 'n_channels_in': 2*8, 'residual': True},
            'normalize_pre_transfer': True,
            'alpha_blend': False,
            'concat': True
        }
    }}

####################################
# Losses
####################################
@ex.named_config
def loss_distill_cross_entropy():
    # only for target encodings
    cfg = {}
    cfg['learner'] = {
        # 'lr': 7e-4,                  # Learning rate for algorithm
        'lr': 1e-3,                  # Learning rate for algorithm
        'optimizer_kwargs' : {
            'weight_decay': 0.000001
        },
    }
    cfg['training'] = {
        'loss_fn': 'dense_softmax_cross_entropy', #dense_softmax_cross_entropy weighted_mse_loss
        'targets': ['class_object'],
        'loss_kwargs' : {},
        # if we have decodings, we do not need annotator
        'suppress_target_and_use_annotator': True,  # instead of getting target labels from dataset, apply annotator on inputs
        'annotator_class': 'ResnetiCifar44',
        'annotator_weights_path': '/mnt/models/resnet44-cifar.pth',
        'annotator_kwargs': {},
    }

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
        'annotator_kwargs': {'eval_only': True, 'normalize_outputs': False},
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
        'annotator_kwargs': {'eval_only': True, 'normalize_outputs': False},
    }

@ex.named_config
def loss_softmax_cross_entropy():
    # only for target encodings
    cfg = {}
    cfg['learner'] = {
        'lr': 1e-6,                  # Learning rate for algorithm
        'optimizer_kwargs' : {
            'weight_decay': 0.00001
        },
    }
    cfg['training'] = {
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'loss_fn': 'softmax_cross_entropy', #dense_softmax_cross_entropy weighted_mse_loss
        'targets': ['class_object'],
        'loss_kwargs' : {},
        # if we have decodings, we do not need annotator

    }

####################################
# Optimization
####################################
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

@ex.named_config
def test():
    cfg = { 'training': {
        'train': False,
        'test': True,
    } }


####################################
# Regularizers
# (wraps around loss function)
####################################
@ex.named_config
def treg():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'transfer_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'reg_loss_fn': 'F.l1_loss'
        },
        'loss_list': ['standard', 'final', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }

@ex.named_config
def dreg_t():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'perceptual_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'decoder_path': '/mnt/models/curvature_decoder.dat',  # user needs to input
            'reg_loss_fn': 'F.mse_loss'
        },
        'loss_list': ['standard', 'final', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }


@ex.named_config
def dreg():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'perceptual_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'decoder_path': '/mnt/models/curvature_decoder.dat',  # user needs to input
            'use_transfer': False,
            'reg_loss_fn': 'F.mse_loss'
        },
        'loss_list': ['standard', 'final', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }

