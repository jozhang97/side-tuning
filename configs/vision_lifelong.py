## SACRED THING: Make sure to never set kwargs (or any field that can possibly be a dictionary) to None
## SACRED THING: Make sure not to put any comments in the line after `def config():`
import tlkit.data.splits as splits
import random


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
        'amp': False,
        'post_aggregation_transform_fn': None,
        'post_aggregation_transform_fn_kwargs': {},
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
        'seed': 269,
        'suppress_target_and_use_annotator': False,  # instead of getting target labels from dataset, apply annotator on inputs
        'sources': ['rgb'],
        'targets': ['normal_decoding'],
        'train': True,
        'pretrain': False,
        'test': False,
        'use_masks': True,
        'dataloader_fn': None,
        'dataloader_fn_kwargs': {},
        'sources_as_dict': False,
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



##################
# Merge Methods
##################
@ex.named_config
def merge_alpha():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.Alpha',
            } } }

@ex.named_config
def merge_film():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.FiLM',
            } } }

@ex.named_config
def merge_product():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.Product',
            } } }

@ex.named_config
def merge_resmlp2():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.ResMLP2',
            } },
    }

@ex.named_config
def merge_mlp():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.MLP',
            } },
    }

@ex.named_config
def merge_mlp2():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.MLP2',
            } },
    }

@ex.named_config
def merge_mlp_hidden_a():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.MLPHidden',
                'base_kwargs':   { 'final_act': False, },
                'side_kwargs': { 'final_act': False, },
            } },
    }

@ex.named_config
def merge_mlp_hidden_b():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'merge_method': 'merge_operators.MLPHidden',
                'base_kwargs':   { 'final_act': True,  },
                'side_kwargs': { 'final_act': False, },
            } },
    }

@ex.named_config
def dense():
    # use side_{0,...,i-1} for side_{i}
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'dense': True,
        } },
    }

@ex.named_config
def merge_dense_mlp():
    # use side_{0,...,i-1} for side_{i}
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'dense': True,
            'merge_method': 'merge_operators.MLP',
        } },
    }


##################
# Others
##################
@ex.named_config
def test():
    cfg = { 'training': {
        'train': False,
        'test': True,
    } }

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



##################
# Models
##################
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
            'base_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': True},
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
            'base_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': True},
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
def model_pix_only_base():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user can input for smart initialization
            'base_kwargs': {'train': True, 'eval_only': False, 'normalize_outputs': True},
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
            'base_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': True},
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
            'base_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': True},
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
def model_transfer_pix_only_base():
    n_channels_out = 3
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'n_channels_in': 3,
            'n_channels_out': n_channels_out,

            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user can input for smart initialization
            'base_kwargs': {'train': True, 'eval_only': False, 'normalize_outputs': True},
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


@ex.named_config
def gsn_base_resnet50():
    # base is frozen by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'TaskonomyEncoder',
            'base_weights_path': None,  # user needs to input
            'base_kwargs': {'train': False, 'eval_only': True, 'normalize_outputs': False},
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
            'base_kwargs': {'img_channels': 3, 'train': False, 'eval_only': True, 'normalize_outputs': False},
            'use_baked_encoding': True,
        } } }

@ex.named_config
def gsn_base_learned():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'base_kwargs': {'train': True, 'eval_only': False},
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
            'side_kwargs': {'train': True, 'eval_only': False, 'normalize_outputs': False},
        } } }

@ex.named_config
def gsn_side_fcn5s():
    # side is learned by default
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'side_class': 'FCN5',
            'side_weights_path': None,  # user needs to input
            'side_kwargs': {'img_channels':3, 'train': True, 'eval_only': False, 'normalize_outputs': False},
        } } }

@ex.named_config
def gsn_side_frozen():
    cfg = { 'learner': {
        'model': 'GenericSidetuneNetwork',
        'model_kwargs': {
            'side_kwargs': {'train': False, 'eval_only': True},
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


