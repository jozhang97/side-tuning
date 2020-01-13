from evkit.utils.misc import append_dict
import numpy as np
##################
# Old to New name mappings
# (always use taskonomy_base_data now)
# taskonomy_louis_data -> taskonomy_3_data
# taskonomy_louis_rand_data -> taskonomy_shuffle3_data
# taskonomy_big_data -> taskonomy_12_data
# taskonomy_big_rand_data -> taskonomy_shuffle12_data
# taskonomy_debug_data -> taskonomy_12_data debug2 (probably a good one to run more)
##################



##################
# Task-Specific Configs
##################
epochs_per_task = 3
stop_recurse_keys = ['task_specific_transfer_kwargs']  # helper to update configs
get_geometric_kwargs = lambda n_out: {
    'training': {
        'loss_fn': 'weighted_l1_loss',
        'masks': True,
        'use_masks': True,
        'task_is_classification': False,
    },
    'learner': {'model_kwargs': {'task_specific_transfer_kwargs': {'out_channels': n_out,    'is_decoder_mlp': False}}}
}
tasks_to_kwargs = {
    'normal': get_geometric_kwargs(3),
    'rgb': get_geometric_kwargs(3),
    'reshading': get_geometric_kwargs(1),
    'keypoints3d': get_geometric_kwargs(1),
    'keypoints2d': get_geometric_kwargs(1),
    'edge_texture': get_geometric_kwargs(1),
    'edge_occlusion': get_geometric_kwargs(1),
    'depth_zbuffer': get_geometric_kwargs(1),
    'depth_euclidean': get_geometric_kwargs(1),
    'curvature': {
        'training': {
            'loss_fn': 'weighted_l2_loss',
            'masks': True,
            'use_masks': True,
            'task_is_classification': False,
        },
        'learner': {'model_kwargs': {'task_specific_transfer_kwargs': {'out_channels': 2,    'is_decoder_mlp': False}}}
    },
    'segment_semantic': {
        'training': {
            'loss_fn': 'softmax_cross_entropy',
            'masks': False,
            'use_masks': False,
            'task_is_classification': False,
        },
        'learner': {'model_kwargs': {'task_specific_transfer_kwargs': {'out_channels': 18,    'is_decoder_mlp': False}}}
    },
    'class_object': {
        'training': {
            'loss_fn': 'dense_cross_entropy',
            'masks': False,
            'use_masks': False,
            'task_is_classification': True,
        },
        'learner': {'model_kwargs': {'task_specific_transfer_kwargs': {'out_channels': 1000,    'is_decoder_mlp': True}}}
    }
}
tasks = list(tasks_to_kwargs.keys())
tasks_np = np.array(tasks)

##################
# Datasets
##################
@ex.named_config
def taskonomy_base_data():
    cfg = {
        'training': {
            'dataloader_fn': 'taskonomy_dataset.get_lifelong_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/tiny',
                'split': 'tiny',
                'load_to_mem': False,  # if dataset small enough, can load activations to memory
                'num_workers': 8,
                'pin_memory': True,
                'epochs_per_task': epochs_per_task,
                'epochs_until_cycle': 0,
                'batch_size': 32,
                'batch_size_val': 64,
            },
        },
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'dataset': 'taskonomy',
            },
            'optimizer_class': 'optim.Adam',
            'lr': 1e-4,
            'optimizer_kwargs' : {
                'weight_decay': 2e-6
            },
            'use_feedback': False,
        },
        'saving': {
            'ticks_per_epoch': 100,
            'log_interval': 1/epochs_per_task,
            'save_interval': 1,
        },
    }

@ex.named_config
def taskonomy_3_data():
    uuid = 'louis_taskonomy'
    n_tasks = 3
    cfg = {
        'training': {
            'num_epochs': 3,
            'dataloader_fn_kwargs': {
                'split': 'debug2',
            },
            'loss_fn': ['weighted_l1_loss', 'weighted_l1_loss', 'weighted_l1_loss'],
            'loss_kwargs': [{}] * n_tasks,
            'sources': [['rgb']] * n_tasks,
            'targets': [['normal'], ['rgb'], ['normal']],
            'masks': [True, True, True],
            'use_masks': [True, True, True],
            'task_is_classification': [False, False, False],
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
                'task_specific_side_kwargs': [{} for _ in range(n_tasks)],
                'task_specific_transfer_kwargs': [
                    {'out_channels': 3,    'is_decoder_mlp': False},
                    {'out_channels': 3,    'is_decoder_mlp': False},
                    {'out_channels': 3,    'is_decoder_mlp': False},
                ],
            },
        },
    }
    del n_tasks

@ex.named_config
def taskonomy_shuffle3_data():
    tasks = ['normal', 'rgb', 'normal']
    random.shuffle(tasks)
    n_tasks = 3

    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'dataloader_fn_kwargs': {
                'split': 'debug2',  # debug2
            },
            'sources': [['rgb']] * n_tasks,
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
            },
        },
    }

    cfg['training']['targets'] = [[t] for t in tasks]
    for task in tasks:
        cfg_task = tasks_to_kwargs[task]
        cfg = append_dict(cfg, cfg_task, stop_recurse_keys=stop_recurse_keys)
    del task, cfg_task, tasks, n_tasks


@ex.named_config
def taskonomy_12_data():
    n_tasks = 12
    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'loss_fn': ['weighted_l2_loss', 'softmax_cross_entropy', 'weighted_l1_loss',
                        'weighted_l1_loss', 'weighted_l1_loss',  'weighted_l1_loss',  'weighted_l1_loss',  'weighted_l1_loss',
                        'weighted_l1_loss', 'weighted_l1_loss', 'dense_cross_entropy', 'weighted_l1_loss'],
            'loss_kwargs': [{}] * n_tasks,
            'sources': [['rgb']] * n_tasks,
            'targets': [['curvature'], ['segment_semantic'], ['reshading'],
                        ['keypoints3d'], ['keypoints2d'], ['edge_texture'], ['edge_occlusion'], ['depth_zbuffer'],
                        ['depth_euclidean'], ['normal'], ['class_object'], ['rgb']],
            'masks': [True, False, True,
                      True, True, True, True, True,
                      True, True, False, True],
            'use_masks': [True, False, True,
                          True, True, True, True, True,
                          True, True, False, True],
            'task_is_classification': [False, False, False,
                                       False, False, False, False, False,
                                       False, False, True, False],
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
                'task_specific_side_kwargs': [{} for _ in range(n_tasks)],
                'task_specific_transfer_kwargs': [
                    {'out_channels': 2,    'is_decoder_mlp': False},  # curvature
                    {'out_channels': 18,    'is_decoder_mlp': False}, # segment_semantic
                    {'out_channels': 1,    'is_decoder_mlp': False},  # reshading
                    {'out_channels': 1,    'is_decoder_mlp': False},  # keypoints3d
                    {'out_channels': 1,    'is_decoder_mlp': False},  # keypoints2d
                    {'out_channels': 1,    'is_decoder_mlp': False},  # edge_texture
                    {'out_channels': 1,    'is_decoder_mlp': False},  # edge_occlusion
                    {'out_channels': 1,    'is_decoder_mlp': False},  # depth_zbuffer
                    {'out_channels': 1,    'is_decoder_mlp': False},  # depth_euclidean
                    {'out_channels': 3,    'is_decoder_mlp': False},  # normal
                    {'out_channels': 1000, 'is_decoder_mlp': True},   # class_object
                    {'out_channels': 3,    'is_decoder_mlp': False},  # rgb
                ],
            },
        },
    }
    del n_tasks


@ex.named_config
def taskonomy_shuffle12_data():
    # in order to randomize the order of the tasks, I need to change the order here
    # and also update train_lifelong.py to switch to the proper TASK0_encoder.dat or TASK0_encoder_student.dat
    random.shuffle(tasks)
    n_tasks = len(tasks)

    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'sources': [['rgb']] * n_tasks,
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
        }, }, }

    cfg['training']['targets'] = [[t] for t in tasks]
    for task in tasks:
        cfg_task = tasks_to_kwargs[task]
        cfg = append_dict(cfg, cfg_task, stop_recurse_keys=stop_recurse_keys)
    del task, cfg_task, n_tasks


@ex.named_config
def taskonomy_12cls_data():
    order = [11, 7, 0, 2, 8, 9, 5, 10, 4, 1, 3, 6]
    tasks = list(tasks_np[order])
    n_tasks = len(tasks)

    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'sources': [['rgb']] * n_tasks,
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
            }, }, }

    cfg['training']['targets'] = [[t] for t in tasks]
    for task in tasks:
        cfg_task = tasks_to_kwargs[task]
        cfg = append_dict(cfg, cfg_task, stop_recurse_keys=stop_recurse_keys)
    del task, cfg_task, n_tasks, order, tasks


@ex.named_config
def taskonomy_12txtr_data():
    order = [5, 8, 9, 10, 1, 3, 6, 2, 11, 4, 0, 7]
    tasks = list(tasks_np[order])
    n_tasks = len(tasks)

    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'sources': [['rgb']] * n_tasks,
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
            }, }, }

    cfg['training']['targets'] = [[t] for t in tasks]
    for task in tasks:
        cfg_task = tasks_to_kwargs[task]
        cfg = append_dict(cfg, cfg_task, stop_recurse_keys=stop_recurse_keys)
    del task, cfg_task, n_tasks, order, tasks

@ex.named_config
def taskonomy_12rgb_data():
    order = [1, 0, 5, 7, 2, 6, 4, 3, 9, 10, 8, 11]
    tasks = list(tasks_np[order])
    n_tasks = len(tasks)

    cfg = {
        'training': {
            'num_epochs': n_tasks,
            'sources': [['rgb']] * n_tasks,
        },
        'learner': {
            'model_kwargs': {
                'tasks': list(range(n_tasks)),
            }, }, }

    cfg['training']['targets'] = [[t] for t in tasks]
    for task in tasks:
        cfg_task = tasks_to_kwargs[task]
        cfg = append_dict(cfg, cfg_task, stop_recurse_keys=stop_recurse_keys)
    del task, cfg_task, n_tasks, order, tasks

@ex.named_config
def taskonomy_louis_gtnormal_data():
    cfg = {
        'training': {
            'sources': [['rgb', 'normal']] * N_TASKONOMY_TASKS,
        },
        'learner': {
            'model_kwargs': {
                'base_class': 'SampleGroupStackModule',
                'base_weights_path': None,
                'base_uses_other_sensors': True,
            },
        },
    }

@ex.named_config
def taskonomy_gtcurv_data():
    cfg = {
        'training': {
            'sources': [['rgb', 'curvature']] * 12,
        },
        'learner': {
            'model_kwargs': {
                'base_class': 'SampleGroupStackModule',
                'base_weights_path': None,
                'base_uses_other_sensors': True,
            }
        },
    }


@ex.named_config
def debug2():
    cfg = { 'training': { 'dataloader_fn_kwargs': { 'split': 'debug2' } } }

##################
# Models
##################

@ex.named_config
def model_lifelong_independent_std_taskonomy():
    # Independent
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'side_class': 'GenericSidetuneNetwork',
                'side_kwargs': {
                    'n_channels_in': 3,
                    'n_channels_out': 8,
                    'base_class': 'TaskonomyEncoder',
                    'base_kwargs': {'eval_only': False, 'normalize_outputs': False },
                    'base_weights_path': '/mnt/models/curvature_encoder.dat',
                    'use_baked_encoding': False,
                    'normalize_pre_transfer': False,

                    'side_class': 'FCN5',
                    'side_kwargs': {'eval_only': False, 'normalize_outputs': False },
                    'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
                },
                'normalize_pre_transfer': True,
            } },
        'training': {
            'dataloader_fn': 'taskonomy_dataset.get_lifelong_dataloaders',
            'dataloader_fn_kwargs': {
                'speedup_no_rigidity': True,
                'batch_size': 16,
            }
        }
    }

@ex.named_config
def projected():
    # used with indepenedent_std to calibrate rigidity for parameter superposition
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'side_class': 'GenericSidetuneNetwork',
                'side_kwargs': {
                    'base_kwargs': {'projected': True},
                    'side_kwargs': {'projected': True},
                },
            } } }

@ex.named_config
def model_lifelong_sidetune_std_taskonomy():
    # Sidetuning
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'base_class': 'TaskonomyEncoder',
                'base_weights_path': '/mnt/models/curvature_encoder.dat',  # user needs to input
                'base_kwargs': {'eval_only': True, 'normalize_outputs': False },
                'use_baked_encoding': False,

                'side_class': 'FCN5',
                'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
                'side_kwargs': {'eval_only': False, 'normalize_outputs': False },

                'normalize_pre_transfer': True,
                'merge_method': 'merge_operators.Alpha',
            } },
        'training': {
            'dataloader_fn_kwargs': { 'speedup_no_rigidity': True }  # speedup is partially tested, remove if issues
        }
    }


@ex.named_config
def model_lifelong_sidetune_nobase_taskonomy():
    # How much is the base helping? This is related to Independent with fewer parameters.
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'base_class': None,
                'use_baked_encoding': False,

                'side_class': 'FCN5',
                'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
                'side_kwargs': {'eval_only': False, 'normalize_outputs': False},

                'normalize_pre_transfer': True,
            } },
        'training': {
            'dataloader_fn_kwargs': { 'speedup_no_rigidity': True }  # speedup is partially tested, remove if issues
        }
    }

@ex.named_config
def model_lifelong_finetune_std_taskonomy():
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'base_class': 'GenericSidetuneNetwork',
            'base_kwargs': {
                'n_channels_in': 3,
                'n_channels_out': 8,
                'base_class': 'TaskonomyEncoder',
                'base_kwargs': {'eval_only': False, 'normalize_outputs': False },
                'base_weights_path': '/mnt/models/curvature_encoder.dat',
                'use_baked_encoding': False,
                'normalize_pre_transfer': False,

                'side_class': 'FCN5',
                'side_kwargs': {'eval_only': False, 'normalize_outputs': False },
                'side_weights_path': '/mnt/models/curvature_encoder_student.dat',
            },
            'normalize_pre_transfer': True,
        } } }

@ex.named_config
def model_lifelong_features_std_taskonomy():
    cfg = {
        'learner': {
            'model': 'LifelongSidetuneNetwork',
            'model_kwargs': {
                'base_class': 'TaskonomyEncoder',
                'base_weights_path': '/mnt/models/curvature_encoder.dat',  # user needs to input
                'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
                'use_baked_encoding': False,

                'side_class': None,
                'side_weights_path': None,

                'normalize_pre_transfer': True,
            } },
        'training': {
            'dataloader_fn_kwargs': { 'speedup_no_rigidity': True }  # speedup is partially tested, remove if issues
        }
    }

@ex.named_config
def pnn_v4_mlp():
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
            'merge_method': 'merge_operators.MLP',
        } },
    }

@ex.named_config
def model_learned_decoder():
    # This should be default
    cfg = { 'learner': {
        'model': 'LifelongSidetuneNetwork',
        'model_kwargs': {
            'transfer_class': 'PreTransferedDecoder',
            'transfer_kwargs': {
                'transfer_class': 'TransferConv3',
                'transfer_weights_path': None,
                'transfer_kwargs': {'n_channels': 8, 'residual': True},

                'decoder_class': 'TaskonomyDecoder',
                'decoder_weights_path': None,  # user can input for smart initialization
                'decoder_kwargs': {'eval_only': False},
            },
        } } }

