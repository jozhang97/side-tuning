from gym import spaces

####################################
# Task
####################################
@ex.named_config
def imitation_learning():
    # default is a sidetune network setup
    cfg = {}
    cfg['learner'] = {
        'model': 'PolicyWithBase',
        'lr': 0.0002,
        'optimizer_kwargs' : {
            'weight_decay': 3.8e-7
        },
        'model_kwargs': {
            'base': None,
            'action_space': spaces.Discrete(3),
            'base_class': 'NaivelyRecurrentACModule',
            'base_kwargs': {
                'use_gru': False,
                'internal_state_size': 512,
                'perception_unit': None,
                'perception_unit_class': 'RLSidetuneWrapper',
                'perception_unit_kwargs': {
                    'n_frames': 4,
                    'n_map_channels': 3,
                    'use_target': True,
                    'blind': False,
                    'extra_kwargs': {
                        'main_perception_network': 'TaskonomyFeaturesOnlyNet',  # for sidetune
                        'sidetune_kwargs': {
                            'n_channels_in': 3,
                            'n_channels_out': 8,
                            'normalize_pre_transfer': False,

                            'base_class': None,
                            'base_weights_path': None,
                            'base_kwargs': {},

                            'side_class': 'FCN5',
                            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
                            'side_weights_path': None,
                        }
                    }
                }
            }
        }
    }
    cfg['training'] = {
        'sources': ['rgb_filled', 'map', 'target', 'taskonomy'],
        'sources_as_dict': True,
        'targets': ['action'],
    }

@ex.named_config
def expert_data():
    cfg = {
        'training': {
            'dataloader_fn': 'expert_dataset.get_dataloaders',
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/large',
                'num_frames': 4,
                'load_to_mem': False,  # if dataset small enough, can load activations to memory ???
                'num_workers': 8,
                'pin_memory': False,
                'batch_size': 64,
                'batch_size_val': 64,
                'remove_last_step_in_traj': True,  # remove STOP action
                # 'removed_actions': [3],  # actions to make random, need if one hot
            },
            'loss_fn': 'softmax_cross_entropy',
            'loss_kwargs': {},
            'use_masks': False,
        }
    }

@ex.named_config
def il_source_rmtt():
    cfg = {}
    cfg['training'] = {
        'sources': ['rgb_filled', 'map', 'target', 'taskonomy'],
        'sources_as_dict': True,
    }

@ex.named_config
def il_source_rmt():
    cfg = {}
    cfg['training'] = {
        'sources': ['rgb_filled', 'map', 'target'],
        'sources_as_dict': True,
    }


####################################
# Methods
####################################
@ex.named_config
def il_blind():
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'main_perception_network': 'TaskonomyFeaturesOnlyNet',
                        'sidetune_kwargs': {
                            'base_class': None,
                            'base_weights_path': None,
                            'base_kwargs': {},

                            'side_class': None,
                            'side_weights_path': None,
                            'side_kwargs': {},
                        }
                    }
                }
            }
        }
    }
    cfg['training'] = {
        'sources': ['map', 'target'],
    }

@ex.named_config
def il_sidetune():
    cfg = {}
    cfg['learner'] = {
            'model_kwargs': {
                'base_kwargs': {
                    'perception_unit_kwargs': {
                        'extra_kwargs': {
                            'sidetune_kwargs': {
                                'n_channels_in': 3,
                                'n_channels_out': 8,
                                'normalize_pre_transfer': False,

                                'base_class': 'TaskonomyEncoder',
                                'base_weights_path': None,
                                'base_kwargs': {'eval_only': True, 'normalize_outputs': False},

                                'side_class': 'FCN5',
                                'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
                                'side_weights_path': None,

                                'alpha_blend': True,
                            },
                            'attrs_to_remember': ['base_encoding', 'side_output', 'merged_encoding'],   # things to remember for supp. losses
                        }
                    }
                }
            }
        }

####################################
# Dataset Sizes
####################################
@ex.named_config
def il_debug():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/tiny',
            },
            'num_epochs': 10,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 5,
            'save_interval': 5,
        }
    }

@ex.named_config
def il_tiny():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/tiny',
            },
            'num_epochs': 3000,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 500,
            'save_interval': 100,
        }
    }

@ex.named_config
def il_small():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/small',
            },
            'num_epochs': 600,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 20,
            'save_interval': 20,
        }
    }

@ex.named_config
def il_medium():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/medium',
            },
            'num_epochs': 100,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 10,
            'save_interval': 10,
        }
    }

@ex.named_config
def il_large():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/large',
            },
            'num_epochs': 12,
        },
        'saving': {
            'ticks_per_epoch': 1,
            'log_interval': 1,
            'save_interval': 1,
        }
    }

@ex.named_config
def il_largeplus():
    cfg = {
        'training': {
            'dataloader_fn_kwargs': {
                'data_path': '/mnt/data/expert_trajs/largeplus',
            },
            'num_epochs': 5,
        },
        'saving': {
            'ticks_per_epoch': 100,
            'log_interval': 0.1,
            'save_interval': 1,
        }
    }


####################################
# Base
####################################
@ex.named_config
def ilgsn_base_resnet50():
    # base is frozen by default
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'base_class': 'TaskonomyEncoder',
                            'base_weights_path': None,  # user needs to input
                            'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
    }}}}}}

@ex.named_config
def ilgsn_base_fcn5():
    # base is frozen by default
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'base_class': 'FCN5',
                            'base_weights_path': None,  # user needs to input
                            'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
    }}}}}}

@ex.named_config
def ilgsn_base_learned():
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'base_kwargs': {'eval_only': False},
    }}}}}}


####################################
# Side
####################################
@ex.named_config
def ilgsn_side_resnet50():
    # side is learned by default
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'side_class': 'TaskonomyEncoder',
                            'side_weights_path': None,  # user needs to input
                            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
    }}}}}}

@ex.named_config
def ilgsn_side_fcn5():
    # side is learned by default
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'side_class': 'FCN5',
                            'side_weights_path': None,  # user needs to input
                            'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
    }}}}}}

@ex.named_config
def ilgsn_no_side():
    # side is learned by default
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'side_class': None,
                            'side_weights_path': None,  # user needs to input
                            'side_kwargs': {},
                            'alpha_blend': False,
                        }}}}}}

@ex.named_config
def ilgsn_side_frozen():
    cfg = {}
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'side_kwargs': {'eval_only': True},
    }}}}}

####################################
# Alpha Blending
####################################
@ex.named_config
def alpha_blend():
    cfg = {}
    cfg['learner'] = {
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'alpha_blend': True
                }
            }
        },
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'alpha_blend': True,
                        }}}}}}

@ex.named_config
def alpha8():
    cfg = {}
    cfg['learner'] = {
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'alpha_blend': True,
                    'alpha_kwargs': {'init_value': 1.39},
                }
            }
        },
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'sidetune_kwargs': {
                            'alpha_blend': True,
                            'alpha_kwargs': {'init_value': 1.39},
                        }}}}}}

####################################
# Regularization
####################################
@ex.named_config
def dreg_il():
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
        'batch_size': 16,
    }
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'attrs_to_remember': ['base_encoding', 'merged_encoding', 'side_output']   # things to remember for supp. losses
                    }
                }
            }
        }
    }

@ex.named_config
def treg_il():
    cfg = {}
    cfg['training'] = {
        'regularizer_fn': 'transfer_regularizer',
        'regularizer_kwargs': {
            'coef': 1e-3,
            'reg_loss_fn': 'F.l1_loss'
        },
        'loss_list': ['standard', 'final', 'weight_tying'],      # ones to report, 'standard' and 'final' are already represented
    }
    cfg['learner'] = {
        'model_kwargs': {
            'base_kwargs': {
                'perception_unit_kwargs': {
                    'extra_kwargs': {
                        'attrs_to_remember': ['base_encoding', 'transfered_encoding']   # things to remember for supp. losses
                    }
                }
            }
        }
    }

####################################
# Debug
####################################
@ex.named_config
def expert():
    # read from a trajectory sequence the actions to take
    uuid='habitat_expert'
    cfg = {}
    override = {}
    cfg['learner'] = {
        'algo': 'expert',
        'algo_class': 'Expert',
        'algo_kwargs': {
            'data_dir': '/mnt/data/expert_trajs/large',
            'compare_with_saved_trajs': False,
            'follower': None,
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',        # Environment to use
    }
    override['env'] = {
        'num_processes': 1,
        'num_val_processes': 1,
        "env_specific_kwargs": {
            'debug_mode': True,
        }
    }

