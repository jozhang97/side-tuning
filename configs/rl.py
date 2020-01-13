# Habitat configs
#   This should be sourced by the training script,
#   which must save a sacred experiment in the variable "ex"
#   For descriptions of all fields, see configs/core.py


####################################
# Standard methods
####################################
@ex.named_config
def taskonomy_features():
    ''' Implements an agent with some mid-level feature.
        From the paper:
            From Learning to Navigate Using Mid-Level Visual Priors (Sax et al. '19)
            Taskonomy: Disentangling Task Transfer Learning
            Amir R. Zamir, Alexander Sax*, William B. Shen*, Leonidas Guibas, Jitendra Malik, Silvio Savarese.
            2018
        Viable feature options are:
            []
    '''
    uuid = 'habitat_taskonomy_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'main_perception_network': 'TaskonomyFeaturesOnlyNet',  # for sidetune
            }
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'taskonomy':'rescale_centercrop_resize((3,256,256))',
            },
        },
        'transform_fn_post_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_post_aggregation_kwargs': {
            'names_to_transforms': {
                'taskonomy':"taskonomy_features_transform('/mnt/models/curvature_encoder.dat')",
            },
            'keep_unnamed': True,
        }
    }

@ex.named_config
def blind():
    ''' Implements a blinded agent. This has no visual input, but is still able to reason about its movement
        via path integration.
    '''
    uuid = 'blind'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'taskonomy': 'blind((8,16,16))',
                #       'rgb_filled': 'rescale_centercrop_resize((3,84,84))',
            },
        },
    }

@ex.named_config
def midtune():
    # Specific type of finetune where we train the policy then open the representation to be learned.
    # Specifically, we take trained midlevel agents and finetune all the weights.
    uuid = 'habitat_midtune'
    cfg = {}
    cfg['learner'] = {
        'perception_network_reinit': True,  # reinitialize the perception_module, used when checkpoint is used
        'rollout_value_batch_multiplier': 1,
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'main_perception_network': 'TaskonomyFeaturesOnlyNet',  # for sidetune
                'sidetune_kwargs': {
                    'n_channels_in': 3,
                    'n_channels_out': 8,
                    'normalize_pre_transfer': False,
                    'base_class': 'FCN5',
                    'base_kwargs': {'normalize_outputs': False},
                    'base_weights_path': None,  # user needs to specify
                    'side_class': 'FCN5',
                    'side_kwargs': {'normalize_outputs': False},
                    'side_weights_path': None,  # user needs to specify
                }
            }
        },
    }
    cfg['saving'] = {
        'checkpoint': None,
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'rgb_filled': 'rescale_centercrop_resize((3,256,256))',
            },
        },
    }

@ex.named_config
def finetune():
    uuid = 'habitat_finetune'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'main_perception_network': 'TaskonomyFeaturesOnlyNet',  # for sidetune
                'sidetune_kwargs': {
                    'n_channels_in': 3,
                    'n_channels_out': 8,
                    'normalize_pre_transfer': False,
                    'side_class': 'FCN5',
                    'side_kwargs': {'normalize_outputs': False},
                    'side_weights_path': None,  # user needs to specify
                }
            }
        },
        'rollout_value_batch_multiplier': 1,
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'rgb_filled': 'rescale_centercrop_resize((3,256,256))',
            },
        },
    }

@ex.named_config
def sidetune():
    uuid = 'habitat_sidetune'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'n_channels_in': 3,
                    'n_channels_out': 8,
                    'normalize_pre_transfer': False,

                    'base_class': 'TaskonomyEncoder',
                    'base_weights_path': None,
                    'base_kwargs': {'eval_only': True, 'normalize_outputs': False},

                    'side_class': 'FCN5',
                    'side_kwargs': {'normalize_outputs': False},
                    'side_weights_path': None,

                    'alpha_blend': True,
                },
                'attrs_to_remember': ['base_encoding', 'side_output', 'merged_encoding'],   # things to remember for supp. losses / visualization
            }
        },
        'rollout_value_batch_multiplier': 1,
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'rgb_filled': 'rescale_centercrop_resize((3,256,256))',
            },
        },
    }

####################################
# Base Network
####################################
@ex.named_config
def rlgsn_base_resnet50():
    # base is frozen by default
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'base_class': 'TaskonomyEncoder',
                    'base_weights_path': None,  # user needs to input
                    'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
                }
            }
        },
    }

@ex.named_config
def rlgsn_base_fcn5s():
    # base is frozen by default
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'base_class': 'FCN5',
                    'base_weights_path': None,  # user needs to input
                    'base_kwargs': {'eval_only': True, 'normalize_outputs': False},
                }
            }
        },
    }

@ex.named_config
def rlgsn_base_learned():
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'base_kwargs': {'eval_only': False},
                }
            }
        },
    }

####################################
# Side Network
####################################
@ex.named_config
def rlgsn_side_resnet50():
    # side is learned by default
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'side_class': 'TaskonomyEncoder',
                    'side_weights_path': None,  # user needs to input
                    'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
                }
            }
        },
    }

@ex.named_config
def rlgsn_side_fcn5s():
    # side is learned by default
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'side_class': 'FCN5',
                    'side_weights_path': None,  # user needs to input
                    'side_kwargs': {'eval_only': False, 'normalize_outputs': False},
                }
            }
        },
    }

@ex.named_config
def rlgsn_side_frozen():
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'sidetune_kwargs': {
                    'side_kwargs': {'eval_only': True},
                }
            }
        },
    }


