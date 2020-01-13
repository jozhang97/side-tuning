####################################
# Baselines
####################################
@ex.named_config
def alexnet():
    ''' Implements an agent with some alexnet features. '''
    uuid = 'habitat_alexnet_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AlexNetFeaturesOnlyNet',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'main_perception_network': 'AlexNetFeaturesOnlyNet',  # for sidetune
            }
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 13,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'taskonomy':'alexnet_transform((3, 224, 224))',
            },
        },
        'transform_fn_post_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_post_aggregation_kwargs': {
            'names_to_transforms': {
                'taskonomy':"alexnet_features_transform('{load_path}')".format(load_path='/mnt/models/alexnet-owt-4df8aa71.pth'),
            },
            'keep_unnamed': True,
        },
    }

@ex.named_config
def slam():
    uuid='habitat_slam'
    cfg = {}
    cfg['learner'] = {
        'algo': 'slam',
        'num_stack': 1,              # Frames that each cell (CNN) can see
        'slam_class': "DepthMapperAndPlanner",
        'slam_kwargs': {
            'map_size_cm': 1200,
            'mark_locs': False,
            'reset_if_drift': False,
            'count': -1,
            'close_small_openings': False,
            'recover_on_collision': False,
            'fix_thrashing': False,
            'goal_f': 1.1,
            'point_cnt': 2,
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',        # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {               # specific to the scenario - pointnav or exploration
                'use_depth': True,
            },
        },
        'transform_fn_pre_aggregation': None,  # Transformation to apply to each individual image (before batching)
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'rgb_filled':identity_transform(),
                    'depth':identity_transform(),
                    'pointgoal':identity_transform(),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }
    override = {}
    override['env'] = {
        'num_processes': 1,
        'num_val_processes': 1,
        "env_specific_kwargs": {
            'debug_mode': True,
        },
    }

@ex.named_config
def slam_estimated():
    uuid='habitat_slam_estimated_depth'
    cfg = {}
    cfg['learner'] = {
        'algo': 'slam',
        'num_stack': 1,              # Frames that each cell (CNN) can see
        'slam_class': "TaskonomyDepthMapperAndPlanner",
        'slam_kwargs': {
            'map_size_cm': 1200,
            'out_dir': None,
            'mark_locs': True,
            'reset_if_drift': True,
            'count': -1,
            'close_small_openings': True,
            'recover_on_collision': False,
            'fix_thrashing': False,
            'goal_f': 1.1,
            'point_cnt': 2,
            'depth_estimator_kwargs': {
                'load_encoder_path': '/mnt/models/depth_euclidean_encoder.dat',
                'load_decoder_path': '/mnt/models/depth_euclidean_decoder.dat',
            }
        }
    }
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',        # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {               # specific to the scenario - pointnav or exploration
                'use_depth': False,
            },
        },
        'transform_fn_pre_aggregation': None,  # Transformation to apply to each individual image (before batching)
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'rgb_filled':identity_transform(),
                    'pointgoal':identity_transform(),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }
    override = {}
    override['env'] = {
        'num_processes': 1,
        'num_val_processes': 1,
        "env_specific_kwargs": {
            'debug_mode': True,
        },
    }

@ex.named_config
def srl_features():
    ''' Implements an agent with some alexnet features. '''
    uuid = 'habitat_alexnet_feature'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'BaseModelAutoEncoder',
        'perception_network_kwargs': {
            'n_map_channels': 1,
            'use_target': False,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 6,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy': rescale_centercrop_resize((3,224,224)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':srl_features_transform('{load_path}'),
                    'map':identity_transform(),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            load_path='/mnt/share/midlevel_control/baselines/srl_models/HabitatPlanning/forward_inverse/srl_model.pth'),
    }

@ex.named_config
def curiosity():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_curiosity'
    cfg = {}
    cfg['learner'] = {
        'curiosity_reward_coef':0.1,
        'forward_loss_coef':0.2,
        'inverse_loss_coef':0.8,
    }

@ex.named_config
def scratch():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_scratch_map'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'rgb_filled': 'rescale_centercrop_resize((3,84,84))',
            },
        },
    }

####################################
# Feature Sets
####################################
@ex.named_config
def all_features_resnet():
    features_list_resnet = ['denoising', 'egomotion', 'fixated_pose', 'jigsaw',
                            'nonfixated_pose', 'point_matching', 'room_layout', 'segment_unsup25d',
                            'segment_unsup2d', 'segment_semantic', 'class_scene', 'inpainting', 'vanishing_point',
                            'autoencoding', 'class_object', 'curvature', 'denoising', 'depth_euclidean',
                            'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d',
                            'normal', 'reshading']
    features_list_distil = []
    features_paths = [f'/mnt/models/{feat}_encoder.dat' for feat in features_list_resnet] + \
                     [f'/mnt/models/{feat}-distilled.pth' for feat in features_list_distil]

    uuid = 'many_features'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': len(features_paths),
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=features_paths),
    }
    del features_list_resnet, features_list_distil, features_paths

@ex.named_config
def all_features():
    features_list_resnet = ['denoising', 'egomotion', 'fixated_pose', 'jigsaw',
                            'nonfixated_pose', 'point_matching', 'room_layout', 'segment_unsup25d',
                            'segment_unsup2d', 'segment_semantic', 'class_scene', 'inpainting', 'vanishing_point']
    features_list_distil = ['autoencoding', 'class_object', 'curvature', 'denoising', 'depth_euclidean',
                            'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d',
                            'normal', 'reshading']
    features_paths = [f'/mnt/models/{feat}_encoder.dat' for feat in features_list_resnet] + \
                     [f'/mnt/models/{feat}-distilled.pth' for feat in features_list_distil]

    uuid = 'many_features'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': len(features_paths),
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=features_paths),
    }
    del features_list_resnet, features_list_distil, features_paths

@ex.named_config
def many_features():
    features_list_resnet = ['fixated_pose', 'jigsaw', 'random', 'room_layout', 'segment_unsup25d',
                            'segment_unsup2d', 'segment_semantic', 'class_scene' ]
    features_list_distil = ['autoencoding', 'class_object', 'curvature', 'denoising', 'depth_euclidean',
                            'depth_zbuffer', 'edge_occlusion', 'keypoints3d',
                            'normal', 'reshading']
    features_paths = [f'/mnt/models/{feat}_encoder.dat' for feat in features_list_resnet] + \
                     [f'/mnt/models/{feat}-distilled.pth' for feat in features_list_distil]

    uuid = 'many_features'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': len(features_paths),
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=features_paths),
    }
    del features_list_resnet, features_list_distil, features_paths

@ex.named_config
def max_coverage_perception():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 4,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=['/mnt/models/normal_encoder.dat',
                           '/mnt/models/keypoints2d_encoder.dat',
                           '/mnt/models/segment_unsup2d_encoder.dat',
                           '/mnt/models/segment_semantic_encoder.dat']),
    }

@ex.named_config
def max_coverage_perception3():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 3,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=['/mnt/models/edge_texture_encoder.dat',
                           '/mnt/models/curvature_encoder.dat',
                           '/mnt/models/reshading_encoder.dat']),
    }

@ex.named_config
def max_coverage_perception2():
    ''' Implements an agent with a Max-Coverage Min-Distance Featureset
        From the paper:
            Mid-Level Visual Representations Improve Generalization and Sample Efficiency for Learning Visuomotor Policies
            Alexander Sax, Bradley Emi, Amir R. Zamir, Silvio Savarese, Leonidas Guibas, Jitendra Malik.
            2018
    '''
    uuid = 'habitat_max_coverage_featureset'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
            'num_tasks': 2,
        },
    }
    cfg['env'] = {
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_multi_features_transform({encoder_paths}),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder_paths=['/mnt/models/segment_unsup2d_encoder.dat',
                           '/mnt/models/segment_unsup25d_encoder.dat']),
    }

####################################
# No Map/IMU
####################################
@ex.named_config
def taskonomy_features_nomap():
    ''' Implements an agent with some mid-level feature.
        From the paper:
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
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                   'taskonomy':rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'taskonomy':taskonomy_features_transform('{taskonomy_encoder}'),
                    'target':identity_transform(),
                    'map':blind((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            taskonomy_encoder='/mnt/models/normal_encoder.dat'),
    }

@ex.named_config
def scratch_nomap():
    # scratch is not compatible with collate because we need to perform Image operations (resize) to go from
    # 256 to 84. This is not implemented with collate code
    uuid = 'habitat_scratch_map'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        }
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'map':blind((3,84,84)),
                    'rgb_filled':rescale_centercrop_resize((3,84,84)),
                    'target':identity_transform(),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': None,
    }

@ex.named_config
def blind_nomap():
    ''' Implements a blinded agent. This has no visual input, but is still able to reason about its movement
        via path integration.
    '''
    uuid = 'blind'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'TaskonomyFeaturesOnlyNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 16,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
            TransformFactory.independent(
                {
                    'taskonomy': blind((8,16,16)),
                    'target': identity_transform(),
                    'map': blind((3, 84, 84)),
                    'global_pos':identity_transform(),
                },
                keep_unnamed=False)
            """.translate(remove_whitespace),
    }


####################################
# Train on fewer buidlings
####################################
all_buildings = ['Rancocas', 'Cooperstown', 'Hominy', 'Placida', 'Arkansaw', 'Delton', 'Capistrano', 'Mesic', 'Roeville', 'Angiola', 'Mobridge', 'Nuevo', 'Oyens', 'Quantico', 'Colebrook', 'Sawpit', 'Hometown', 'Sasakwa', 'Stokes', 'Soldier', 'Rosser', 'Superior', 'Nemacolin', 'Pleasant', 'Eagerville', 'Sanctuary', 'Hainesburg', 'Avonia', 'Crandon', 'Spotswood', 'Roane', 'Dunmor', 'Spencerville', 'Goffs', 'Silas', 'Applewold', 'Nicut', 'Shelbiana', 'Azusa', 'Reyno', 'Dryville', 'Haxtun', 'Ballou', 'Adrian', 'Stanleyville', 'Monson', 'Stilwell', 'Seward', 'Hambleton', 'Micanopy', 'Parole', 'Nimmons', 'Pettigrew', 'Bolton', 'Sumas', 'Sodaville', 'Mosinee', 'Maryhill', 'Woonsocket', 'Springhill', 'Annawan', 'Albertville', 'Anaheim', 'Roxboro', 'Beach', 'Bowlus', 'Convoy', 'Hillsdale', 'Kerrtown', 'Mifflintown', 'Andover', 'Brevort']

@ex.named_config
def buildings1():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:1],
            }
        }
    }

@ex.named_config
def buildings2():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:2],
            }
        }
    }

@ex.named_config
def buildings4():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:4],
            }
        }
    }

@ex.named_config
def buildings8():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:8],
            }
        }
    }

@ex.named_config
def buildings16():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:16],
            }
        }
    }

@ex.named_config
def buildings32():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings[:32],
            }
        }
    }

@ex.named_config
def buildings72():
    cfg = {
        'env': {
            'env_specific_kwargs': {
                'train_scenes': all_buildings,
            }
        }
    }


####################################
# Train on shorter distances
####################################
@ex.named_config
def short3m():
    uuid = 'habitat_planning_3m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 3
            }}}

@ex.named_config
def short5m():
    uuid = 'habitat_planning_5m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 5
            }}}

@ex.named_config
def short7m():
    uuid = 'habitat_planning_7m'
    cfg = {}
    cfg['env'] = {
        'env_name': 'Habitat_PointNav',    # Environment to use
        "env_specific_kwargs": {
            'scenario_kwargs': {           # specific to the scenario - pointnav or exploration
                'max_geodesic_dist': 7
            }}}

####################################
# Decoding - Instead of using the latent code as our representation, directly use the decoding
# (UNet performs better than ResNet bottleneck)
# (Do not remember which one we used)
####################################
@ex.named_config
def taskonomy_decoding():
    ''' Implements an agent with some mid-level decoding.
        From the paper:
            Taskonomy: Disentangling Task Transfer Learning
            Amir R. Zamir, Alexander Sax*, William B. Shen*, Leonidas Guibas, Jitendra Malik, Silvio Savarese.
            2018
        Viable feature options are:
            []
    '''
    uuid = 'habitat_taskonomy_decoding'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'AtariNet',
        'perception_network_kwargs': {
            'n_map_channels': 3,
            'use_target': True,
        },
    }
    cfg['env'] = {
        'env_specific_kwargs': {
            'target_dim': 9,  # Taskonomy reps: 16, scratch: 9, map_only: 1
        },
        'transform_fn_pre_aggregation': """
                TransformFactory.independent(
                {
                    'rgb_filled': rescale_centercrop_resize((3,256,256)),
                },
                keep_unnamed=True)
            """.translate(remove_whitespace),
        'transform_fn_post_aggregation': """
                TransformFactory.independent(
                {{
                    'rgb_filled':cross_modal_transform(TaskonomyNetwork(load_encoder_path='{encoder}', load_decoder_path='{decoder}').cuda()),
                    'taskonomy': identity_transform(),
                    'target':identity_transform(),
                    'map':map_pool_collated((3,84,84)),
                    'global_pos':identity_transform(),
                }},
                keep_unnamed=False)
            """.translate(remove_whitespace).format(
            encoder='/mnt/models/normal_encoder.dat',
            decoder='/mnt/models/normal_decoder.dat'),
    }

@ex.named_config
def cfg_taskonomy_decoding():
    # Transforms RGB images to the output from one of the networks from the paper:
    #    Taskonomy: Disentangling Task Transfer Learning (Zamir et al. '18)
    cfg = {}
    cfg['env'] = {
        'collate_env_obs': False,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform(TaskonomyNetwork(load_encoder_path='/mnt/models/normal_encoder.dat',
                                                            load_decoder_path='/mnt/models/normal_decoder.dat').cuda().eval(),
                                                            (3,{image_dim}, {image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)
            """.format(
            encoder_type=cfg['learner']['encoder_type'],
            taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
            image_dim=84),
    }
    cfg['learner'] = { 'perception_network': 'scratch' }

@ex.named_config
def cfg_taskonomy_decoding_collate():
    # Transforms RGB images to the output from one of the networks from the paper:
    #    Taskonomy: Disentangling Task Transfer Learning (Zamir et al. '18)
    # Collated versions are slightly faster, at the cost of using slightly more GPU memory

    image_dim = 84
    cfg = {}
    cfg['learner'] = { 'perception_network': 'scratch',
                       'taskonomy_encoder': '/mnt/models/normal_encoder.dat',
                       'encoder_type': 'taskonomy'
                       }
    cfg['env'] = {
        'collate_env_obs': True,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform_collated(TaskonomyNetwork(
                    load_encoder_path='/mnt/models/normal_encoder.dat',
                    load_decoder_path='/mnt/models/normal_decoder.dat').cuda().eval(),
                    (3,{image_dim},{image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)""".format(
            encoder_type=cfg['learner']['encoder_type'],
            taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
            image_dim=image_dim),
    }

@ex.named_config
def cfg_unet_decoding():
    # Transforms RGB images to the output from a trained UNet

    image_dim = 84
    cfg = {}
    cfg['env'] = {
        'collate_env_obs': False,
        'transform_fn': """
            TransformFactory.independent(
            {{
                'rgb_filled':cross_modal_transform(load_from_file(
                        UNet(),
                        '/mnt/logdir/homoscedastic_normal_regression-checkpoints-ckpt-4.dat').cuda().eval(),
                        (3,{image_dim}, {image_dim})),
                'target':identity_transform()
            }},
            keep_unnamed=False)""".format(
            encoder_type=cfg['learner']['encoder_type'],
            taskonomy_encoder=cfg['learner']['taskonomy_encoder'],
            image_dim=image_dim),
    }
    cfg['learner'] = { 'perception_network': 'scratch' }

#########################
# Regularization
#########################
# The MemoryFrameStack wrapper over the perception network will hold on to the `attrs_to_remember` in the cache
# The policy (e.g. PolicyWithBase) computes the intrinsic losses (with the remembered attributes)
# PPOReplay wrapper uses the intrinsic loss to compute and backward the final loss
# For transfer, the network is passed into the cache and use net.transfer
# For decoder, a separate network is used to map the encodings to a perceptual space. This is agnostic of the training procedure.
# We load it into the policy (e.g. PolicyWithBase) and run it forward as needed.
@ex.named_config
def treg_l1():
    uuid = 'habitat_regularization_transfer_l1'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'attrs_to_remember': ['base_encoding', 'transfered_encoding']   # things to remember for supp. losses
            }
        },
        'loss_kwargs': {
            'intrinsic_loss_coefs': [0.1],
            'intrinsic_loss_types': ['transfer_l1'],
        }
    }

@ex.named_config
def treg_l2():
    uuid = 'habitat_regularization_transfer_l2'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'attrs_to_remember': ['base_encoding', 'transfered_encoding']   # things to remember for supp. losses
            }
        },
        'loss_kwargs': {
            'intrinsic_loss_coefs': [0.1],
            'intrinsic_loss_types': ['transfer_l2'],
        }
    }

@ex.named_config
def dreg_t():
    # theoretically we can use any network with the proper input space but we specify decoder to know number of output channels
    uuid = 'habitat_regularization_perceptual_transfer'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'attrs_to_remember': ['base_encoding', 'transfered_encoding']   # things to remember for supp. losses
            }
        },
        'loss_kwargs': {
            'intrinsic_loss_coefs': [0.1],
            'intrinsic_loss_types': ['perceptual_transfer'],
            'decoder_path': '/mnt/models/curvature_decoder.dat'
        }
    }

@ex.named_config
def dreg():
    # theoretically we can use any network with the proper input space but we specify decoder to know number of output channels
    uuid = 'habitat_regularization_perceptual'
    cfg = {}
    cfg['learner'] = {
        'perception_network': 'RLSidetuneWrapper',
        'perception_network_kwargs': {
            'extra_kwargs': {
                'attrs_to_remember': ['base_encoding', 'merged_encoding', 'side_output']   # things to remember for supp. losses/visualization
            }
        },
        'loss_kwargs': {
            'loss_fn': 'F.mse_loss',  # or F.l1_loss
            'intrinsic_loss_coefs': [0.1],
            'intrinsic_loss_types': ['perceptual'],
            'decoder_path': '/mnt/models/curvature_decoder.dat'
        }
    }

####################################
# Other
####################################
@ex.named_config
def unorm_t_only():
    # This is to allow us apply midtune on two towers
    cfg = {}
    cfg['learner'] = {
        'perception_network_kwargs': {
            'extra_kwargs': {
                'normalize_taskonomy': False
            }
        }
    }

@ex.named_config
def features_double():
    # This is to allow us apply midtune on two towers
    cfg = {}
    cfg['learner'] = {
        'perception_network_kwargs': {
            'extra_kwargs': {
                'features_double': True
            }
        }
    }

@ex.named_config
def transform_rgb256():
    cfg = {}
    cfg['env'] = {
        'transform_fn_pre_aggregation_fn': 'TransformFactory.independent',
        'transform_fn_pre_aggregation_kwargs': {
            'names_to_transforms': {
                'rgb_filled':'rescale_centercrop_resize((3,256,256))',
            },
        },
    }

