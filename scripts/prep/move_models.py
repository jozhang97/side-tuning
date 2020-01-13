import subprocess

task_mapping = {
 'autoencoder': 'autoencoding',
 'colorization': 'colorization',
 'curvature': 'curvature',
 'denoise': 'denoising',
 'edge2d':'edge_texture',
 'edge3d': 'edge_occlusion',
 'ego_motion': 'egomotion', 
 'fix_pose': 'fixated_pose', 
 'jigsaw': 'jigsaw',
 'keypoint2d': 'keypoints2d',
 'keypoint3d': 'keypoints3d',
 'non_fixated_pose': 'nonfixated_pose',
 'point_match': 'point_matching', 
 'reshade': 'reshading',
 'rgb2depth': 'depth_zbuffer',
 'rgb2mist': 'depth_euclidean',
 'rgb2sfnorm': 'normal',
 'room_layout': 'room_layout',
 'segment25d': 'segment_unsup25d',
 'segment2d': 'segment_unsup2d',
 'segmentsemantic': 'segment_semantic',
 'class_1000': 'class_object',
 'class_places': 'class_scene',
 'inpainting_whole': 'inpainting',
 'vanishing_point': 'vanishing_point'
}


if __name__ == '__main__':
    for old, new in task_mapping.items():
        subprocess.call(f'mv taskonomy_data/{old}_encoder.dat taskonomy_data/{new}_encoder.dat', shell=True)
        subprocess.call(f'mv taskonomy_data/{old}_decoder.dat taskonomy_data/{new}_decoder.dat', shell=True)
