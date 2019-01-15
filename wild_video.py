import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time

args = parse_args()
print(args)

print('Loading 3D dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz' #  dataset 'h36m'
from common.h36m_dataset import Human36mDataset
dataset = Human36mDataset(dataset_path)


# according to output name,generate some format. we use detectron
from data.data_utils import suggest_metadata, suggest_pose_importer
import_func = suggest_pose_importer('detectron_pt_coco')
metadata = suggest_metadata('detectron_pt_coco')


print('Loading 2D detections keypoints ...')
# handle 2d keypoints format
if args.input_npz:
    file_input = args.input_npz
else:
    file_input = '../Detectron/lxy-2d-detectron.npz'
keypoints = import_func(file_input)

keypoints_symmetry = metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

# normlization keypoints
# 假设use the camera parameter
cam = dataset.cameras()['S1'][0]
keypoints[..., :2] = normalize_screen_coordinates(keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])


model_pos = TemporalModel(17, 3, 17,filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
if torch.cuda.is_available():
    model_pos = model_pos.cuda()
# load trained model
chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)# 把loc映射到storage
model_pos.load_state_dict(checkpoint['model_pos'])

#  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
receptive_field = model_pos.receptive_field()
pad = (receptive_field - 1) // 2 # Padding on each side
causal_shift = 0

def evaluate(test_generator, action=None, return_predictions=False):
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


print('Rendering...')
input_keypoints = keypoints.copy()

gen = UnchunkedGenerator(None, None, [input_keypoints],
                            pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
prediction = evaluate(gen, return_predictions=True)


# If the ground truth is not available, take the camera extrinsic params from a random subject.
# They are almost the same, and anyway, we only need this for visualization purposes.
for subject in dataset.cameras():
    if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
        rot = dataset.cameras()[subject][args.viz_camera]['orientation']
        break
prediction = camera_to_world(prediction, R=rot, t=0)

# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])

anim_output = {'Reconstruction': prediction}

input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

from common.visualization import render_animation
render_animation(input_keypoints, anim_output,
                    dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                    limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                    input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                    input_video_skip=args.viz_skip)

