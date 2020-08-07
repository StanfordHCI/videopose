import os
import cv2
import sys
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
#from src.system.interface import AnnotatorInterface
import ipdb;pdb=ipdb.set_trace
import time
import pickle

from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
bboxModel, poseModel = getTwoModel()
interface2D = getKptsFromImage
from tools.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()
from tools.utils import interface as VideoPoseInterface
interface3D = VideoPoseInterface
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img


sys.path.append('./joints_detectors')
import posenet

import torch


def save_pose(pose):
    print("sssss", len(pose[0]))
    with open('temp.pkl','wb') as f:
        pickle.dump(pose,f)

def get_2d_pose_1(frame):
    try:
        joint2D = interface2D(bboxModel, poseModel, frame)  
        #print('HrNet comsume {:0.3f} s'.format(time.time() - t0))
    except Exception as e:
        print(e)
    return joint2D
'''
def get_2d_pose_2(sess, input_image, output_stride, model_outputs):
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image}
    )

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=3,
        min_pose_score=0.15)

    #keypoint_coords *= output_scale
    joint2 = np.zeros(keypoint_coords[0].shape)
    joint2[:,0] = keypoint_coords[0][:,1].copy()
    joint2[:,1] = keypoint_coords[0][:,0].copy()
    return joint2
'''

def get_2d_pose_torch(input_image, output_stride, model):
    with torch.no_grad():
        input_image = torch.Tensor(input_image).cuda()

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)

    #keypoint_coords *= output_scale
    joint2 = np.zeros(keypoint_coords[0].shape)
    joint2[:,0] = keypoint_coords[0][:,1].copy()
    joint2[:,1] = keypoint_coords[0][:,0].copy()
    return joint2

def main(VideoName, model_layes):
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride
    cap, cap_length = videoInfo(VideoName)
    kpt2Ds = []
    pose_3d = []
    #annotator = AnnotatorInterface.build(max_persons=1)
    for i in range(cap_length): #tqdm(range(cap_length)):
        #if i < 90: continue
        if i > 300: break
        _, frame = cap.read()
        input_image, display_image, output_scale = posenet.process_input(frame, 1/3.0, output_stride)
        frame, W, H = resize_img(frame)


        time0 = time.time() 
        joint2D = get_2d_pose_torch(input_image, output_stride, model)#get_2d_pose_1(frame)
        time1 = time.time()

        #print(output_scale)
        #joint2 = 0#get_2d_pose_2(sess, input_image, output_stride, model_outputs)
        #persons = annotator.update(frame)
        #poses_2d = [p['pose_2d'].get_joints() for p in persons]
        #joint2D2 = poses_2d[0]
        #print(joint2D)
        #joint2D = np.vstack((joint2D[0:1, :], joint2D[5:17, :]))
        #print(joint2D3.shape)
        time2 = time.time()
        #raise KeyboardInterrupt
        if i == 0:
            for _ in range(30):
                kpt2Ds.append(joint2D)
        else:
            kpt2Ds.append(joint2D)
            kpt2Ds.pop(0)
        
        #if i < 15:
        #    kpt2Ds.append(joint2D)
        #    kpt2Ds.pop(0)
        #else:
        #    kpt2Ds.append(joint2D)
    
        #print(len(kpt2Ds))
        joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
        joint3D_item = joint3D[-1] #(17, 3)
        time3 = time.time()
        pose_3d.append((joint3D_item, joint2D))
        print(time1 - time0, time2-time1, time3-time2, time3- time1)
        #draw_3Dimg(joint3D_item, frame, display=1, kpt2D=joint2D)
    save_pose(pose_3d)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-video", "--video_input", help="input video file name", default="/home/xyliu/Videos/sports/dance.mp4")
    parser.add_argument('--model', type=int, default=101)
    args = parser.parse_args()
    VideoName = args.video_input
    print('Input Video Name is ', VideoName)
    main(VideoName, args.model)

