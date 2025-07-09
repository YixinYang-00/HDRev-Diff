'''
edit: hylz
Slightly modified from code of Haram Kim.
'''

"""
edit: Haram Kim
email: rlgkfka614@gmail.com
github: https://github.com/haram-kim
homepage: https://haram-kim.github.io/
"""

from cv2 import CV_16SC2, CV_32FC1
import numpy as np
import cv2
import yaml
from numpy import linalg as la


class Camera:
    K_rect = np.eye(3)
    K_rect_inv = np.eye(3)
    camera_num = 0
    width = 0
    height = 0
    resolution = (height, width)
    baseline = 0.0        
    def set_baseline(baseline, baseline_gt=-1):
        Camera.baseline = baseline
        if baseline_gt == -1:
            Camera.baseline_gt = baseline
        else:
            Camera.baseline_gt = baseline_gt

class CameraDSEC(Camera):
    def __init__(self, params, rect_mat=np.eye(3)):
        # dsec dataset
        # cam0: event left
        # cam1: frame left
        # cam2: frame right
        # cam3: event right
        
        # resolution
        self.imsize = np.array(params['resolution'])
        self.resolution = np.array([self.imsize[1], self.imsize[0]])
        self.height = self.imsize[1]
        self.width = self.imsize[0]
        
        # intrinsics 
        self.fx = params['camera_matrix'][0]
        self.fy = params['camera_matrix'][1]
        self.cx = params['camera_matrix'][2]
        self.cy = params['camera_matrix'][3]
        self.K = np.array([[self.fx, 0.0, self.cx],[0.0, self.fy, self.cy],[0.0, 0.0, 1.0]])
        # distortions
        try:
            self.dist_coeff = np.array(params['distortion_coeffs'])
        except:
            self.dist_coeff = np.zeros((4, 1))
        self.cv_dist_coeff = np.array([self.dist_coeff[0], self.dist_coeff[1], 0.0, 0.0, self.dist_coeff[2]], dtype=object)
        # rectification
        self.rect_mat = rect_mat

        CameraDSEC.camera_num += 1

        # generate rect map
        self.generate_undist_map()

    @staticmethod
    def set_cam_rect(params):
        fx = params['camera_matrix'][0]
        fy = params['camera_matrix'][1]
        cx = params['camera_matrix'][2]
        cy = params['camera_matrix'][3]
        Camera.width = params['resolution'][0]
        Camera.height = params['resolution'][1]
        Camera.K_rect = np.array([[fx, 0.0, cx],[0.0, fy, cy],[0.0, 0.0, 1.0]])
        Camera.K_rect_inv = np.linalg.inv(Camera.K_rect)
        Camera.resolution = (Camera.height, Camera.width)

    def generate_undist_map(self):
        self.undist_map1, self.undist_map2 = cv2.initUndistortRectifyMap(self.K, self.dist_coeff, None, self.K, self.imsize, CV_16SC2)
        self.rect_map1, self.rect_map2 = cv2.initUndistortRectifyMap(self.K, self.dist_coeff, self.rect_mat, Camera.K_rect, self.imsize, CV_16SC2)

    def rectify(self, image, interp = cv2.INTER_LINEAR):
        image_rect = cv2.remap(image.astype(np.float32), self.rect_map1, self.rect_map2, interpolation=interp)[:Camera.height, :Camera.width]
        return image_rect

    def undist(self, image):
        image_undist = cv2.remap(image, self.undist_map1, self.undist_map2, interpolation=cv2.INTER_LINEAR)
        return image_undist

    def rectify_events_default(self, events):
        event_reshaped = np.reshape(events, (-1,1,4))
        uv_rect = cv2.undistortPoints(event_reshaped[:,:,0:2], self.K, self.dist_coeff, self.rect_mat, Camera.K_rect)
        events_rect = np.concatenate((np.reshape(uv_rect, (-1, 2)), events[:,2:4]), axis = 1)
        return events_rect

    def rectify_events(self, events):
        event_reshaped = np.reshape(events, (-1,1,4))
        uv_rect = cv2.undistortPointsIter(event_reshaped[:,:,0:2], self.K, self.dist_coeff, self.rect_mat, Camera.K_rect, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.01))
        events_rect = np.concatenate((np.reshape(uv_rect, (-1, 2)), events[:,2:4]), axis = 1)
        return events_rect

    def project_events(self, events):
        image = np.zeros((self.height, self.width))
        for index, value in enumerate(events):
            u = int(value[0])
            v = int(value[1])
            if ((u >= 0) & (v >= 0) & (u < self.width) & (v < self.height)):                
                image[v, u] += 2*value[3] - 1

        return image



def load_calib_data(calibpath):
    # # frame: left cam0
    # # event right cam1
    calib_data = []
    with open(calibpath, "r") as stream:
        try:
            calib_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    rect_mat = np.eye(3)
    T_32 = np.asarray(calib_data['extrinsics']['T_32'])
    T_21 = np.asarray(calib_data['extrinsics']['T_21'])
    T_rect1 = np.eye(4)
    T_rect1[:3,:3] = np.asarray(calib_data['extrinsics']['R_rect1'])
    T_rect31 = np.matmul(np.matmul(np.linalg.inv(T_32), np.linalg.inv(T_21)), T_rect1)

    # Rectify right event (cam3) to right image (cam2)
    rect_mat = np.eye(3)
    T_32 = np.asarray(calib_data['extrinsics']['T_32'])
    T_rect2 = np.eye(4)
    T_rect2[:3,:3] = np.asarray(calib_data['extrinsics']['R_rect2'])
    T_rect32 = np.matmul(np.linalg.inv(T_32), T_rect2)
    rect_mat = T_rect32[:3, :3]
    baseline = la.norm(T_rect31[:3, 3])
    baseline_gt = la.norm(T_21[:3, 3])

    # rect_mat = np.eye(3)

    CameraDSEC.set_baseline(baseline, baseline_gt)
    CameraDSEC.set_cam_rect(calib_data['intrinsics']['camRect3'])

    # Left image: "cam1" in yaml.
    cam_li = CameraDSEC(calib_data['intrinsics']['camRect1'])
    # Right event: "cam3" in yaml. It is rectified to right image.
    cam_re = CameraDSEC(calib_data['intrinsics']['cam3'], rect_mat)        

    # Right image: "cam2" in yaml
    cam_ri = CameraDSEC(calib_data['intrinsics']['camRect2'])
    
    # Left event: "cam0" in yaml. It is rectified to left image (cam1).
    rect_map = np.eye(3)
    T_10 = np.asarray(calib_data['extrinsics']['T_10'])
    T_rect1 = np.eye(4)
    T_rect1[:3,:3] = np.asarray(calib_data['extrinsics']['R_rect1'])
    T_rect01 = np.matmul(T_10, T_rect1)
    rect_map = T_rect01[:3, :3]
    cam_le = CameraDSEC(calib_data['intrinsics']['cam0'], rect_map)
    
    
    return cam_le, cam_li, cam_ri, cam_re