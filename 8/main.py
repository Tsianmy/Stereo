import cv2 as cv
import numpy as np
import os
from homography import get_homography
from intrinsics import get_intrinsics_param
from extrinsics import get_extrinsics_param
from distortion import get_distortion
from refine_all import refinall_all_param
 
 
def calibrate():
    print('get_homography\n')
    H = get_homography(pic_points, real_points_x_y)

    print('get_intrinsics\n')
    intrinsics_param = get_intrinsics_param(H)

    print('get_extrinsics\n')
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)

    print('get_distortion\n')
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)

    #print('old')
    #print(H, '\n', intrinsics_param, '\n', k, '\n', 0, '\n')
    
    print('optimize\n')
    [new_intrinsics_param, new_k, new_extrinsics_param]  = refinall_all_param(intrinsics_param,
                                                            k, extrinsics_param, real_points, pic_points)
 
    return new_intrinsics_param, new_k, new_extrinsics_param

def undistort(file_dir, intrinsics_param, k):
    k = np.concatenate((k,[0, 0]))
    print(k)
    pic_name = os.listdir(file_dir)
    out_path = '../8output/'
    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)
        print('distort', pic)
        
        shp = pic_data.shape
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(intrinsics_param,k,(shp[0],shp[1]),0,(shp[0],shp[1]))
        pic_undist = cv.undistort(pic_data, intrinsics_param, k, None, newcameramtx)

        cv.imwrite(out_path + 'r' + pic, pic_undist)
 
 
if __name__ == "__main__":
    file_dir = r'../left'
    pic_name = os.listdir(file_dir)
 
    cross_corners = [9, 6]
    real_coor = np.zeros((cross_corners[0] * cross_corners[1], 3), np.float32)
    real_coor[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    
    real_points = []
    real_points_x_y = []
    pic_points = []

    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path)
        print('read', pic)
 
        succ, pic_coor = cv.findChessboardCorners(pic_data, (cross_corners[0], cross_corners[1]), None)
 
        if succ:
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)
 
            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])
    
    print('\ncalibrate\n')
    intrinsics_param, k, extrinsics_param = calibrate()
    
    print("intrinsics_parm:\t", intrinsics_param)
    print("distortionk:\t", k)
    #print("extrinsics_parm:\t", extrinsics_param)

    ifud = False
    if ifud:
        undistort(file_dir, intrinsics_param, k)
    
