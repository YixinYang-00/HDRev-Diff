import numpy as np
import cv2
from matplotlib import pyplot as plt

def matlab_resize(img, scale=None, output_size=None, interploate='bilinear', antialiasing=False):
    ndim = len(np.shape(img))
    if ndim == 2:
        [sh, sw], c = img.shape, 1
        img = np.expand_dims(img, 2)
    elif ndim == 3:
        sh, sw, c = img.shape
    else:
        raise(ValueError("Do not support img size {}".format(img.shape)))
    img = np.transpose(img, [2, 0, 1])

    
    if scale and not output_size:
        # th, tw = int(sh * scale), int(sw * scale)
        th, tw = int(np.ceil(sh * scale)), int(np.ceil(sw * scale))
        th, tw = max(th, 1), max(tw, 1)
        scale_h = scale_w = scale
    elif not scale and output_size:
        th, tw = output_size
        scale_h, scale_w = th / sh, tw / sw
        if th < 1 or tw < 1:
            raise(ValueError("output size illegal {} {}".format(th, tw)))
    elif not scale and not output_size:
        raise(ValueError("Please specify the target size!"))

    

    ind_map_w = np.expand_dims(np.arange(tw), axis=0).repeat(th, axis=0)
    ind_map_h = np.expand_dims(np.arange(th), axis=1).repeat(tw, axis=1)

    ind_map_w = np.expand_dims(ind_map_w, 0)
    ind_map_h = np.expand_dims(ind_map_h, 0)
    ind_map_dst = np.concatenate([ind_map_h, ind_map_w], axis=0)

    ind_map_src_h = (ind_map_dst + 0.5)[0,:,:] * 1/scale_h - 0.5
    ind_map_src_h[ind_map_src_h<0] = 0
    ind_map_src_h[ind_map_src_h>sh-1] = sh - 1

    ind_map_src_w = (ind_map_dst + 0.5)[1,:,:] * 1/scale_w - 0.5
    ind_map_src_w[ind_map_src_w<0] = 0
    ind_map_src_w[ind_map_src_w>sw-1] = sw - 1

    ind_map_src = np.stack([ind_map_src_h, ind_map_src_w], axis=0)
    ind_map_adj4 = np.zeros([8, th, tw])

    ind_map_adj4[0:2, :, :] = np.floor(ind_map_src)
    ind_map_adj4[2, :, :] = np.floor(ind_map_src[0,:,:])
    ind_map_adj4[3, :, :] = np.ceil(ind_map_src[1,:,:])
    ind_map_adj4[4, :, :] = np.ceil(ind_map_src[0,:,:])
    ind_map_adj4[5, :, :] = np.floor(ind_map_src[1,:,:])
    ind_map_adj4[6:, :, :] = np.ceil(ind_map_src)
    ind_map_adj4 = ind_map_adj4.astype(np.int32)

    adj4_val = np.zeros([4*c, th, tw])
    adj4_val[0:1*c] = img[:, ind_map_adj4[0], ind_map_adj4[1]]
    adj4_val[1*c:2*c] = img[:, ind_map_adj4[2], ind_map_adj4[3]]
    adj4_val[2*c:3*c] = img[:, ind_map_adj4[4], ind_map_adj4[5]]
    adj4_val[3*c:] = img[:, ind_map_adj4[6], ind_map_adj4[7]]

    out = interploate_impl(ind_map_adj4, adj4_val, ind_map_src, c)


    return out

def interploate_impl(adj4_cor, adj4_val, src_cor, c):
    Ab = (adj4_cor[3, :, :] - adj4_cor[1, :, :])
    Ab[Ab==0] = 1
    A = (src_cor[1, :, :] - adj4_cor[1, :, :]) / Ab
    # A[np.isnan(A)] = 0

    Bb = (adj4_cor[4] - adj4_cor[0])
    Bb[Bb==0] = 1
    B = (src_cor[0] - adj4_cor[0]) / Bb
    # B[np.isnan(B)] = 0

    Q = (1-A-B+A*B) * adj4_val[0:1*c] + (A-A*B)*adj4_val[1*c:2*c] + (B - A*B)*adj4_val[2*c:3*c]+ A*B*adj4_val[3*c:]
    Q = np.squeeze(np.transpose(Q, [1, 2, 0])) 
    return Q

if __name__ == '__main__':

    # upsample 
    # a = np.arange(9).reshape(3,3)
    # print(a)
    # b = matlab_resize(a, output_size=[6, 6])
    # print(b)
    # b = matlab_resize(a, output_size=[5, 5])
    # print(b)
    # b = matlab_resize(a, output_size=[6, 7])
    # print(b)
    # b = matlab_resize(a, output_size=[7, 6])
    # print(b)

    # downsample 3 channle
    # a = np.arange(6*6*3).reshape(3, 6, 6)
    # print(a)
    # a = np.transpose(a, [1,2,0])
    # # b = matlab_resize(a, output_size=[3, 3])
    # # b = matlab_resize(a, output_size=[4, 4])
    # # b = matlab_resize(a, output_size=[3, 4])
    # b = matlab_resize(a, output_size=[4, 3])
    # print(b[:,:,0])
    # print(b[:,:,1])
    # print(b[:,:,2])


    # downsample 3 channle
    a = np.arange(6*11).reshape(11, 6)
    print(a)
    b = matlab_resize(a, output_size=[3, 5])
    # b = matlab_resize(a, output_size=[4, 4])
    # b = matlab_resize(a, output_size=[3, 4])
    # b = matlab_resize(a, output_size=[4, 3])
    print(b[:,:,0])


    img = cv2.imread('/home/SENSETIME/weiyunxuan1/Downloads/sr_052647/sr_052647_02_ev0_rs.png')
    img_rs = matlab_resize(img, output_size=[173, 225])
    cv2.imwrite('/home/SENSETIME/weiyunxuan1/Downloads/sr_052647/sr_052647_02_ev0_rs_0p5_test_python.png', img_rs)
