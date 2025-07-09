import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
import torch
if __name__ == '__main__':
    from matlab_resize_antiFalse import matlab_resize
    from predefined_args import predefined_args
else :
    from .matlab_resize_antiFalse import matlab_resize
    from .predefined_args import predefined_args


eps = 1e-10
def local_int_align_satDark_hist_norm_pytorch(indata, ref, args=predefined_args()):  
    indata = indata.float()
    ref = ref.float()
    b, c, ori_h, ori_w = indata.shape
    
    patch_h = args.patch_h
    patch_w = args.patch_w

    anchor_dark0 = args.anchor_align_dark0
    anchor_dark1 = args.anchor_align_dark1
    anchor_sat0 = args.anchor_align_sat0
    anchor_sat1 = args.anchor_align_sat1
    align_thresh = args.anchor_align_thresh

    guass_kernel = args.guass_kernel
    guass_sigma = args.guass_sigma
    
    ds = args.local_int_align_ds
    indata_ds = F.interpolate(indata, scale_factor=1/ds, mode='bilinear', align_corners=False)
    ref_ds = F.interpolate(ref, scale_factor=1/ds, mode='bilinear', align_corners=False)

    meanMax_blend = args.meanMax_blend
    in_h, in_w = indata_ds.shape[-2], indata_ds.shape[-1]

    align_in_anchor = torch.zeros((c, 6, patch_h * 2 - 1, patch_w * 2 - 1), device=indata.device)
    align_ref_anchor = torch.zeros((c, 6, patch_h * 2 - 1, patch_w * 2 - 1), device=indata.device)

    for ci in range(c):
        for h in range(patch_h * 2 - 1):
            h_start = round(in_h / patch_h / 2 * h)
            h_end = round(in_h / patch_h / 2 * (h + 2))

            for w in range(patch_w * 2 - 1):
                w_start = round(in_w / patch_w / 2 * w)
                w_end = round(in_w / patch_w / 2 * (w + 2))

                patch_in = indata_ds[0, ci, h_start:h_end, w_start:w_end]
                patch_ref = ref_ds[0, ci, h_start:h_end, w_start:w_end]

                patch_in_sort = torch.sort(patch_in.flatten())[0]
                patch_in_anchor = torch.tensor([
                    patch_in_sort[0],
                    patch_in_sort[int(len(patch_in_sort) * 0.20)],
                    patch_in_sort[int(len(patch_in_sort) * 0.40)],
                    patch_in_sort[int(len(patch_in_sort) * 0.60)],
                    patch_in_sort[int(len(patch_in_sort) * 0.80)],
                    patch_in_sort[-1],
                ], device=indata.device)

                patch_ref_sort = torch.sort(patch_ref.flatten())[0]
                patch_ref_anchor = torch.tensor([
                    patch_ref_sort[0],
                    patch_ref_sort[int(len(patch_ref_sort) * 0.20)],
                    patch_ref_sort[int(len(patch_ref_sort) * 0.40)],
                    patch_ref_sort[int(len(patch_ref_sort) * 0.60)],
                    patch_ref_sort[int(len(patch_ref_sort) * 0.80)],
                    patch_ref_sort[-1],
                ], device=indata.device)

                a = 1 / (anchor_sat0 - anchor_sat1)
                b = anchor_sat1 / (anchor_sat1 - anchor_sat0)
                patch_ref_nosat_anchor_conf = a * patch_ref_anchor + b
                patch_ref_nosat_anchor_conf = torch.clamp(patch_ref_nosat_anchor_conf, 0, 1)

                a = 1 / (anchor_dark1 - anchor_dark0)
                b = anchor_dark0 / (anchor_dark0 - anchor_dark1)
                patch_ref_nodark_anchor_conf = a * patch_ref_anchor + b
                patch_ref_nodark_anchor_conf = torch.clamp(patch_ref_nodark_anchor_conf, 0, 1)

                anchor_conf = patch_ref_nosat_anchor_conf + patch_ref_nodark_anchor_conf - 1
                patch_target_anchor = patch_ref_anchor * anchor_conf + patch_in_anchor * (1 - anchor_conf)

                as_anchor_idx, ds_anchor_idx = 0, 5
                for i in range(6):
                    if i > 0 and anchor_conf[i] > anchor_conf[i-1]:
                        as_anchor_idx = i
                    if i < 5 and anchor_conf[5-i] < anchor_conf[4-i]:
                        ds_anchor_idx = 4 - i

                dark_min = patch_ref_anchor[0]
                norm_bot, norm_top = min(dark_min, patch_target_anchor[0]), patch_target_anchor[as_anchor_idx]
                norm_anchor = (patch_target_anchor[:as_anchor_idx+1] - norm_bot + eps) / (norm_top - norm_bot + eps) * (norm_top - dark_min) + dark_min
                patch_target_anchor[:as_anchor_idx+1] = norm_anchor

                norm_bot, norm_top = patch_target_anchor[ds_anchor_idx], max(1, patch_target_anchor[-1])
                norm_anchor = (patch_target_anchor[ds_anchor_idx+1:] - norm_bot + eps) / (norm_top - norm_bot + eps) * (1 - norm_bot) + norm_bot
                patch_target_anchor[ds_anchor_idx+1:] = norm_anchor

                align_in_anchor[ci, :, h, w] = patch_in_anchor    
                align_ref_anchor[ci, :, h, w] = patch_target_anchor

    # Apply Gaussian filter
    filter = cv2.getGaussianKernel(guass_kernel, guass_sigma)
    kernel = torch.tensor(np.dot(filter, np.transpose(filter)), device=indata.device, dtype=torch.float32)
    kernel_size = kernel.shape[-1]

    c, _, h_before, w_before = align_in_anchor.shape
    align_in_anchor = F.pad(align_in_anchor, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
    align_ref_anchor = F.pad(align_ref_anchor, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
    
    c, _, h, w = align_in_anchor.shape
    align_in_anchor = F.conv2d(align_in_anchor.reshape(c * 6, 1, h, w), kernel.unsqueeze(0).unsqueeze(0), padding='valid').reshape(c, 6, h_before, w_before)
    align_ref_anchor = F.conv2d(align_ref_anchor.reshape(c * 6, 1, h, w), kernel.unsqueeze(0).unsqueeze(0), padding='valid').reshape(c, 6, h_before, w_before)

    align_in_anchor = F.interpolate(align_in_anchor, size=[ori_h, ori_w], mode='bilinear', align_corners=False)
    align_ref_anchor = F.interpolate(align_ref_anchor, size=[ori_h, ori_w], mode='bilinear', align_corners=False)
    # print(align_in_anchor.shape)
    out = torch.zeros_like(indata)
    for j in range(c):
        for i in range(5):
            x1, x2 = align_in_anchor[j, i, ...], align_in_anchor[j, i+1, ...]
            y1, y2 = align_ref_anchor[j, i, ...], align_ref_anchor[j, i+1, ...]
            out_tmp = (indata[:, j] - x1) * (y2 - y1 + eps) / (x2 - x1 + eps) + y1

            if i == 0:
                mask = indata[:, j] < x2
            elif i == 4:
                mask = indata[:, j] >= x1
            else:
                mask = torch.logical_and(indata[:, j] >= x1, indata[:, j] < x2)

            out[:, j][mask] = out_tmp[mask]

    del align_in_anchor, align_ref_anchor
    return out


def local_int_align_satDark_hist_norm(indata, ref, args=predefined_args()):  

    # if not os.path.exists(os.path.join(args.debug_savepath, 'patches')):
    #      os.mkdir(os.path.join(args.debug_savepath, 'patches'))

    indata = indata.astype(np.float32)
    ref = ref.astype(np.float32)
    ori_h, ori_w = indata.shape
    
    patch_h  =  args.patch_h
    patch_w  =  args.patch_w

    anchor_dark0 = args.anchor_align_dark0
    anchor_dark1 = args.anchor_align_dark1
    anchor_sat0  =  args.anchor_align_sat0
    anchor_sat1  =  args.anchor_align_sat1
    align_thresh  =  args.anchor_align_thresh

    guass_kernel = args.guass_kernel
    guass_sigma = args.guass_sigma
    
    ds = args.local_int_align_ds
    indata_ds = matlab_resize(indata, scale=1/ds)
    ref_ds = matlab_resize(ref, scale=1/ds)

    meanMax_blend = args.meanMax_blend
    in_h, in_w  =  indata_ds.shape

    align_in_anchor  =  np.zeros((patch_h*2-1, patch_w*2-1, 6))
    align_ref_anchor  =  np.zeros((patch_h*2-1, patch_w*2-1, 6))

    for h in range(patch_h*2-1):
        h_start  =  round(in_h/patch_h/2*h)
        h_end  =  round(in_h/patch_h/2*(h+2))

        for w in range(patch_w*2-1):
            w_start  =  round(in_w/patch_w/2*w)
            w_end  =  round(in_w/patch_w/2*(w+2))

            
            patch_in  =  indata_ds[h_start:h_end, w_start:w_end]
            patch_ref  =  ref_ds[h_start:h_end, w_start:w_end]
            # tmplen = patch_in.shape[0] * patch_in.shape[1]

            # a = 1 / (anchor_sat0 - anchor_sat1)
            # b = anchor_sat1 / (anchor_sat1 - anchor_sat0)
            # patch_ref_nosat = a*patch_ref+b
            # patch_ref_nosat[patch_ref_nosat<0] = 0
            # patch_ref_nosat[patch_ref_nosat>1] = 1

            # a = 1 / (anchor_dark1 - anchor_dark0)
            # b = anchor_dark0 / (anchor_dark0 - anchor_dark1)
            # patch_ref_nodark = a*patch_ref+b
            # patch_ref_nodark[patch_ref_nodark<0] = 0
            # patch_ref_nodark[patch_ref_nodark>1] = 1


            # patch_conf  =  np.sum(patch_ref_nosat.flatten() * patch_ref_nodark.flatten()) / tmplen
            
            # patch_conf  =  patch_conf ** 3
            
            # patch_conf  =  max(align_thresh, patch_conf)
            

            #--------------------------- hist version --------------------------------#
            patch_in_sort = np.sort(patch_in.flatten())
            patch_in_anchor = np.array([
                patch_in_sort[0],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.20))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.40))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.60))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.80))],
                patch_in_sort[-1],
            ])

            patch_ref_sort = np.sort(patch_ref.flatten())
            patch_ref_anchor = np.array([
                patch_ref_sort[0],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.20))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.40))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.60))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.80))],
                patch_ref_sort[-1],
            ])


            a = 1 / (anchor_sat0 - anchor_sat1)
            b = anchor_sat1 / (anchor_sat1 - anchor_sat0)
            patch_ref_nosat_anchor_conf = a*patch_ref_anchor+b
            patch_ref_nosat_anchor_conf = np.clip(patch_ref_nosat_anchor_conf, 0, 1)

            a = 1 / (anchor_dark1 - anchor_dark0)
            b = anchor_dark0 / (anchor_dark0 - anchor_dark1)
            patch_ref_nodark_anchor_conf = a*patch_ref_anchor+b
            patch_ref_nodark_anchor_conf = np.clip(patch_ref_nodark_anchor_conf, 0, 1)

            anchor_conf = patch_ref_nosat_anchor_conf + patch_ref_nodark_anchor_conf - 1
            # anchor_conf = patch_conf * anchor_conf
            patch_target_anchor = patch_ref_anchor * anchor_conf + patch_in_anchor * (1 - anchor_conf)

            as_anchor_idx, ds_anchor_idx = 0, 5
            for i in range(6):
                if i > 0 and anchor_conf[i] > anchor_conf[i-1]:
                    as_anchor_idx = i
                if i < 5 and anchor_conf[5-i] < anchor_conf[4-i]:
                    ds_anchor_idx = 4 - i


            
            dark_min = patch_ref_anchor[0]
            norm_bot, norm_top = min(dark_min, patch_target_anchor[0]), patch_target_anchor[as_anchor_idx]
            norm_anchor = (patch_target_anchor[:as_anchor_idx+1] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (norm_top - dark_min) + dark_min
            # norm_anchor = (patch_target_anchor[:as_anchor_idx+1] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (norm_top - 0) + 0.0
            patch_target_anchor[:as_anchor_idx+1] = norm_anchor


            
            norm_bot, norm_top = patch_target_anchor[ds_anchor_idx], max(1, patch_target_anchor[-1])
            norm_anchor = (patch_target_anchor[ds_anchor_idx+1:] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (1 - norm_bot) + norm_bot
            patch_target_anchor[ds_anchor_idx+1:] = norm_anchor # warning, in C++ [idx+1] may exceed boundary


            align_in_anchor[h,w,:] = patch_in_anchor    
            align_ref_anchor[h,w,:] = patch_target_anchor
            #--------------------------- hist version --------------------------------#


    filter = cv2.getGaussianKernel(guass_kernel, guass_sigma)
    kernel = np.dot(filter, np.transpose(filter))

    align_in_anchor = cv2.filter2D(align_in_anchor, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    align_ref_anchor = cv2.filter2D(align_ref_anchor, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    align_in_anchor  =  matlab_resize(align_in_anchor, output_size=[ori_h, ori_w])
    align_ref_anchor  =  matlab_resize(align_ref_anchor, output_size=[ori_h, ori_w])
    # print(align_in_anchor.shape, align_ref_anchor.shape)


    # print('LIA: norm')
    out = np.zeros_like(indata)
    for i in range(5):
        x1, x2 = align_in_anchor[:,:,i], align_in_anchor[:,:,i+1]
        y1, y2 = align_ref_anchor[:,:,i], align_ref_anchor[:,:,i+1]
        out_tmp = (indata - x1) * (y2 - y1 + eps) / (x2 - x1 + eps) + y1
        # cv2.imwrite(os.path.join(args.debug_savepath, 'lin_outTmp_{}.png'.format(i)), out_tmp*255)

        if i == 0:
            mask = indata < x2
        elif i == 4:
            mask = indata >= x1
        else:
            mask = np.logical_and(indata >= x1, indata < x2)
        # cv2.imwrite(os.path.join(args.debug_savepath, 'lin_outMask_{}.png'.format(i)), mask*255)
        out[mask] = out_tmp[mask]

    del align_in_anchor, align_ref_anchor   
    return out


def local_int_align_satDark_hist_norm_sat(indata, ref, args):  

    # if not os.path.exists(os.path.join(args.debug_savepath, 'patches')):
    #      os.mkdir(os.path.join(args.debug_savepath, 'patches'))

    indata = indata.astype(np.float32)
    ref = ref.astype(np.float32)
    ori_h, ori_w = indata.shape
    
    patch_h  =  args.patch_h
    patch_w  =  args.patch_w

    anchor_dark0 = args.anchor_align_dark0
    anchor_dark1 = args.anchor_align_dark1
    anchor_sat0  =  args.anchor_align_sat0
    anchor_sat1  =  args.anchor_align_sat1
    align_thresh  =  args.anchor_align_thresh

    guass_kernel = args.guass_kernel
    guass_sigma = args.guass_sigma
    
    ds = args.local_int_align_ds
    indata_ds = matlab_resize(indata, scale=1/ds)
    ref_ds = matlab_resize(ref, scale=1/ds)

    meanMax_blend = args.meanMax_blend
    in_h, in_w  =  indata_ds.shape

    align_in_anchor  =  np.zeros((patch_h*2-1, patch_w*2-1, 6))
    align_ref_anchor  =  np.zeros((patch_h*2-1, patch_w*2-1, 6))

    for h in range(patch_h*2-1):
        h_start  =  round(in_h/patch_h/2*h)
        h_end  =  round(in_h/patch_h/2*(h+2))

        for w in range(patch_w*2-1):
            w_start  =  round(in_w/patch_w/2*w)
            w_end  =  round(in_w/patch_w/2*(w+2))

            
            patch_in  =  indata_ds[h_start:h_end, w_start:w_end]
            patch_ref  =  ref_ds[h_start:h_end, w_start:w_end]
            tmplen = patch_in.shape[0] * patch_in.shape[1]

            a = 1 / (anchor_sat0 - anchor_sat1)
            b = anchor_sat1 / (anchor_sat1 - anchor_sat0)
            patch_ref_nosat = a*patch_ref+b
            patch_ref_nosat[patch_ref_nosat<0] = 0
            patch_ref_nosat[patch_ref_nosat>1] = 1

            a = 1 / (anchor_dark1 - anchor_dark0)
            b = anchor_dark0 / (anchor_dark0 - anchor_dark1)
            patch_ref_nodark = a*patch_ref+b
            patch_ref_nodark[patch_ref_nodark<0] = 0
            patch_ref_nodark[patch_ref_nodark>1] = 1


            patch_conf  =  np.sum(patch_ref_nosat.flatten() * patch_ref_nodark.flatten()) / tmplen
            
            patch_conf  =  patch_conf ** 3
            
            patch_conf  =  max(align_thresh, patch_conf)
            

            #--------------------------- hist version --------------------------------#
            patch_in_sort = np.sort(patch_in.flatten())
            patch_in_anchor = np.array([
                patch_in_sort[0],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.20))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.40))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.60))],
                patch_in_sort[int(np.floor(len(patch_in_sort)*0.80))],
                patch_in_sort[-1],
            ])

            patch_ref_sort = np.sort(patch_ref.flatten())
            patch_ref_anchor = np.array([
                patch_ref_sort[0],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.20))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.40))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.60))],
                patch_ref_sort[int(np.floor(len(patch_ref_sort)*0.80))],
                patch_ref_sort[-1],
            ])


            a = 1 / (anchor_sat0 - anchor_sat1)
            b = anchor_sat1 / (anchor_sat1 - anchor_sat0)
            patch_ref_nosat_anchor_conf = a*patch_ref_anchor+b
            patch_ref_nosat_anchor_conf = np.clip(patch_ref_nosat_anchor_conf, 0, 1)

            a = 1 / (anchor_dark1 - anchor_dark0)
            b = anchor_dark0 / (anchor_dark0 - anchor_dark1)
            patch_ref_nodark_anchor_conf = a*patch_ref_anchor+b
            patch_ref_nodark_anchor_conf = np.clip(patch_ref_nodark_anchor_conf, 0, 1)

            anchor_conf = patch_ref_nosat_anchor_conf + patch_ref_nodark_anchor_conf - 1
            anchor_conf = patch_conf * anchor_conf
            patch_target_anchor = patch_ref_anchor * patch_conf + patch_in_anchor * (1 - patch_conf)

            as_anchor_idx, ds_anchor_idx = 0, 5
            for i in range(6):
                if i > 0 and anchor_conf[i] > anchor_conf[i-1]:
                    as_anchor_idx = i
                if i < 5 and anchor_conf[5-i] < anchor_conf[4-i]:
                    ds_anchor_idx = 4 - i


            
            dark_min = patch_ref_anchor[0]
            norm_bot, norm_top = min(dark_min, patch_target_anchor[0]), patch_target_anchor[as_anchor_idx]
            norm_anchor = (patch_target_anchor[:as_anchor_idx+1] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (norm_top - dark_min) + dark_min
            # norm_anchor = (patch_target_anchor[:as_anchor_idx+1] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (norm_top - 0) + 0.0
            patch_target_anchor[:as_anchor_idx+1] = norm_anchor


            
            norm_bot, norm_top = patch_target_anchor[ds_anchor_idx], max(1, patch_target_anchor[-1])
            norm_anchor = (patch_target_anchor[ds_anchor_idx+1:] - norm_bot + 1e-10) / (norm_top - norm_bot + 1e-10) * (1 - norm_bot) + norm_bot
            patch_target_anchor[ds_anchor_idx+1:] = norm_anchor # warning, in C++ [idx+1] may exceed boundary


            align_in_anchor[h,w,:] = patch_in_anchor    
            align_ref_anchor[h,w,:] = patch_target_anchor
            #--------------------------- hist version --------------------------------#


    filter = cv2.getGaussianKernel(guass_kernel, guass_sigma)
    kernel = np.dot(filter, np.transpose(filter))

    align_in_anchor = cv2.filter2D(align_in_anchor, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    align_ref_anchor = cv2.filter2D(align_ref_anchor, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    align_in_anchor  =  matlab_resize(align_in_anchor, output_size=[ori_h, ori_w])
    align_ref_anchor  =  matlab_resize(align_ref_anchor, output_size=[ori_h, ori_w])


    # print('LIA: norm')
    out = np.zeros_like(indata)
    for i in range(5):
        x1, x2 = align_in_anchor[:,:,i], align_in_anchor[:,:,i+1]
        y1, y2 = align_ref_anchor[:,:,i], align_ref_anchor[:,:,i+1]
        out_tmp = (indata - x1) * (y2 - y1) / (x2 - x1) + y1
        # cv2.imwrite(os.path.join(args.debug_savepath, 'lin_outTmp_{}.png'.format(i)), out_tmp*255)

        if i == 0:
            mask = indata < x2
        elif i == 4:
            mask = indata >= x1
        else:
            mask = np.logical_and(indata >= x1, indata < x2)
        # cv2.imwrite(os.path.join(args.debug_savepath, 'lin_outMask_{}.png'.format(i)), mask*255)
        out[mask] = out_tmp[mask]

    del align_in_anchor, align_ref_anchor   
    return out


def local_int_align_satDark_hist_norm_RGB(src_rgb, dst_rgb):  
    src_rgb, dst_rgb = src_rgb.cpu(), dst_rgb.cpu()
    args = predefined_args()
    b, c = src_rgb.shape[:2]
    aligned_rgb = torch.zeros_like(dst_rgb).float()
    for i in range(b):
        # for j in range(c):
        #     aligned_rgb[i, j, :, :] = local_int_align_satDark_hist_norm(src_rgb[i, j, :, :], dst_rgb[i, j, :, :], args)

        aligned_rgb[i:i+1] = local_int_align_satDark_hist_norm_pytorch(src_rgb[i:i+1], dst_rgb[i:i+1], args)
            # aligned_rgb[:,:,i] = local_int_align_satDark_hist_norm_sat(src_rgb[:,:,i], dst_rgb[:,:,i], args)
    # return torch.from_numpy(aligned_rgb)
    return aligned_rgb


def PCA_decompose(img, pca):
    h, w, c = img.shape
    # 将图像展平为 (num_pixels, 3) 的矩阵
    img_flat = img.reshape(-1, 3)

    # 获取变换后的主成分
    transformed_img = pca.transform(img_flat)

    # 主成分中的第一个表示强度，后两个表示颜色成分
    intensity = transformed_img[:, 0]  # 强度
    color_components = transformed_img[:, 1:]  # 颜色成分
    return transformed_img.reshape(h, w, c)

def PCA_inverse(img, pca):
    h, w, c = img.shape
    # 将图像展平为 (num_pixels, 3) 的矩阵
    img_flat = img.reshape(-1, 3)

    # 获取变换后的主成分
    transformed_img = pca.inverse_transform(img_flat)

    return transformed_img.reshape(h, w, c)

if __name__ == '__main__':

    args = predefined_args()

    
    src_rgb = '../results/debug/images/100_0_gt.jpg'
    dst_rgb = '../results/debug/images/100_0_results_sample.jpg'

    src_rgb_org = cv2.imread(src_rgb)[:,:,::-1]
    gray_src = np.expand_dims(cv2.cvtColor(src_rgb_org, cv2.COLOR_RGB2GRAY), -1)

    dst_rgb_org = cv2.imread(dst_rgb)[:,:,::-1]
    gray_dst = np.expand_dims(cv2.cvtColor(dst_rgb_org, cv2.COLOR_RGB2GRAY), -1)

    # mask = (src_rgb_org[:, :, 0:1] > 0.8 * 255) | (src_rgb_org[:, :, 1:2] > 0.8 * 255) | (src_rgb_org[:, :, 2:] > 0.8 * 255)
    # mask |= (src_rgb_org[:, :, 0:1] < 0.2 * 255) | (src_rgb_org[:, :, 1:2] < 0.2 * 255) | (src_rgb_org[:, :, 2:] < 0.2 * 255)
    # mask = np.repeat(mask, 3, axis=-1)
    # print(np.max((src_rgb_org / (gray_src + eps))), np.min((src_rgb_org / (gray_src + eps))))
    # cv2.imwrite('./aligned_rgb.png', np.concatenate([(src_rgb_org / (gray_src + eps) / 2) * 255 , np.repeat((gray_dst + eps), 3, axis=-1), src_rgb_org, mask * 255], axis=-2)[:,:,::-1])
    # exit()
    # src_rgb_org = src_rgb_org / (gray_src + eps) * (gray_dst + eps)
    import copy
    src_rgb = copy.deepcopy(src_rgb_org)
    # src_rgb[mask] = dst_rgb_org[mask]
    # cv2.imwrite('./aligned_rgb.png', np.concatenate([src_rgb_org, dst_rgb_org, src_rgb, mask * 255], axis=-2)[:,:,::-1])
    # exit()
    # src_rgb_org = src_rgb

    aligned_rgb = np.zeros_like(dst_rgb)

    src_rgb = np.expand_dims(np.transpose(src_rgb / 255.0, (2, 0, 1)), 0)
    dst_rgb = np.expand_dims(np.transpose(dst_rgb_org / 255.0, (2, 0, 1)), 0)
    # print(np.max(src_rgb), np.min(src_rgb), np.min(dst_rgb), np.max(dst_rgb))
    
    # aligned_rgb = np.transpose(aligned_rgb[0], (1, 2, 0))

    src_rgb = torch.from_numpy(src_rgb)
    dst_rgb = torch.from_numpy(dst_rgb)
    # aligned_rgb = local_int_align_satDark_hist_norm_RGB(dst_rgb, src_rgb)
    aligned_rgb = local_int_align_satDark_hist_norm_pytorch(src_rgb, dst_rgb)

    aligned_rgb = np.transpose(aligned_rgb[0].cpu().numpy(), (1, 2, 0))
    # for i in range(3):
    #     aligned_rgb[:,:,i] = local_int_align_satDark_hist_norm(src_rgb[:,:,i], dst_rgb[:,:,i], args)
        # aligned_rgb[:,:,i] = local_int_align_satDark_hist_norm_sat(src_rgb[:,:,i], dst_rgb[:,:,i], args)
    
    cv2.imwrite('./aligned_rgb.png', np.concatenate([src_rgb_org, dst_rgb_org, aligned_rgb*255], axis=-2)[:,:,::-1])
    # cv2.imwrite('./aligned_rgb_sat.png', aligned_rgb[:,:,::-1]*255)
