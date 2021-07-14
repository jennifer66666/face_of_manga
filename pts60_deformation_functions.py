import numpy as np
from deformation_functions import check_deformation_spatial_errors

# original repo deform 68 points
# here we deform 60 pts for mangas
def pts60_deform_part(landmarks, part_inds, scale_y=1., scale_x=1., shift_ver=0., shift_horiz=0.):
    """ deform facial part landmarks - matching ibug annotations of 68 landmarks """

    landmarks_part = landmarks[part_inds, :].copy()
    part_mean = np.mean(landmarks_part, 0)

    landmarks_norm = landmarks_part - part_mean
    landmarks_deform = landmarks_norm.copy()
    landmarks_deform[:, 1] = scale_x * landmarks_deform[:, 1]
    landmarks_deform[:, 0] = scale_y * landmarks_deform[:, 0]

    landmarks_deform = landmarks_deform + part_mean
    landmarks_deform = landmarks_deform + shift_ver * np.array([1, 0]) + shift_horiz * np.array([0, 1])

    deform_shape = landmarks.copy()
    deform_shape[part_inds] = landmarks_deform
    return deform_shape


def pts60_deform_mouth(lms, p_scale=0, p_shift=0, pad=5):
    """ deform mouth landmarks - matching ibug annotations of 68 landmarks """

    jaw_line_inds = np.arange(0, 17)#0~16
    nose_inds = np.arange(49, 50)# nose get only one point 49
    mouth_inds = np.arange(50, 60)# 50~59

    part_inds = mouth_inds.copy()

    # find part spatial limitations
    jaw_pad = 4
    x_max = np.max(lms[part_inds, 1]) + (np.max(lms[jaw_line_inds[jaw_pad:-jaw_pad], 1]) - np.max(
        lms[part_inds, 1])) * 0.5 - pad
    x_min = np.min(lms[jaw_line_inds[jaw_pad:-jaw_pad], 1]) + (np.min(lms[part_inds, 1]) - np.min(
        lms[jaw_line_inds[jaw_pad:-jaw_pad], 1])) * 0.5 + pad
    y_min = np.max(lms[nose_inds, 0]) + (np.min(lms[part_inds, 0]) - np.max(lms[nose_inds, 0])) * 0.5
    max_jaw = np.minimum(np.max(lms[jaw_line_inds, 0]), lms[8, 0])
    y_max = max_jaw - (max_jaw - np.max(lms[part_inds, 0])) * 0.5 - pad

    # scale facial feature
    scale = np.random.rand()
    if p_scale > 0.5 and scale > 0.5:
        # 求嘴部分的x，y方向均值
        part_mean = np.mean(lms[part_inds, :], 0)
        # 鼻子部分基于均值归一化
        lms_part_norm = lms[part_inds, :] - part_mean
        # 找出归一化后鼻子部分的框
        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)
        # y部分的拉伸不能超过scale_max_y, 因为拉伸后的y不能越过y_min,y_max
        scale_max_y = np.minimum(
            (y_min - part_mean[0]) / part_y_bound_min,
            (y_max - part_mean[0]) / part_y_bound_max)
        # 而且拉伸不超过1.2倍
        scale_max_y = np.minimum(scale_max_y, 1.2)
        # 同理x部分的拉伸不能超过scale_max_x
        scale_max_x = np.minimum(
            (x_min - part_mean[1]) / part_x_bound_min,
            (x_max - part_mean[1]) / part_x_bound_max)
        # 而且拉伸不超过1.2倍
        scale_max_x = np.minimum(scale_max_x, 1.2)
        
        # 在0.7到最大拉伸值之间找随机数
        scale_y = np.random.uniform(0.7, scale_max_y)
        scale_x = np.random.uniform(0.7, scale_max_x)

        # 对嘴部分拉伸scale_x,scale_y; 不平移
        # 产出拉伸点lms_def_scale
        lms_def_scale = pts60_deform_part(lms, part_inds, scale_y=scale_y, scale_x=scale_x, shift_ver=0., shift_horiz=0.)

        # check for spatial errors
        error = check_deformation_spatial_errors(lms_def_scale, part_inds, pad=pad)
        if error:
            lms_def_scale = lms.copy()
    else:
        lms_def_scale = lms.copy()

    # shift facial feature
    if p_shift > 0.5 and (np.random.rand() > 0.5 or not scale):

        part_mean = np.mean(lms_def_scale[part_inds, :], 0)
        lms_part_norm = lms_def_scale[part_inds, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        shift_x = np.random.uniform(x_min - (part_mean[1] + part_x_bound_min),
                                    x_max - (part_mean[1] + part_x_bound_max))
        shift_y = np.random.uniform(y_min - (part_mean[0] + part_y_bound_min),
                                    y_max - (part_mean[0] + part_y_bound_max))
        # scale设为1就是不拉伸，仅作平移
        lms_def = pts60_deform_part(lms_def_scale, part_inds, scale_y=1., scale_x=1., shift_ver=shift_y, shift_horiz=shift_x)
        error = check_deformation_spatial_errors(lms_def, part_inds, pad=pad)
        if error:
            lms_def = lms_def_scale.copy()
    else:
        lms_def = lms_def_scale.copy()

    return lms_def


def pts60_deform_nose(lms, p_scale=0, p_shift=0, pad=5):
    """ deform nose landmarks - matching ibug annotations of 68 landmarks """
    
    # 左右眼珠要管吗？
    nose_inds = np.arange(49, 50)
    left_eye_inds = np.arange(27, 37)
    right_eye_inds = np.arange(37, 47)
    mouth_inds = np.arange(50, 60)

    part_inds = nose_inds.copy()

    # find part spatial limitations
    #                  鼻梁四点x轴上的最大值 +  右眼x轴上最小值                   -   鼻梁4点x轴最大值的一半   -pad？
    x_max = np.max(lms[part_inds[:4], 1]) + (np.min(lms[right_eye_inds, 1]) - np.max(lms[part_inds[:4], 1])) * 0.5 - pad
    #                  左眼x轴上最大值      +  鼻梁四点x轴最小值                - 左眼x最大值的一半            +pad？
    x_min = np.max(lms[left_eye_inds, 1]) + (np.min(lms[part_inds[:4], 1]) - np.max(lms[left_eye_inds, 1])) * 0.5 + pad

    max_brows = np.max(lms[21:23, 0])
    y_min = np.min(lms[part_inds, 0]) + (max_brows - np.min(lms[part_inds, 0])) * 0.5
    min_mouth = np.min(lms[mouth_inds, 0])
    y_max = np.max(lms[part_inds, 0]) + (np.max(lms[part_inds, 0]) - min_mouth) * 0 - pad

    # scale facial feature
    scale = np.random.rand()
    if p_scale > 0.5 and scale > 0.5:

        part_mean = np.mean(lms[part_inds, :], 0)
        lms_part_norm = lms[part_inds, :] - part_mean

        part_y_bound_min = np.min(lms_part_norm[:, 0])
        part_y_bound_max = np.max(lms_part_norm[:, 0])

        scale_max_y = np.minimum(
            (y_min - part_mean[0]) / part_y_bound_min,
            (y_max - part_mean[0]) / part_y_bound_max)
        scale_y = np.random.uniform(0.7, scale_max_y)
        scale_x = np.random.uniform(0.7, 1.5)

        lms_def_scale = deform_part(lms, part_inds, scale_y=scale_y, scale_x=scale_x, shift_ver=0., shift_horiz=0.)

        error1 = check_deformation_spatial_errors(lms_def_scale, part_inds[:4], pad=pad)
        error2 = check_deformation_spatial_errors(lms_def_scale, part_inds[4:], pad=pad)
        error = error1 + error2
        if error:
            lms_def_scale = lms.copy()
    else:
        lms_def_scale = lms.copy()

    # shift facial feature
    if p_shift > 0.5 and (np.random.rand() > 0.5 or not scale):

        part_mean = np.mean(lms_def_scale[part_inds, :], 0)
        lms_part_norm = lms_def_scale[part_inds, :] - part_mean

        part_x_bound_min = np.min(lms_part_norm[:4], 0)
        part_x_bound_max = np.max(lms_part_norm[:4], 0)
        part_y_bound_min = np.min(lms_part_norm[:, 0])
        part_y_bound_max = np.max(lms_part_norm[:, 0])

        shift_x = np.random.uniform(x_min - (part_mean[1] + part_x_bound_min),
                                    x_max - (part_mean[1] + part_x_bound_max))
        shift_y = np.random.uniform(y_min - (part_mean[0] + part_y_bound_min),
                                    y_max - (part_mean[0] + part_y_bound_max))

        lms_def = deform_part(lms_def_scale, part_inds, scale_y=1., scale_x=1., shift_ver=shift_y, shift_horiz=shift_x)

        error1 = check_deformation_spatial_errors(lms_def, part_inds[:4], pad=pad)
        error2 = check_deformation_spatial_errors(lms_def, part_inds[4:], pad=pad)
        error = error1 + error2
        if error:
            lms_def = lms_def_scale.copy()
    else:
        lms_def = lms_def_scale.copy()

    return lms_def


def pts60_deform_eyes(lms, p_scale=0, p_shift=0, pad=10,scale_x=None,scale_y=None):
    """ deform eyes + eyebrows landmarks - matching ibug annotations of 68 landmarks """

    nose_inds = np.arange(49, 50)
    left_eye_inds = np.arange(27, 37)
    right_eye_inds = np.arange(37, 47)
    left_brow_inds = np.arange(17, 22)
    right_brow_inds = np.arange(22, 27)

    part_inds_right = np.hstack((right_brow_inds, right_eye_inds))
    part_inds_left = np.hstack((left_brow_inds, left_eye_inds))

    # find part spatial limitations

    # right eye+eyebrow
    x_max_right = np.max(lms[part_inds_right, 1]) + (lms[16, 1] - np.max(lms[part_inds_right, 1])) * 0.5 - pad
    x_min_right = np.max(lms[nose_inds[:4], 1]) + (np.min(lms[part_inds_right, 1]) - np.max(
        lms[nose_inds[:4], 1])) * 0.5 + pad
    y_max_right = np.max(lms[part_inds_right, 0]) + (lms[33, 0] - np.max(lms[part_inds_right, 0])) * 0.25 - pad
    y_min_right = 2 * pad

    # left eye+eyebrow
    x_max_left = np.max(lms[part_inds_left, 1]) + (np.min(lms[nose_inds[:4], 1]) - np.max(
        lms[part_inds_left, 1])) * 0.5 - pad
    x_min_left = lms[0, 1] + (np.min(lms[part_inds_left, 1]) - lms[0, 1]) * 0.5 + pad

    y_max_left = np.max(lms[part_inds_left, 0]) + (lms[33, 0] - np.max(lms[part_inds_left, 0])) * 0.25 - pad
    y_min_left = 2 * pad

    # scale facial feature
    scale = np.random.rand()
    if p_scale > 0.5 and scale > 0.5:

        # right eye+eyebrow
        part_mean = np.mean(lms[part_inds_right, :], 0)
        lms_part_norm = lms[part_inds_right, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        scale_max_y = np.minimum(
            (y_min_right - part_mean[0]) / part_y_bound_min,
            (y_max_right - part_mean[0]) / part_y_bound_max)
        scale_max_y_right = np.minimum(scale_max_y, 1.5)

        scale_max_x = np.minimum(
            (x_min_right - part_mean[1]) / part_x_bound_min,
            (x_max_right - part_mean[1]) / part_x_bound_max)
        scale_max_x_right = np.minimum(scale_max_x, 1.5)

        # left eye+eyebrow
        part_mean = np.mean(lms[part_inds_left, :], 0)
        lms_part_norm = lms[part_inds_left, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        scale_max_y = np.minimum(
            (y_min_left - part_mean[0]) / part_y_bound_min,
            (y_max_left - part_mean[0]) / part_y_bound_max)
        scale_max_y_left = np.minimum(scale_max_y, 1.5)

        scale_max_x = np.minimum(
            (x_min_left - part_mean[1]) / part_x_bound_min,
            (x_max_left - part_mean[1]) / part_x_bound_max)
        scale_max_x_left = np.minimum(scale_max_x, 1.5)

        scale_max_x = np.minimum(scale_max_x_left, scale_max_x_right)
        scale_max_y = np.minimum(scale_max_y_left, scale_max_y_right)
        if scale_x==None and scale_y==None:
            scale_y = np.random.uniform(0.8, scale_max_y)
            scale_x = np.random.uniform(0.8, scale_max_x)

        lms_def_scale = pts60_deform_part(lms, part_inds_right, scale_y=scale_y, scale_x=scale_x, shift_ver=0.,
                                    shift_horiz=0.)
        lms_def_scale = pts60_deform_part(lms_def_scale.copy(), part_inds_left, scale_y=scale_y, scale_x=scale_x,
                                    shift_ver=0., shift_horiz=0.)

        error1 = check_deformation_spatial_errors(lms_def_scale, part_inds_right, pad=pad)
        error2 = check_deformation_spatial_errors(lms_def_scale, part_inds_left, pad=pad)
        error = error1 + error2
        if error:
           lms_def_scale = lms.copy()
    else:
        lms_def_scale = lms.copy()

    # shift facial feature
    if p_shift > 0.5 and (np.random.rand() > 0.5 or not scale):

        y_min_right = np.maximum(0.8 * np.min(lms_def_scale[part_inds_right, 0]), pad)
        y_min_left = np.maximum(0.8 * np.min(lms_def_scale[part_inds_left, 0]), pad)

        # right eye
        part_mean = np.mean(lms_def_scale[part_inds_right, :], 0)
        lms_part_norm = lms_def_scale[part_inds_right, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        shift_x = np.random.uniform(x_min_right - (part_mean[1] + part_x_bound_min),
                                    x_max_right - (part_mean[1] + part_x_bound_max))
        shift_y = np.random.uniform(y_min_right - (part_mean[0] + part_y_bound_min),
                                    y_max_right - (part_mean[0] + part_y_bound_max))

        lms_def_right = pts60_deform_part(lms_def_scale, part_inds_right, scale_y=1., scale_x=1., shift_ver=shift_y,
                               shift_horiz=shift_x)

        error1 = check_deformation_spatial_errors(lms_def_right, part_inds_right, pad=pad)
        if error1:
            lms_def_right = lms_def_scale.copy()

        # left eye
        part_mean = np.mean(lms_def_scale[part_inds_left, :], 0)
        lms_part_norm = lms_def_scale[part_inds_left, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        shift_x = np.random.uniform(x_min_left - (part_mean[1] + part_x_bound_min),
                                    x_max_left - (part_mean[1] + part_x_bound_max))
        shift_y = np.random.uniform(y_min_left - (part_mean[0] + part_y_bound_min),
                                    y_max_left - (part_mean[0] + part_y_bound_max))

        lms_def = pts60_deform_part(lms_def_right.copy(), part_inds_left, scale_y=1., scale_x=1., shift_ver=shift_y,
                              shift_horiz=shift_x)

        error2 = check_deformation_spatial_errors(lms_def, part_inds_left, pad=pad)
        if error2:
            lms_def = lms_def_right.copy()
    else:
        lms_def = lms_def_scale.copy()

    return lms_def


def pts60_deform_scale_face(lms, p_scale=0, pad=5, image_size=256):
    """ change face landmarks scale & aspect ratio - matching ibug annotations of 68 landmarks """

    part_inds = np.arange(60)

    # find spatial limitations
    x_max = np.max(lms[part_inds, 1]) + (image_size - np.max(lms[part_inds, 1])) * 0.5 - pad
    x_min = np.min(lms[part_inds, 1]) * 0.5 + pad

    y_min = 2 * pad
    y_max = np.max(lms[part_inds, 0]) + (image_size - np.max(lms[part_inds, 0])) * 0.5 - pad

    if p_scale > 0.5:

        part_mean = np.mean(lms[part_inds, :], 0)
        lms_part_norm = lms[part_inds, :] - part_mean

        part_y_bound_min, part_x_bound_min = np.min(lms_part_norm, 0)
        part_y_bound_max, part_x_bound_max = np.max(lms_part_norm, 0)

        scale_max_y = np.minimum(
            (y_min - part_mean[0]) / part_y_bound_min,
            (y_max - part_mean[0]) / part_y_bound_max)
        scale_max_y = np.minimum(scale_max_y, 1.2)

        scale_max_x = np.minimum(
            (x_min - part_mean[1]) / part_x_bound_min,
            (x_max - part_mean[1]) / part_x_bound_max)
        scale_max_x = np.minimum(scale_max_x, 1.2)

        scale_y = np.random.uniform(0.6, scale_max_y)
        scale_x = np.random.uniform(0.6, scale_max_x)

        lms_def_scale = pts60_deform_part(lms, part_inds, scale_y=scale_y, scale_x=scale_x, shift_ver=0., shift_horiz=0.)

        # check for spatial errors
        error2 = np.sum(lms_def_scale >= image_size) + np.sum(lms_def_scale < 0)
        error1 = len(np.unique((lms_def_scale).astype('int'), axis=0)) != len(lms_def_scale)
        error = error1 + error2
        if error:
            lms_def_scale = lms.copy()
    else:
        lms_def_scale = lms.copy()

    return lms_def_scale


def pts60_deform_face_geometric_style(lms, p_scale=0, p_shift=0,scale_x=None,scale_y=None):
    """ deform facial landmarks - matching ibug annotations of 68 landmarks """

    lms = pts60_deform_scale_face(lms.copy(), p_scale=p_scale, pad=0)
    # 做鼻子好像有溢出的问题
    #lms = pts60_deform_nose(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0)
    lms = pts60_deform_mouth(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0)
    lms = pts60_deform_eyes(lms.copy(), p_scale=p_scale, p_shift=p_shift, pad=0,scale_x=scale_x,scale_y=scale_y)
    return lms


