from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths
import mmcv
import cv2
import json
import os
import os.path as osp
import scipy.optimize
import scipy.spatial
import numpy as np
# import utils.boxes as bbox_utils
import utils.keypoints as kps_utils
import utils.image as img_utils
from core.config import config as cfg



MAX_TRACK_IDS = 999
FIRST_TRACK_ID = 0

root_path = '/mnt/data-1/data/yang.bai/PoseTrack2018/images/posetrack_data'

def _cs2box(center, scale):
    scale = np.array(scale)
    if center[0] != -1:
        scale = scale / 1.25
    scale = scale * 200
    w, h = scale
    x1 = center[0] - w / 2
    y1 = center[1] - h / 2
    x2 = center[0] + w / 2
    y2 = center[1] + h / 2
    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if y1 < 0:
        y1 = 0
    if y2 < 0:
        y2 = 0
    return [x1, y1, x2, y2]



def _get_boxes_cs2xywh(dets, det_id):
    # if det_id in que_list:
    #     temp = [0,0,0,0]
    #     return np.array(temp)

    value_list = dets[det_id]
    bbox_list = []
    for item in value_list:
        temp = _cs2box(item[0], item[1])
        bbox_list.append(temp)
    return np.array(bbox_list)
    # return bbox_list


def _get_boxes(dets, det_id):
    value_list = dets[det_id]
    bbox_list = []
    for item in value_list:
        bbox_list.append(item[3])
    return np.array(bbox_list)

def _get_poses(dets, det_id):
    # if det_id in que_list:
    #     return [0]*17
    value_list = dets[det_id]
    # print (det_id, value_list)
    kps_list = []
    for item in value_list:
        kps_list.append(item[2])
    # return np.array(kps_list)
    # print (det_id, kps_list)
    return kps_list

def _write_det_file(det, dets_withTracks, out_det_file):
    pass



# def _convert_pose_3x17(kps):
#     """
#     kps: 1x51
#     """


#     return res

def _convert_pose_3x17(kps):
    res = []
    kps = np.array(kps)
    kps_x = kps[0::3]
    kps_y = kps[1::3]
    kps_vis = kps[2::3]
    # kps_vis = np.ones((17,))
    res.append(kps_x)
    res.append(kps_y)
    res.append(kps_vis)
    res = np.array(res)
    return res


def _convert_det_result_to_track_cal(det_data):
    """
    det_data: kps, center, scale.
    """
    image_id_list = []
    for i in range(len(det_data['annotations'])):
        image_id_list.append(det_data['annotations'][i]['image_id'])
    image_id_set = list(set(image_id_list))
    image_id_set.sort()

    det_data_for_track = {}
    """
    {'image_id':[(center, scale, kps), (center, scale, kps), ...]}
    0 -> cetner
    1 -> scale
    2 -> keypoints
    """

    for item in image_id_set:
        det_data_for_track[item] = []

    for i in range(len(det_data['annotations'])):
        img_id = det_data['annotations'][i]['image_id']
        center = det_data['annotations'][i]['center']
        scale = det_data['annotations'][i]['scale']
        kps = det_data['annotations'][i]['keypoints']
        box = det_data['annotations'][i]['box']
        temp = (center, scale, kps, box)
        # print (temp)
        # print (type(center), type(scale), type(kps))
        det_data_for_track[img_id].append(temp)

    return det_data_for_track




    # test the new data format for oks-similarity.
    # cnt = 0
    # key_a = key_b = 0
    # for key in det_data_for_track.keys():
    #     if cnt == 0:
    #         key_a = key
    #     if cnt == 1:
    #         key_b = key
    #         break
    #     cnt += 1
    # print (key_a, key_b)

    # pose_a = det_data_for_track[key_a][0][2]
    # pose_b = det_data_for_track[key_b][0][2]
    # scale_a = det_data_for_track[key_a][0][1]
    # scale_b = det_data_for_track[key_b][0][1]
    # pose_a = _convert_pose_3x17(pose_a)
    # pose_b = _convert_pose_3x17(pose_b)
    # difference = _compute_pairwise_kps_oks_distance(pose_a, scale_a, pose_b, scale_b)
    # print (difference)





# def _test_pairwise_kps_distance(kps_a, kps_b):
#     print (kps_a)
#     print (kps_b)

def _compute_pairwise_iou():
    """
    a, b (np.ndarray) of shape Nx4T and Mx4T.
    The output is NxM, for each combination of boxes.
    """
    return bbox_utils.bbox_overlap(a,b)


def _compute_deep_features(imname, boxes):
    import utils.cnn_features as cnn_utils
    print (imname)
    I = cv2.imread(imname)
    if I is None:
        raise ValueError('Image not found {}'.format(imname))
    # print ('the shape of I',I.shape)
    all_feats = []
    # print ('Now is the {} info'.format(imname))
    # print ('Bbox info:', boxes)
    for box in boxes:
        patch = I[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2]), :]
        # print (int(box[1]), int(box[3]), int(box[0]), int(box[2]))

        # if type(box) != np.array:
        #     return np.zeros((0,))
        # patch = I[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        all_feats.append(cnn_utils.extract_features(
            patch, layers = (cfg.TRACKING.CNN_MATCHING_LAYER,)))
    return np.stack(all_feats) if len(all_feats) > 0 else np.zeros((0, ))



def _compute_pairwise_deep_cosine_dist(a_imname, a, b_imname, b):
    f1 = _compute_deep_features(a_imname, a)
    f2 = _compute_deep_features(b_imname, b)
    if f1.size * f2.size == 0:
        return np.zeros((f1.shape[0], f2.shape[0]))
    return scipy.spatial.distance.cdist(
        f1.reshape((f1.shape[0], -1)), f2.reshape((f2.shape[0], -1)),
        'cosine')

def _compute_pairwise_kps_pck_distance(kps_a, kps_b):
    res = np.zeros((len(kps_a), len(kps_b)))
    print (res.shape[0], res.shape[1])
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            pose_a = _convert_pose_3x17(kps_a[i])
            pose_b = _convert_pose_3x17(kps_b[j])
            res[i, j] = kps_utils.pck_distance(pose_a, pose_b)
    return res




def _compute_pairwise_kps_oks_distance(kps_a, box_a, kps_b, box_b):
    # print (type(kps_a), type(kps_b))
    # print (box_a.shape)
    # pose_a = _convert_pose_3x17(kps_a)
    # pose_b = _convert_pose_3x17(kps_b)
    # print (type(pose_a), type(pose_b))
    # print ('---------------------------')
    # print (pose_a.shape, pose_b.shape)
    # print (pose_a, pose_b)
    res = np.zeros((len(kps_a), len(kps_b)))
    kps_a = np.array(kps_a)
    kps_b = np.array(kps_b)
    print ('kps_ab.shape', kps_a.shape, kps_b.shape)
    # print (res.shape)
    for i in range(len(kps_a)):
        temp_list = []
        for j in range(len(kps_b)):
            pose_a = _convert_pose_3x17(kps_a[i])
            pose_b = _convert_pose_3x17(kps_b[j])
            # print (pose_a, pose_b)
            b_a = box_a[i]
            b_b = box_b[j]
            res[i, j] = kps_utils.compute_oks(pose_a, b_a, pose_b, b_b)
            temp_list.append(res[i,j])

        temp_list = np.array(temp_list)
        # print ('{} => {}'.format(i, temp_list.argmax()))

    # print (res)
    # for i in range(res.shape[0]):
    #     for j in range(res.shape[1]):
    #         if i==j:
    #             print (res[i, j])
    return res



# def _compute_pairwise_kps_pck_distance(kps_a, kps_b):
#     """
#     kps_a, kps_b  has shape of (51,)
#     """
#     res_shape = (len(kps_a), len(kps_b))
#     res = np.zeros(res_shape)
#     for i in range(len(kps_a)):
#         for j in range(len(kps_b)):
#             res[i, j] = kps_utils.compute_oks(kps_a[i], kps_b[j])
#     return res


def _compute_nframes(dets):
    nframe = []
    for key in dets.keys():
        nframe.append(key)
    nframe = list(set(nframe))
    return len(nframe), nframe

def _get_frame_data_det_id(data, frame_id):
    frame_data = data['images'][frame_id]['file_name']
    det_id = data['images'][frame_id]['id']
    return frame_data, det_id

def bipartite_matching_greedy(C):
    """
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])
    while C.size > 0:
        # Find the lowest cost element
        i, j = np.unravel_index(C.argmin(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]
        prev_ids.append(row_id)
        cur_ids.append(col_id)
        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids

def _compute_distance_matrix(prev_json_data, prev_boxes, prev_poses,
    cur_json_data, cur_boxes, cur_poses,
    cost_types, cost_weights):
    assert(len(cost_types) == len(cost_weights))
    all_Cs = []
    for cost_type, cost_weight in zip(cost_types, cost_weights):
        if cost_weight == 0:
            continue
        if cost_type == 'bbox_overlap':
            all_Cs.append((1 - _compute_pairwise_iou(prev_boxes, cur_boxes)))
        elif cost_type == 'cnn-cosdist':
            all_Cs.append(_compute_pairwise_deep_cosine_dist(
                img_utils.get_image_path(prev_json_data), prev_boxes,
                img_utils.get_image_path(cur_json_data), cur_boxes))
        elif cost_type == 'pose-pck':
            all_Cs.append(_compute_pairwise_kps_pck_distance(prev_poses, cur_poses))
        elif cost_type == 'pose-oks':
            all_Cs.append(_compute_pairwise_kps_oks_distance(prev_poses, prev_boxes, cur_poses, cur_boxes))
        else:
            raise NotImplementedError('Unknown cost type {}'.format(cost_type))
        # print ('cost_weight', cost_weight)
        # print ('all_Cs', all_Cs)
        all_Cs[-1] *= cost_weight
        # print ('before sum', all_Cs)
    return np.sum(np.stack(all_Cs, axis=0), axis=0)



def _compute_matches(prev_frame_data, cur_frame_data, prev_boxes, cur_boxes,
                     prev_poses, cur_poses,
                     cost_types, cost_weights,
                     bipart_match_algo,
                     C = None):
    """
    C (cost matrix): num_prev_boxes x num_current_boxes
    Optionally input the cost matrix, in which case you can input dummy values
    for the boxes and poses
    Returns:
        matches: A 1D np.ndarray with as many elements as boxes in current
        frame (cur_boxes). For each, there is an integer to index the previous
        frame box that it matches to, or -1 if it doesnot match to any previous
        box.
    """
    # matches structure keeps track of which of the current boxes matches to
    # which box in the previous frame. If any idx remains -1, it will be set
    # as a new track.
    if C is None:
        nboxes = cur_boxes.shape[0]
        matches = -np.ones((nboxes,), dtype = np.int32)
        C = _compute_distance_matrix(
            prev_frame_data, prev_boxes, prev_poses,
            cur_frame_data, cur_boxes, cur_poses,
            cost_types = cost_types,
            cost_weights = cost_weights)
        # print ('after sum', C)
    else:
        matches = -np.ones((C.shape[1],), dtype = np.int32)

    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = scipy.optimize.linear_sum_assignment(C)
    elif bipart_match_algo == 'greedy':
        prev_inds, next_inds = bipartite_matching_greedy(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(bipart_match_algo))
    assert(len(prev_inds) == len(next_inds))

    for i in range(len(prev_inds)):
        matches[next_inds[i]] = prev_inds[i]
    return matches


# def _compute_tracks(data, dets):
#     nframe, frame_list = _compute_nframes(dets)
#     nframe2 = len(data['images'])
#     frame2_list = []
#     for i in range(len(data['images'])):
#         frame2_list.append(data['images'][i]['id'])
#     # when nframe is not equal to nframe2
#     # print (nframe, nframe2)
#     # # for item in data['images']:
#     # #     print (item['id'])
#     # # if nframe != nframe2:
#     # for item in frame_list:
#     #     print (item)
#     print ('dets_nframe:', nframe)
#     print ('data_nframe:', nframe2)
#     if nframe != nframe2:
#         for item in frame2_list:
#             if item not in frame_list:
#                 que_list.append(item)
#     print(que_list)





#     video_tracks = []
#     next_track_id = FIRST_TRACK_ID

#     for frame_id in range(nframe):
#         frame_tracks = []

#         # frame_data, det_id is {'images':[{'file_name':,'id':,}]}
#         frame_data, det_id = _get_frame_data_det_id(data, frame_id)
#         cur_boxes = _get_boxes(dets, det_id)
#         cur_poses = _get_poses(dets, det_id)
#         # print (type(cur_poses))
#         # print ('xxxxxxxxxxxxxxxxxxxxxxx')

#         if frame_id == 0:
#             matches = -np.ones((cur_boxes.shape[0],))
#             # print ('matches', matches)
#         else:
#             cur_frame_data = frame_data
#             prev_boxes = _get_boxes(dets, _get_frame_data_det_id(data, frame_id-1)[1])
#             prev_poses = _get_poses(dets, _get_frame_data_det_id(data, frame_id-1)[1])
#             prev_frame_data = _get_frame_data_det_id(data, frame_id-1)[0]

#             # prev_poses = np.array(prev_poses)
#             # cur_poses = np.array(cur_poses)
#             # print (prev_poses.shape, cur_poses.shape)
#             matches = _compute_matches(
#                 prev_frame_data, cur_frame_data,
#                 prev_boxes, cur_boxes, prev_poses, cur_poses,
#                 cost_types = cfg.TRACKING.DISTANCE_METRICS,
#                 cost_weights = cfg.TRACKING.DISTANCE_METRICS_WTS,
#                 bipart_match_algo = cfg.TRACKING.BIPARTITE_MATCHING_ALGO)

#         prev_tracks = video_tracks[frame_id-1] if frame_id > 0 else None
#         # print ('@@@@@@@@@@@@@@@@@@@')
#         # print ('matches',matches)
#         for m in matches:
#             if m == -1:
#                 frame_tracks.append(next_track_id)
#                 next_track_id += 1
#                 if next_track_id >= MAX_TRACK_IDS:
#                     next_track_id %= MAX_TRACK_IDS
#             else:
#                 frame_tracks.append(prev_tracks[m])

#         video_tracks.append(frame_tracks)
#         # print (video_tracks)
#         # print (frame_tracks)
#         # if frame_id == 5:
#         #     break
#     return video_tracks


def _compute_tracks(data, dets):
    nframe, frame_list = _compute_nframes(dets)
    nframe2 = len(data['images'])
    frame2_list = []
    for i in range(len(data['images'])):
        frame2_list.append(data['images'][i]['id'])
    # when nframe is not equal to nframe2
    # print (nframe, nframe2)
    # # for item in data['images']:
    # #     print (item['id'])
    # # if nframe != nframe2:
    # for item in frame_list:
    #     print (item)
    print ('dets_nframe:', nframe)
    print ('data_nframe:', nframe2)
    que_list = []
    if nframe != nframe2:
        for item in frame2_list:
            if item not in frame_list:
                que_list.append(item)
    print(que_list)

    video_tracks = []
    next_track_id = FIRST_TRACK_ID
    # pre_frame = [] # pre frame of current, normal, 0 1 2 3 4 5 6. abnormal, 0 3 4 5 6 7 
    # for i in range(nframe):
    #     if i != 0:
    #         pre_frame[i] = i-1
    # print (pre_frame)
    skip_num = 0
    cnt = 0
    is_first = 1
    if data['images'][0]['id'] in que_list:
        is_first = 0
    for frame_id in range(nframe):
        frame_tracks = []

        # frame_data, det_id is {'images':[{'file_name':,'id':,}]}
        if data['images'][frame_id]['id'] in que_list:
            skip_num = skip_num + 1
            continue
        # print (data['images'][frame_id]['id'])
        frame_data, det_id = _get_frame_data_det_id(data, frame_id)
        cur_boxes = _get_boxes(dets, det_id)
        cur_poses = _get_poses(dets, det_id)
        print ('cur_boxes', cur_boxes)
        # print (type(cur_poses))
        # print ('xxxxxxxxxxxxxxxxxxxxxxx')

        if frame_id == 0:
            matches = -np.ones((cur_boxes.shape[0],))
            # print ('matches', matches)
        else:
            if is_first == 0:
                skip_num = 0
                matches = -np.ones((cur_boxes.shape[0], ))
                is_first = 1
                prev_tracks = video_tracks[cnt-1] if cnt > 0 else None
                cnt = cnt + 1
                skip_num = 0
                for m in matches:
                    if m == -1:
                        frame_tracks.append(next_track_id)
                        next_track_id += 1
                        if next_track_id >= MAX_TRACK_IDS:
                            next_track_id %= MAX_TRACK_IDS
                video_tracks.append(frame_tracks)
                continue
            cur_frame_data = frame_data
            # print ('skip_num', skip_num)
            # print ('cur_frame_id', frame_id)
            # print ('pre_frame_id', frame_id-1-skip_num)
            prev_boxes = _get_boxes(dets, _get_frame_data_det_id(data, frame_id-1-skip_num)[1])
            prev_poses = _get_poses(dets, _get_frame_data_det_id(data, frame_id-1-skip_num)[1])
            prev_frame_data = _get_frame_data_det_id(data, frame_id-1-skip_num)[0]

            # prev_poses = np.array(prev_poses)
            # cur_poses = np.array(cur_poses)
            # print (prev_poses.shape, cur_poses.shape)
            matches = _compute_matches(
                prev_frame_data, cur_frame_data,
                prev_boxes, cur_boxes, prev_poses, cur_poses,
                cost_types = cfg.TRACKING.DISTANCE_METRICS,
                cost_weights = cfg.TRACKING.DISTANCE_METRICS_WTS,
                bipart_match_algo = cfg.TRACKING.BIPARTITE_MATCHING_ALGO)
        print (video_tracks)
        prev_tracks = video_tracks[cnt-1] if cnt > 0 else None
        cnt = cnt + 1
        skip_num = 0
        # print ('@@@@@@@@@@@@@@@@@@@')
        # print ('matches',matches)
        # print (matches)
        for m in matches:
            if m == -1:
                frame_tracks.append(next_track_id)
                next_track_id += 1
                if next_track_id >= MAX_TRACK_IDS:
                    next_track_id %= MAX_TRACK_IDS
            else:
                frame_tracks.append(prev_tracks[m])

        video_tracks.append(frame_tracks)
        # print (video_tracks)
        # print (frame_tracks)
        # if frame_id == 5:
        #     break
    return que_list, video_tracks



def _summarize_track_stats(data, tracks):
    pass





def _compute_matches_tracks(data, dets):
    tracks = _compute_tracks(data, dets)

    # resort and assign:
    _summarize_track_stats(data, tracks)
    return tracks


def _write_det_file(out_det_file, dict_file):
    with open(out_det_file, 'w') as fin:
        json.dump(dict_file, fin)


def main(det_file):
    # Only for one video
    test_output_dir = '/home/users/yang.bai/project/analysis_result_tf_pytorch/cascaded_rcnn_person_data_posetrack_detectorwithTracks_050_result'
    # det_file = '/home/users/yang.bai/project/analysis_result_tf_pytorch/cascaded_rcnn_person_detector_070_result/000522_mpii_test.json'
    file_name = det_file.split('/')[-1]
    out_det_file = osp.join(test_output_dir, file_name)
    # gt_file = '/mnt/data-1/data/yang.bai/PoseTrack2018/images/posetrack_data/annotations_original/val/000522_mpii_test.json'

    if not osp.exists(det_file):
        raise ValueError('det file not found {}'.format(det_file))

    # dets_withTracks = _compute_matches_tracks()
    # _write_det_file(dets_withTracks, out_det_file)
    data = mmcv.load(det_file)
    dets = _convert_det_result_to_track_cal(data)
    # frame0 = 10005220000
    # frame1 = 10005220001
    # # print (dets[10005220000])
    # # print (dets[10005220001])
    # for item1, item2 in zip(dets[frame0], dets[frame1]):
    #     entre1 = np.array(item1[2])
    #     entre2 = np.array(item2[2])
    #     diff = entre1 - entre2
    #     print (diff, diff.shape)


    # print (dets)
    # print (data['annotations'][0]['image_id'])
    # print (data['annotations'][16]['image_id'])
    # for i in range(len(data['annotations'])):
    #     img_id = data['annotations'][i]['image_id']
    # _test_pairwise_kps_distance()
    # vid_name = data['images'][0]['file_name'].split('/')[-2]
    vid_name = data['images'][0]['vid_id']
    print ('Computing tracks for {} video'.format(vid_name))
    que_list, dets_withTracks = _compute_matches_tracks(data, dets)
    print (dets_withTracks)
    # print (dets_withTracks)
    # cnt = 0 
    # for i in range(len(dets_withTracks)):
    #     for j in range(len(dets_withTracks[i])):
    #         cnt += 1
    # print (cnt)
    # print (len(data['annotations']))
    # cnt = 0
    # for i in range(len(data['annotations'])):
    #     kps_det = data['annotations'][i]['keypoints']
    #     kps_frameid = data['annotations'][i]['image_id']
    #     assert dets[kps_frameid][cnt][2] == kps_det
    #     cnt += 1
    #     print (kps_frameid)
    res_track = []
    for i in range(len(dets_withTracks)):
        for j in range(len(dets_withTracks[i])):
            res_track.append(dets_withTracks[i][j])
    # print (res_track)
    print (file_name)
    print ('track_length:', len(res_track))
    print ('need to cal track:', len(data['annotations']))

    # print (res_track)
    # cnt = 0
    # for i in range(len(data['annotations'])):
    #     data['annotations'][i]['track_id'] = res_track[cnt]
    #     cnt += 1
    # print (cnt)

    cnt = 0
    # print (que_list)
    for i in range(len(res_track)):
        # print (data['annotations'][i]['image_id'])
        if data['annotations'][i]['image_id'] in que_list:
            # data['annotations'][i]['track_id'] = -1     
            print ('Not exists!')
        else:
            data['annotations'][i]['track_id'] = res_track[cnt]
        # print (cnt)
        # print (data['annotations'][i]['track_id'])
        cnt += 1
    print (cnt)

    _write_det_file(out_det_file, data) 


if __name__ == '__main__':
    det_root = '/home/users/yang.bai/project/analysis_result_tf_pytorch/cascaded_rcnn_person_detector_data_posetrack_050_result/'
    json_file = os.listdir(det_root)
    cnt = 0
    for item in json_file:
        det_path = osp.join(det_root, item)
        print ('Processing {} video for tracking'.format(cnt))
        # abnormal_path = '/home/users/yang.bai/project/analysis_result_tf_pytorch/cascaded_rcnn_person_detector_080_result/024158_mpii_test.json'
        print (det_path)
        main(det_path)
        # break
        cnt = cnt + 1
