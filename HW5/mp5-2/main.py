import os
import time
from PIL import Image
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
cmap = cm.jet(np.random.permutation(256))

from utils import orientation_correction, process_dets, \
    get_frame_det, Box3D, load_detection
from matching import data_association
from kitti_oxts import ego_motion_compensation, load_oxts
from kitti_calib import Calibration
from vis import visualization, vis_image_with_obj, vis_obj
from kalman_filter import Kalman

def track_sequence(seq_dets, num_frames, oxts, calib, vis_dir, image_dir, eval_file, max_age=3, algm="greedy"):
    '''
    seq_dets: the detected objects in 154 images: (1054, 15)
    max_frame: images number 154
    oxts_imu: OXTS文件中提供的GPS/IMU数据集合并转换为 (154, 4, 4)
    calib: A class with read-in data 相机标定： for instance: y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo (https://zhuanlan.zhihu.com/p/99114433)
    '''
    # Don't need to initialize trackers, first iteration will birth new ones
    # Each tracker in this list is a Kalman Filter, i.e. object of type kalman_filter.Kalman
    trackers = []
    ID_count = 0

    # You can set this to True to see visualization on screen each frame
    viz_on_screen = False
    Q6 = False
    if viz_on_screen is True:
        plt.figure()
    for frame in range(num_frames):
        # seq_dets contains all the detections for the entire sequence
        # So first we need to process the detections into the right format, and extract current frame
        frame_dets = get_frame_det(seq_dets, frame)
        # frame_dets: N x 7, float numpy array
        frame_dets, info = frame_dets['dets'], frame_dets['info']
        # print("frame_dets: {}".format(frame_dets)," and info:{}".format(info))
        frame_dets = process_dets(frame_dets) # put the data into a bbox class : [[h,w,l,x,y,z,theta],...]
                                              # center x, center y,center z, height, width, length, orientation, detection score(optional)

        # 1. Prediction/Propagation
        #   Here we predict where each tracked object would be according to our dynamics model
        #   TODO (student): You are tasked with implementing the Kalman filter method "predict()"
        # ---------------
        # For the next parts we sometimes need each tracker's 3D bounding box, so we save it here
        trks_bbox = []
        for t in range(len(trackers)):
            tmp_tracker = trackers[t]
            tmp_tracker.predict() # Your implementation

            # Why do we have this?
            #   Imagine a tracker gets occluded for a few frames.
            #   We won't call update, but we will keep propagating it forward
            #   If a tracker isn't updated for too many steps we will later kill it
            tmp_tracker.time_since_update += 1

            # get bounding box for next parts
            tmp_tracker = tmp_tracker.x.reshape((-1))[:7]
            trks_bbox.append(Box3D.array2bbox(tmp_tracker))
        # ---------------

        # 2. Ego-Motion-Compensation
        #   You do not have to write any code here, but try removing this code block and seeing how it affects your results
        #   If the camera itself is moving, even stationary objects will appear to be in motion
        #       This makes the problem of tracking much harder
        #       We can fix this problem with a simple trick: ego-motion compensation
        #   Ego-motion compensation means we use IMU and camera information to update each tracker's location
        #       In other words, since we've already propagated each tracker forward,
        #           after this step detections and trackers should be very similar
        # ---------------
        if (frame > 0):
            # Note this will also update the state in each tracker, not just trks_bbox
            trks_bbox = ego_motion_compensation(frame, trks_bbox, trackers, oxts, calib)
            # To use this code below define img as we do in the visualization section
            # visualization(img, [], trks_bbox, trackers, calib, save_path) # show only detections
            # visualization(img, frame_dets, [], [], calib, save_path) # show only trackers
            # visualization(img, frame_dets, trks_bbox, trackers, calib, save_path) # show both
        # ---------------

        # 3. Matching
        #   Now we find a matching between our current frame detections and previous frame trackers
        #   We must do matching because our detections are observations,
        #       and we need to know which observation should update which kalman filter tracker
        #   More info is in matching.py
        #       we provide you with the cost metric for comparing boxes, i.e. 3D box IoU
        #   TODO (student): You are tasked with implementing the method "data_association(detections, tracks)" in matching.py
        # ---------------
        matched, unmatched_dets, unmatched_trks = data_association(frame_dets, trks_bbox, threshold=-0.2, algm=algm)
        # ---------------

        # 4. Observation Model Update
        #   Now we can do a Kalman Filter update to the trackers with assigned detections
        #   TODO (student): You are tasked with implementing the Kalman Filter method "update(bbox3d)"
        # ---------------
        if Q6:
            trk_color_list = []
            det_color_list = []
            new_dets = []
            new_trks = []
        for t, trk in enumerate(trackers):
            if t not in unmatched_trks:
                # print("matches:", matched)
                d = matched[np.where(matched[:, 1] == t)[0], 0] # a list of single index
                assert len(d) == 1, 'error'

                # update statistics
                trk.time_since_update = 0 # reset because just updated
                trk.hits += 1

                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(frame_dets[d[0]])
                trk.x[3], bbox3d[3] = orientation_correction(trk.x[3], bbox3d[3])

                # kalman filter update with observation
                trk.update(bbox3d) # Your implementation

                trk.info = info[d, :][0]

                if Q6:
                    trk_color_list.append((0, 0, 255)) # blue
                    det_color_list.append((255, 255, 0)) # yellow
                    new_dets.append(bbox3d)
                    new_trks.append(trk)
        # ---------------

        # 5. Birth
        #     Create and initialise new trackers for unmatched detections
        # ---------------
        for i in unmatched_dets: # a scalar of index of detection
            trk = Kalman(Box3D.bbox2array(frame_dets[i]), info[i, :], ID_count)
            trackers.append(trk)
            ID_count += 1
            if Q6:
                new_trks.append(trk)
                trk_color_list.append((0, 255, 0)) # green
        # ---------------

        # 6. Death
        #     Remove tracks that have been inactive for too long
        # ---------------
        num_trks = len(trackers)
        for trk in reversed(trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            d = Box3D.array2bbox(trk.x[:7].reshape((7, )))     # bbox location self
            d = Box3D.bbox2array_raw(d)
            num_trks -= 1
            # remove dead tracker
            if (trk.time_since_update >= max_age):
                trackers.pop(num_trks)
                if Q6:
                    a = 0
                    for i in range(len(new_trks)):
                        if new_trks[i].ID == trk.ID:
                            trk_color_list[i] = (255, 0, 0) # red
                            a = 1
                    if a == 0:
                        new_trks.append(trk)
                        trk_color_list.append((255, 0, 0))
        # ---------------

        # Visualization
        #   There are a couple ways to visualize your results, we provide one example below
        #   Note that most likely you will want to visualize in different places:
        #       before/after each change to trackers
        #       the detections
        #       the detections AND matched trackers
        #   The easiest function you should use is vis_obj, its params are:
        #       bbox3D          - such as each detection, or each tracker (once converted to bbox)
        #       img (np array)  - don't change, use the code we provide below
        #       calib           - don't change
        #       hw              - don't change, (375,1242)
        #       color           - 3 tuple, ie. (255,0,0) for red
        #       str_vis         - string to put above bbox, for example the tracker ID
        #   You should modify and move this code around for your debugging purposes
        # ---------------
        if True:
            if image_dir is None:
                img = np.zeros((375, 1242, 3), np.uint8)
            else:
                img = os.path.join(image_dir, "{:06d}.png".format(frame))
                img = np.array(Image.open(img))

            save_path = os.path.join(vis_dir, "{}.jpg".format(frame))
            hw = (375, 1242)

            for trk in trackers:
                trk_tmp = Box3D.array2bbox(trk.x.reshape((-1))[:7])
                str_vis = "{}".format(trk.ID)
                trk_color = tuple([int(tmp * 255) for tmp in cmap[trk.ID % 256][:3]])
                img = vis_obj(trk_tmp, img, calib, hw, trk_color, str_vis)

            # TODO: Q0 visualize the detection boxs
            # i = 0
            # for frame_det in frame_dets:
            #     img = vis_obj(frame_det, img, calib, hw)
            #     det_vis = "{}".format(i)
            #     det_color = tuple([int(tmp * 255) for tmp in cmap[i % 256][:3]])
            #     img = vis_obj(frame_det, img, calib, hw, det_color, det_vis)
            #     i += 1

            img = Image.fromarray(img)
            img = img.resize((hw[1], hw[0]))
            img.save(save_path)
            if viz_on_screen:
                plt.imshow(img)
                plt.pause(0.2)
        # ---------------

        # Save Evaluation Data
        #   Don't change this code
        #   Evaluating MOT is not easy,
        #       for example it requires performing a matching beetween ground truth boxes and trackers
        # ---------------
        for trk in trackers:
            min_hits = 3
            if trk.hits >= min_hits or frame <= min_hits:
                # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
                d = Box3D.array2bbox(trk.x[:7].reshape((7, )))
                d = Box3D.bbox2array_raw(d)

                id_tmp = trk.ID
                ori_tmp, type_tmp, bbox2d_tmp_trk, conf_tmp = \
                    trk.info[0], "Car", trk.info[2:6], trk.info[6]

                # save in tracking format, for 3D MOT evaluation
                score_threshold = 0.5
                if conf_tmp >= score_threshold:
                    str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp,
                        type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3],
                        d[0], d[1], d[2], d[3], d[4], d[5], d[6], conf_tmp)
                    eval_file.write(str_to_srite)
        # ---------------

        print("\rFrame {:06d} done. Active Trackers {:02d}".format(frame, len(trackers)), end="")
    print()

def main():
    det_root = 'data/detection'
    oxt_root = 'data/oxts/training'
    calib_root = 'data/calib/training'
    all_sequences = os.listdir(det_root)
    import sys
    if len(sys.argv) == 2:
        all_sequences = [sys.argv[1]]

    seq_count = 0
    total_time = 0.0
    for seq_name in all_sequences:
        seq_start_time = time.time()
        print("Sequence {}".format(seq_name))

        seq_det_file = os.path.join(det_root, seq_name)
        seq_dets, flag = load_detection(seq_det_file)
        print("Loaded detections, shape: {}".format(seq_dets.shape))
        if not flag:
            print("Error: missing detection for seq {}".format(seq_name))
            continue

        oxts_path = os.path.join(oxt_root, seq_name)
        oxts_imu = load_oxts(oxts_path) # seq_frames x 4 x 4
        print("Loaded oxts, shape: {}".format(oxts_imu.shape))

        calib_path = os.path.join(calib_root, seq_name)
        calib = Calibration(calib_path)
        # print("Loaded calibration, shape: {}".format(calib))

        vis_path = "results/img_vis/{}/".format(seq_name[:-4])
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        eval_folder = "results/eval"
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        eval_path = "results/eval/{}".format(seq_name)
        eval_file = open(eval_path, 'w')

        if seq_name == "0000.txt":
            image_dir = "data/image_02/training/{}".format(seq_name[:-4])
        else:
            image_dir = None

        # can also get the max frame as max_frame = np.max(seq_dets[:, 0]) + 1
        max_frame = oxts_imu.shape[0]
        # Track
        '''
        seq_dets: the detected objects in 154 images: (1054, 15)
        max_frame: images number 154
        oxts_imu: OXTS文件中提供的GPS/IMU数据集合并转换为 (154, 4, 4)
        calib: A class with read-in data 相机标定： for instance: y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo (https://zhuanlan.zhihu.com/p/99114433)
        '''
        track_sequence(seq_dets, max_frame, oxts_imu, calib, vis_path, image_dir, eval_file)

        seq_count += 1
        seq_total_time = time.time() - seq_start_time
        total_time += seq_total_time

        print("Sequence: {}, Time: {:.2f}, Num Frames: {}".format(seq_count, seq_total_time, max_frame))
        # break

    print("Total time: {:.2f}".format(total_time))


if __name__=="__main__":
    main()
