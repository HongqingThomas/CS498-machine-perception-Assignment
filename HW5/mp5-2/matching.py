import numpy as np
from matching_utils import iou
import copy

def data_association(dets, trks, threshold=-0.2, algm='greedy'):
    """
    Q1. Assigns detections to tracked object

    dets:       a list of Box3D object
    trks:       a list of Box3D object
    threshold:  only mark a det-trk pair as a match if their similarity is greater than the threshold
    algm:       for extra credit, implement the hungarian algorithm as well

    Returns 3 lists:
        matches, kx2 np array of match indices, i.e. [[det_idx1, trk_idx1], [det_idx2, trk_idx2], ...]
        unmatched_dets, a 1d array of indices of unmatched detections
        unmatched_trks, a 1d array of indices of unmatched trackers
    """
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d objects
    # iou() is an implementation of a 3D box GIoU

    matches = []
    unmatched_dets = []
    unmatched_trks = []
    # --------------------------- Begin your code here ---------------------------------------------

    if algm == 'greedy':
        dets_copy = copy.deepcopy(dets)
        trks_copy = copy.deepcopy(trks)
        for det_i, det in enumerate(dets_copy):
            best_iou = threshold
            best_matching_trk = None
            for trk_i, trk in enumerate(trks_copy):
                if trk != 0:
                    iou_score = iou(box_a = det, box_b = trk, metric = 'giou_3d')
                    if iou_score > best_iou:
                        best_iou = iou_score
                        best_matching_trk = trk
            if best_matching_trk == None:
                unmatched_dets.append(det_i)
            else:
                matches.append([det_i, trks_copy.index(best_matching_trk)])
                trks_copy[trks_copy.index(best_matching_trk)] = 0
        for unmached_trk in trks_copy:
            if unmached_trk != 0:
                unmatched_trks.append(trks_copy.index(unmached_trk))
        matches = np.array(matches)
        # print("matches:", matches)
    # --------------------------- End your code here   ---------------------------------------------

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)