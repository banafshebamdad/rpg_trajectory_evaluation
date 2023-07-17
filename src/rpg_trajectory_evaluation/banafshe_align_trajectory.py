#
# Banafshe Bamdad
# Mo Jul 17, 2023 10:35:35 CET
#

import numpy as np 

import align_utils as au

#
# I use the output of banafshe_detect_board_charuco.cpp as grount-truth 
# to determine what scale the output of COLMAP shuld be multiplied 
# Poses calculated through COLMAP replaces "est" in the following formula
# s, R, t so that: gt = R * s * est + t
#

es_file = "/media/banafshe/Banafshe_2TB/ground_truth/colmap_workspace/my_evaluation/352-438/stamped_groundtruth.txt"
gt_file = "/media/banafshe/Banafshe_2TB/ground_truth/colmap_workspace/my_evaluation/352-438/stamped_traj_estimate.txt"

def rotation_2_array(data_file):
    """
        Converts ts Tx Ty Tz Qx Qy Qz Qw into a np.ndarray containing Qx Qy Qz Qw in each member
        Input: 
            data_file: path to thr file containing the above info.
        Output:
            a 2D numpy array
    """

    rotations = np.empty((0, 4))

    with open(data_file, 'r') as f:
        lines = f.readlines()

    for line in lines:

        pose_line = line.strip().split()
        rotaion_components = np.array([np.float64(pose_line[4]), np.float64(pose_line[5]), np.float64(pose_line[6]), np.float64(pose_line[7])])
        rotations = np.vstack((rotations, rotaion_components))

    return rotations

def translation_2_array(data_file):
    """
        Converts ts Tx Ty Tz Qx Qy Qz Qw into a np.ndarray containing Tx Ty Tz in each member
        Input: 
            data_file: path to thr file containing the above info.
        Output:
            a 2D numpy array
    """
    translations = np.empty((0, 3))

    with open(data_file, 'r') as f:
        lines = f.readlines()

    for line in lines:

        pose_line = line.strip().split()
        translation_components = np.array([np.float64(pose_line[1]), np.float64(pose_line[2]), np.float64(pose_line[3])])
        translations = np.vstack((translations, translation_components))

    return translations

# p_es: A numpy array containing estimated position vectors (translations) for the poses.
# p_gt: A numpy array containing ground truth position vectors for the poses.
# q_es: A numpy array containing estimated orientation quaternions for the poses.
# q_gt: A numpy array containing ground truth orientation quaternions for the poses.
# align_type: sim3 | se3 | posyaw | none
# align_num_frames: An optional parameter specifying the number of poses to align. If -1 (default), it indicates aligning all available poses.
p_es = translation_2_array(es_file)
p_gt = translation_2_array(gt_file)
q_es = rotation_2_array(es_file)
q_gt = rotation_2_array(gt_file)
align_type = "sim3"
align_num_frames= -1
scale, rotation, translation = au.alignTrajectory(p_es, p_gt, q_es, q_gt, align_type, align_num_frames)

print("Scale: ", scale)
print("Rotation: ", rotation)
print("Translation: ", translation)


