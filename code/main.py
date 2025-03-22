import os
from io import BytesIO
import numpy as np
from utils import *
from ekf import EKFSLAM
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio

class SLAMConfig:
    def __init__(self, dataset_path, downsample, v_noise, w_noise, landmark_noise, m_noise, show_ori):
        self.dataset_path = dataset_path
        self.downsample = downsample
        self.v_noise = v_noise
        self.w_noise = w_noise
        self.landmark_noise = landmark_noise
        self.m_noise = m_noise
        self.show_ori = show_ori
        
slam_config00 = SLAMConfig(
    dataset_path = "../data/dataset00/dataset00.npy",
    downsample = 4,
    v_noise = np.sqrt(1e-5),
    w_noise = np.sqrt(1e-5),
    landmark_noise = np.sqrt(2),
    m_noise = np.sqrt(2),
    show_ori = False
)

slam_config01 = SLAMConfig(
    dataset_path = "../data/dataset01/dataset01.npy",
    downsample = 10,
    v_noise = np.sqrt(1e-5),
    w_noise = np.sqrt(1e-5),
    landmark_noise = np.sqrt(2),
    m_noise = np.sqrt(2),
    show_ori = False
)

slam_config02 = SLAMConfig(
    dataset_path = "../data/dataset02/dataset02.npy",
    downsample = 4,
    v_noise = np.sqrt(1e-5),
    w_noise = np.sqrt(1e-5),
    landmark_noise = np.sqrt(2),
    m_noise = np.sqrt(4),
    show_ori = False
)

def main(config, only_mapping = False, only_prediction = False):
    dataset_path = config.dataset_path
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(dataset_path)
    T = timestamps.shape[0]
    feature = features[:, ::config.downsample, :]
    ekf = EKFSLAM(v_noise = config.v_noise, # linear velocity noise
                  w_noise = config.w_noise, # angular velocity noise
                  landmark_noise = config.landmark_noise, # landmark noise
                  m_noise = config.m_noise, # measurement noise
                  n_ft = feature.shape[1], # number of features
                  k_l = K_l, # left camera calibration matrix
                  k_r = K_r, # right camera calibration matrix
                  extL_T_imu = extL_T_imu, # left imu to camera transform
                  extR_T_imu = extR_T_imu, # right imu to camera transform
                  num_seen_threshold = 5, # number of seen threshold
                  mapping_only = only_mapping) # mapping only
    robot_poses = []
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    figure_list = []
    for t in tqdm(range(1, T), desc="Processing"):
        u = np.concatenate([v_t[t], w_t[t]])
        dt = timestamps[t] - timestamps[t-1]
        ekf.predict(u[np.newaxis, :], dt)
        if not only_prediction:
            ekf.update(feature[:, :, t][:, :, np.newaxis])
        robot_poses.append(ekf.robot_pose.copy())
        if t % 5 == 1:
            ax.clear()
            visualize_slam(ax, np.array(robot_poses).transpose(1, 2, 0), ekf.landmarks, show_ori=config.show_ori)
            ax.set_facecolor('white')
            plt.draw()
            plt.pause(0.001)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=80, facecolor='white', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            image = imageio.imread(buf)
            image = image[..., :3]  # Remove alpha channel
            figure_list.append(image)
            
    os.makedirs(os.path.join(os.getcwd(), "results"), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "results", config.dataset_path.split("/")[-1].split(".")[0]), exist_ok=True)
    if only_prediction:
        path = os.path.join(os.getcwd(), "results", config.dataset_path.split("/")[-1].split(".")[0], "prediction_only.gif")
    else:
        if only_mapping:
            path = os.path.join(os.getcwd(), "results", config.dataset_path.split("/")[-1].split(".")[0], "mapping_only.gif")
        else:
            path = os.path.join(os.getcwd(), "results", config.dataset_path.split("/")[-1].split(".")[0], "slam_new.gif")
    
    with imageio.get_writer(path, mode='I', fps=50) as writer:
        for image in figure_list:
            writer.append_data(image)
    

if __name__ == '__main__':
    
	# (a) IMU Localization via EKF Prediction
    main(slam_config00, only_mapping=True, only_prediction=True)

	# (b) Landmark Mapping via EKF Update
    main(slam_config00, only_mapping=True, only_prediction=False)
    
	# (c) Visual-Inertial SLAM
    main(slam_config00, only_mapping=False, only_prediction=False)

	# (a) IMU Localization via EKF Prediction
    main(slam_config01, only_mapping=True, only_prediction=True)

	# (b) Landmark Mapping via EKF Update
    main(slam_config01, only_mapping=True, only_prediction=False)
    
	# (c) Visual-Inertial SLAM
    main(slam_config01, only_mapping=False, only_prediction=False)
    
	# (a) IMU Localization via EKF Prediction
    main(slam_config02, only_mapping=True, only_prediction=True)

	# (b) Landmark Mapping via EKF Update
    main(slam_config02, only_mapping=True, only_prediction=False)
    
	# (c) Visual-Inertial SLAM
    main(slam_config02, only_mapping=False, only_prediction=False)
	


