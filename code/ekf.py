import scipy
import numpy as np
from utils import *

class EKFSLAM:
    def __init__(self, 
                 v_noise, # linear velocity noise
                 w_noise, # angular velocity noise
                 landmark_noise, # landmark noise
                 m_noise, # measurement noise
                 n_ft, # number of features
                 k_l, # left camera calibration matrix
                 k_r, # right camera calibration matrix
                 extL_T_imu, # left imu to camera transform
                 extR_T_imu, # right imu to camera transform
                 num_seen_threshold = 5,
                 mapping_only = False,
                 ):
        # camera related parameters
        self.k_l = k_l
        self.k_r = k_r
        self.extL_T_imu = extL_T_imu
        self.extR_T_imu = extR_T_imu

        self.f_u_l = self.k_l[0, 0]
        self.f_v_l = self.k_l[1, 1]
        self.f_u_r = self.k_r[0, 0]
        self.f_v_r = self.k_r[1, 1]
        self.c_u_l = self.k_l[0, 2]
        self.c_v_l = self.k_l[1, 2]
        self.c_u_r = self.k_r[0, 2]
        self.c_v_r = self.k_r[1, 2]

        o_T_cam = np.zeros((4, 4))
        o_T_cam[0, 1] = -1
        o_T_cam[1, 2] = -1
        o_T_cam[2, 0] = 1
        o_T_cam[-1, -1] = 1
        self.o_T_cam = o_T_cam

        self.delta_p = self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.extR_T_imu) @ np.array([[0], [0], [0], [1]])
        self.delta_p_1 = - self.delta_p[0, 0]
        self.delta_p_2 = - self.delta_p[1, 0]
        self.delta_p_3 = - self.delta_p[2, 0]

        # noise parameters
        self.v_noise = v_noise
        self.w_noise = w_noise
        self.landmark_noise = landmark_noise
        self.m_noise = m_noise

        # initializing landmark related parameters
        self.n_ft = n_ft
        self.landmarks = np.zeros((3, n_ft))
        self.landmarks_visibility = np.zeros((n_ft,)).astype(bool)

        # robot state
        self.robot_pose = np.eye(4)
        self.Sigma = np.zeros((3 * self.n_ft + 6, 3 * self.n_ft + 6))
        self.Sigma[:-6, :-6] = np.eye(3 * self.n_ft) * self.landmark_noise ** 2

        
        self.num_seen_threshold = num_seen_threshold
        self.mapping_only = mapping_only
    
    def pose_predict(self, u, dt):
        """
        Predict the robot pose
        """
        self.robot_pose = self.robot_pose @ scipy.linalg.expm(dt * axangle2twist(u)[0])

    def predict(self, u, dt):
        """
        Predict the robot pose and covariance matrix
        """
        # get current pose
        self.pose_predict(u, dt)

        if not self.mapping_only:
            F = np.eye(3 * self.n_ft + 6)
            F[-6:, -6:] = scipy.linalg.expm(-dt * axangle2adtwist(u)[0])
            
            W = np.eye(6)
            W[:3, :3] *= self.v_noise ** 2
            W[3:, 3:] *= self.w_noise ** 2

            self.Sigma = F @ self.Sigma @ F.T
            self.Sigma[-6:, -6:] += W

    def update(self, ft):
        u1, v1, u2, v2 = ft # shape: (n_ft, T)
        u1, u2, v1, v2 = u1.T, u2.T, v1.T, v2.T # shape: (T, n_ft)
        # pix_coord_l = np.einsum('ij,mnj->mni', np.linalg.inv(self.k_l), np.stack([u1, v1, np.ones_like(u1)], axis=-1)) # shape: (T, n_ft, 3)
        # pix_coord_r = np.einsum('ij,mnj->mni', np.linalg.inv(self.k_r), np.stack([u2, v2, np.ones_like(u2)], axis=-1)) # shape: (T, n_ft, 3)

        # A_l = np.hstack([np.eye(3), np.zeros((3, 1))]) @ self.o_T_cam @ self.extL_T_imu # shape: (3, 4)
        # A_r = np.hstack([np.eye(3), np.zeros((3, 1))]) @ self.o_T_cam @ self.extR_T_imu # shape: (3, 4)
        # x_l = np.expand_dims(A_l[-1, :], (0, 1)) * np.repeat(np.expand_dims(pix_coord_l[..., 0], -1), 4, -1) # shape (T, n_ft, 4)
        # x_r = np.expand_dims(A_r[-1, :], (0, 1)) * np.repeat(np.expand_dims(pix_coord_r[..., 0], -1), 4, -1) # shape (T, n_ft, 4)

        # A_l = A_l[:2, :] - x_l[:, :, np.newaxis, :] # shape (T, n_ft, 2, 4)
        # A_r = A_r[:2, :] - x_r[:, :, np.newaxis, :] # shape (T, n_ft, 2, 4)

        # A = np.concatenate([A_l, A_r], axis=-2)[0] # shape (n_ft, 4, 4)

        # points_imu = self.solve_for_xyz(A) # shape (n_ft, 3)
        # points_cam_l = (self.extL_T_imu @ np.hstack([points_imu, np.ones((points_imu.shape[0], 1))]).T).T # shape (n_ft, 4)
        # depth_l = points_cam_l[:, 0]

        # new_idx = ~self.landmarks_visibility & (ft[:, :, 0] != -1).all(axis=0) & (depth_l < 50) & (depth_l > 0)
        # new_points_imu = points_imu[new_idx]
        # new_points_world = self.robot_pose @ np.hstack([new_points_imu, np.ones((new_points_imu.shape[0], 1))]).T # shape (4, n_ft)
        # new_points_world = new_points_world[:3, :].T # shape (n_ft, 3)

        z = self.f_u_l * (self.f_u_r * self.delta_p_1 + self.c_u_r * self.delta_p_3)
        z = z / (self.f_u_r * (u1 - self.c_u_l) - self.f_u_l * (u2 - self.c_u_r))
        x = (u1 - self.c_u_l) * z / self.f_u_l
        y = (v1 - self.c_v_l) * z / self.f_v_l
        points_cam = np.stack([x[0], y[0], z[0]], axis=-1) # shape (n_ft, 3)
        depth_l = z[0]

        new_idx = ~self.landmarks_visibility & (ft[:, :, 0] != -1).all(axis=0) & (depth_l < 100) & (depth_l > 0)
        new_points_cam = points_cam[new_idx]
        new_points_world = self.robot_pose @ np.linalg.inv(self.extL_T_imu) @ np.linalg.inv(self.o_T_cam) @ np.hstack([new_points_cam, np.ones((new_points_cam.shape[0], 1))]).T # shape (4, n_ft)
        new_points_world = new_points_world[:3, :].T # shape (n_ft, 3)

        # print(new_idx.shape, self.landmarks.shape, new_points_world.shape)
        print("number of new index: ", new_idx.sum(), 
              "number of detected features: ", (ft[:, :, 0] != -1).all(axis=0).sum(),
              "number of small depth: ", (depth_l < 50).sum(),
              "number of non-negative depth: ", (depth_l > 0).sum())
        self.landmarks[:, new_idx] = new_points_world.T
        self.landmarks_visibility[new_idx] = True

        detected_idx = (ft[:, :, 0] != -1).all(axis=0) & self.landmarks_visibility
        detected_landmarks = self.landmarks[:, detected_idx] # shape (3, n_detected)
        detected_idx = np.where(detected_idx)[0]
        detected_world = np.vstack([detected_landmarks, np.ones((1, detected_landmarks.shape[1]))]) # shape (4, n_detected)

        predicted_obs = self.cam_project(detected_world) # shape (4, n_detected)
        errors = ft[:, detected_idx, 0] - predicted_obs # shape (4, n_detected)
        errors = errors.T # shape (n_detected, 4)

        print("errors.shape: ", errors.shape)
        filtered_error_mask = np.abs(errors).sum(axis=1) < 20
        filtered_errors = errors[filtered_error_mask, :]
        detected_idx = detected_idx[filtered_error_mask]
        detected_world = detected_world[:, filtered_error_mask]
        num_seen = filtered_errors.shape[0]
        errors = filtered_errors.flatten() # shape (4 * n_detected,)
        # print(num_seen)
        if num_seen <= self.num_seen_threshold:
            return

        H = np.zeros((4 * num_seen, 3 * self.n_ft + 6))

        for i in range(num_seen):
            landmark_idx = detected_idx[i]

            P = np.zeros((3, 4))
            P[0, 0] = P[1, 1] = P[2, 2] = 1

            H[4*i: 4*i+2, 3*landmark_idx: 3*landmark_idx+3] = \
                self.k_l[:2, :] @ projectionJacobian((self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.robot_pose) \
                                                      @ detected_world[:, i][:, np.newaxis]).T)[0, :3, :3] \
                                                      @ P @ self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.robot_pose) @ P.T

            H[4*i+2: 4*i+4, 3*landmark_idx: 3*landmark_idx+3] = \
                self.k_r[:2, :] @ projectionJacobian((self.o_T_cam @ self.extR_T_imu @ np.linalg.inv(self.robot_pose) \
                                                      @ detected_world[:, i][:, np.newaxis]).T)[0, :3, :3] \
                                                      @ P @ self.o_T_cam @ self.extR_T_imu @ np.linalg.inv(self.robot_pose) @ P.T
            
            if not self.mapping_only:
                H[4*i: 4*i+2, -6:] = -self.k_l[2, :] @ projectionJacobian((self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.robot_pose) \
                                                                           @ detected_world[:, i][:, np.newaxis]).T)[0, :3, :] \
                                                                           @ odot((self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.robot_pose) \
                                                                           @ detected_world[:, i][:, np.newaxis]).T)[0]

                H[4*i+2: 4*i+4, -6:] = -self.k_r[2, :] @ projectionJacobian((self.o_T_cam @ self.extR_T_imu @ np.linalg.inv(self.robot_pose) \
                                                                           @ detected_world[:, i][:, np.newaxis]).T)[0, :3, :] \
                                                                           @ odot((self.o_T_cam @ self.extR_T_imu @ np.linalg.inv(self.robot_pose) \
                                                                           @ detected_world[:, i][:, np.newaxis]).T)[0]
                
        R = np.eye(4 * num_seen) * self.m_noise ** 2

        if self.mapping_only:
            K = self.Sigma[:-6, :-6] @ H[:, :-6].T @ np.linalg.inv(H[:, :-6] @ self.Sigma[:-6, :-6] @ H[:, :-6].T + R)
            to_update = np.zeros(3 * detected_idx.shape[0]).astype(int)
            to_update[0::3] = detected_idx.flatten() * 3
            to_update[1::3] += to_update[0::3] + 1
            to_update[2::3] += to_update[0::3] + 2

            cov_to_update = self.Sigma[to_update[:, None], to_update]

            sub_H = H[:, to_update]
            sub_K = K[to_update]

            self.Sigma[to_update[:, None], to_update] = (np.eye(cov_to_update.shape[0]) - sub_K @ sub_H) @ cov_to_update
            self.landmarks[:, detected_idx] = self.landmarks[:, detected_idx] + (K @ errors).reshape(-1, 3).T[:, detected_idx]
        else:
            K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + R)
            to_update = np.concatenate([
                np.zeros(3 * detected_idx.shape[0]),
                np.arange(3 * self.n_ft, 3 * self.n_ft + 6)
            ]).astype(int)
            to_update[0:-6:3] = detected_idx.flatten() * 3
            to_update[1:-6:3] += to_update[0:-6:3] + 1
            to_update[2:-6:3] += to_update[0:-6:3] + 2

            cov_to_update = self.Sigma[to_update[:, None], to_update]

            sub_H = H[:, to_update]
            sub_K = K[to_update]

            self.Sigma[to_update[:, None], to_update] = (np.eye(cov_to_update.shape[0]) - sub_K @ sub_H) @ cov_to_update
            self.robot_pose = self.robot_pose @ scipy.linalg.expm(axangle2twist((K @ errors)[-6:][np.newaxis, :])[0])
            self.landmarks[:, detected_idx] = self.landmarks[:, detected_idx] + (K @ errors)[:-6].reshape(-1, 3).T[:, detected_idx]


    def cam_project(self, points_world):
        """
        Project 3D points in world frame to 2D points in camera frame of left and right cameras

        Args:
            points_world (np.ndarray): 3D points in world frame, shape (4, n)

        Returns:
            np.ndarray: 2D points in camera frame, shape (4, n), 
            in which the first two rows are pixel coordinates in the left camera,
            and the last two rows are pixel coordinates in the right camera
        """
        predicted_left = np.hstack([np.eye(3), np.zeros((3, 1))]) @ self.o_T_cam @ self.extL_T_imu @ np.linalg.inv(self.robot_pose) @ points_world
        predicted_left = self.k_l @ projection(predicted_left.T).T
        predicted_right = np.hstack([np.eye(3), np.zeros((3, 1))]) @ self.o_T_cam @ self.extR_T_imu @ np.linalg.inv(self.robot_pose) @ points_world
        predicted_right = self.k_r @ projection(predicted_right.T).T
        return np.vstack([predicted_left[:2, :], predicted_right[:2, :]])
        

    @staticmethod
    def solve_for_xyz(A):
        """
        Given a batch of 4x4 matrices A (shape: [batch_size, 4, 4]),
        solve for [x, y, z] in A * [x, y, z, 1]^T = 0.
        
        The equation is rewritten as:
            B * [x, y, z]^T = -c,
        where B consists of the first three columns of A and c is the last column.
        
        Parameters:
            A (np.ndarray): A numpy array of shape (batch_size, 4, 4).
        
        Returns:
            np.ndarray: A tensor of shape (batch_size, 3) containing the solution [x, y, z]
                        for each batch element.
        """
        batch_size = A.shape[0]
        solutions = []
        for i in range(batch_size):
            # Extract B (first 3 columns) and c (last column) for the i-th matrix.
            B = A[i, :3, :3]       # Shape: (3, 3)
            c = A[i, :3, 3]        # Shape: (3,)
            
            # Solve B * [x,y,z]^T = -c using least squares.
            # xyz, residuals, rank, s = np.linalg.lstsq(B, -c, rcond=None)
            xyz = np.linalg.solve(B, -c)
            solutions.append(xyz)
        
        return np.stack(solutions, axis=0)
