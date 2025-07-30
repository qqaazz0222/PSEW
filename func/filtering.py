import numpy as np
import torch

class KalmanFilter:
    def __init__(self, dim_state, dim_obs, process_noise=1e-4, observation_noise=1e-2):
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # State vector [position, velocity]
        self.x = np.zeros(dim_state)
        
        # State covariance matrix
        self.P = np.eye(dim_state) * 1e3 # Start with high uncertainty
        
        # State transition matrix (assumes constant velocity model)
        self.F = np.eye(dim_state)
        if dim_state == 2 * dim_obs:
            # position_next = position_prev + velocity_prev * dt (dt=1 frame)
            for i in range(dim_obs):
                self.F[i, i + dim_obs] = 1.0
        
        # Observation matrix (we only observe the position part of the state)
        self.H = np.zeros((dim_obs, dim_state))
        for i in range(dim_obs):
            self.H[i, i] = 1.0
        
        # Process noise covariance (uncertainty in the motion model)
        self.Q = np.eye(dim_state) * process_noise
        
        # Observation noise covariance (uncertainty in the measurement)
        self.R = np.eye(dim_obs) * observation_noise
        
        self.initialized = False
    
    def predict(self):
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z, is_angle=False):
        """Update the state with a new observation."""
        if not self.initialized:
            # Initialize state with the first observation
            if self.dim_state == 2 * self.dim_obs:
                self.x[:self.dim_obs] = z
                self.x[self.dim_obs:] = 0  # Initial velocity is zero
            else:
                self.x = z
            self.initialized = True
            return self.x[:self.dim_obs]
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # --- START: MODIFICATION FOR ROTATION ---
        # If the data is an angle, wrap the innovation to the range [-pi, pi]
        # This ensures the filter takes the shortest path for rotation.
        if is_angle:
            y = (y + np.pi) % (2 * np.pi) - np.pi
        # --- END: MODIFICATION FOR ROTATION ---

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        # Use pseudo-inverse for better numerical stability
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance matrix
        I = np.eye(self.dim_state)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:self.dim_obs]

def filtering_parameters(parameter_list: list) -> list:
    """
    Apply Kalman filter to parameter list for natural behavior.
    
    Args:
        parameter_list: List of parameter dictionaries for each frame.
    
    Returns:
        List of filtered parameter dictionaries.
    """
    print("Filtering parameters...", end=' ')
    if not parameter_list:
        return []
    
    filters = {}
    filtered_parameters = []
    
    filterable_keys = ['loc', 'offset', 'rotvec', 'expression', 'shape']
    
    for params in parameter_list:
        if not isinstance(params, dict):
            # If tracking fails, the entry might not be a dict.
            # Append as is and continue.
            filtered_parameters.append(params)
            continue
            
        filtered_params = params.copy()
        
        for key in filterable_keys:
            if key not in params or not isinstance(params[key], torch.Tensor):
                continue
                
            param_tensor = params[key]
            param_np = param_tensor.detach().cpu().numpy()
            
            if key not in filters:
                filters[key] = {}
            
            for instance_idx in range(param_np.shape[0]):
                if instance_idx not in filters[key]:
                    # --- START: SIMPLIFIED FILTER INITIALIZATION ---
                    # Use a unified initialization logic.
                    # The state dimension is always twice the observation dimension.
                    obs_dim = param_np[instance_idx].size
                    state_dim = obs_dim * 2
                    
                    # Adjust noise parameters based on the key
                    if key == 'loc':
                        # Location needs to be responsive, so less smoothing.
                        filters[key][instance_idx] = KalmanFilter(
                            state_dim, obs_dim, process_noise=1e-2, observation_noise=1e-1
                        )
                    elif key == 'rotvec':
                        # --- START: MODIFICATION FOR RESPONSIVENESS ---
                        # Increased process_noise to allow faster response to changes
                        # Reduced observation_noise to trust measurements more
                        filters[key][instance_idx] = KalmanFilter(
                            state_dim, obs_dim, process_noise=1e-3, observation_noise=1e-1
                        )
                        # --- END: MODIFICATION FOR RESPONSIVENESS ---
                    else: # expression, shape, offset
                        # These parameters change slowly, but still increase responsiveness
                        filters[key][instance_idx] = KalmanFilter(
                            state_dim, obs_dim, process_noise=1e-4, observation_noise=1e-2
                        )
                    # --- END: SIMPLIFIED FILTER INITIALIZATION ---

                kf = filters[key][instance_idx]
                kf.predict()
                
                obs = param_np[instance_idx].flatten()
                
                # --- START: MODIFICATION FOR ROTATION ---
                # Pass `is_angle=True` for 'rotvec' to handle wrapping correctly.
                is_rotation = (key == 'rotvec')
                filtered_obs = kf.update(obs, is_angle=is_rotation)

                # For rotation, also ensure the magnitude doesn't grow uncontrollably.
                # This prevents the spinning artifact.
                if is_rotation:
                    # Reshape to (num_joints, 3)
                    rotvecs = filtered_obs.reshape(-1, 3)
                    magnitudes = np.linalg.norm(rotvecs, axis=1, keepdims=True)
                    # Limit rotation to a maximum of 180 degrees (pi radians)
                    mask = magnitudes > np.pi
                    if np.any(mask):
                        # Squeeze mask to correctly index the rotvecs array
                        rotvecs[mask.squeeze(), :] = rotvecs[mask.squeeze(), :] * (np.pi / magnitudes[mask.squeeze(), :])
                    filtered_obs = rotvecs.flatten()
                # --- END: MODIFICATION FOR ROTATION ---

                param_np[instance_idx] = filtered_obs.reshape(param_np[instance_idx].shape)
            
            filtered_params[key] = torch.from_numpy(param_np).to(param_tensor.device).type(param_tensor.dtype)
        
        filtered_parameters.append(filtered_params)
    print('Done!')
    return filtered_parameters


def filtering_humans(human_list: list) -> list:
    """
    Apply Kalman filter to the mean x, y, and depth (z) of each human's v3d to reduce jitter.
    
    Args:
        human_list: List of humans data for each frame.
    
    Returns:
        List of filtered humans data.
    """
    if not human_list:
        return []
    
    # Dictionary to store filters for each person (indexed by person order in frame)
    xyz_filters = {}
    filtered_human_list = []
    
    for frame_humans in human_list:
        if not isinstance(frame_humans, list):
            filtered_human_list.append(frame_humans)
            continue
            
        filtered_humans = []
        
        for person_idx, human in enumerate(frame_humans):
            if not isinstance(human, dict) or 'v3d' not in human:
                filtered_humans.append(human)
                continue
                
            # Calculate mean x, y, z of v3d
            v3d = human['v3d']
            if not isinstance(v3d, torch.Tensor) or v3d.shape[1] < 3:
                filtered_humans.append(human)
                continue
                
            mean_xyz = v3d.mean(dim=0)[:3].cpu().numpy()  # [mean_x, mean_y, mean_z]
            
            # Initialize filter for this person if not exists
            if person_idx not in xyz_filters:
                # Configure Kalman filter for 3D position [x, y, z]
                # Higher observation_noise makes filter trust observations less
                # Lower process_noise makes filter trust predictions more
                xyz_filters[person_idx] = KalmanFilter(
                    dim_state=6,  # [x, y, z, x_velocity, y_velocity, z_velocity]
                    dim_obs=3,    # [x, y, z]
                    process_noise=1e-6,  # Very low - trust the prediction model
                    observation_noise=1e-1  # Higher - less trust in noisy observations
                )
            
            kf = xyz_filters[person_idx]
            kf.predict()
            
            # Update with observed mean xyz
            filtered_xyz = kf.update(mean_xyz)
            
            # Apply the xyz adjustment to all v3d points
            xyz_adjustment = filtered_xyz - mean_xyz
            human_copy = human.copy()
            adjusted_v3d = v3d.clone()
            adjusted_v3d[:, :3] += torch.from_numpy(xyz_adjustment).to(v3d.device)
            human_copy['v3d'] = adjusted_v3d
            
            filtered_humans.append(human_copy)
        
        filtered_human_list.append(filtered_humans)
    
    return filtered_human_list