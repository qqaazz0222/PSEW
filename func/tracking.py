import numpy as np
from scipy.optimize import linear_sum_assignment

def tracking_parameters(parameter_list: list) -> list:
    print("Tracking parameters...", end=' ')
    if len(parameter_list) <= 1:
        return parameter_list
    
    tracked_parameter_list = [parameter_list[0]]  # First frame as reference
    
    for frame_idx, current_frame in enumerate(parameter_list[1:], 1):
        prev_frame = tracked_parameter_list[-1]
        
        # Extract locations for distance calculation
        prev_locs = prev_frame['loc'].cpu().numpy()
        curr_locs = current_frame['loc'].cpu().numpy()
        
        # Calculate euclidean distance matrix
        distance_matrix = np.zeros((len(prev_locs), len(curr_locs)))
        for i, prev_loc in enumerate(prev_locs):
            for j, curr_loc in enumerate(curr_locs):
                # Check for NaN values and handle them
                if np.any(np.isnan(prev_loc)) or np.any(np.isnan(curr_loc)):
                    distance_matrix[i, j] = np.inf  # Use large distance for invalid locations
                else:
                    distance_matrix[i, j] = np.linalg.norm(prev_loc - curr_loc)
        
        # Apply Hungarian algorithm to find optimal assignment
        # Check if assignment is feasible (not all values are inf)
        if np.all(np.isinf(distance_matrix)) or distance_matrix.size == 0:
            # If all distances are infinite or matrix is empty, create identity mapping up to min length
            min_len = min(len(prev_locs), len(curr_locs))
            row_ind = np.arange(min_len)
            col_ind = np.arange(min_len)
        else:
            try:
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
            except ValueError:
                # Fallback to identity mapping if assignment fails
                min_len = min(len(prev_locs), len(curr_locs))
                row_ind = np.arange(min_len)
                col_ind = np.arange(min_len)
        
        # Create reordered frame with same length as previous frame
        reordered_frame = {}
        prev_length = len(prev_locs)
        
        for key, value in current_frame.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                # Create new tensor/array with same length as previous frame
                if len(value) > 0:
                    # Initialize with None-like values
                    if hasattr(value, 'device'):  # PyTorch tensor
                        new_tensor = value.new_zeros(prev_length, *value.shape[1:])
                        new_tensor.fill_(float('nan'))
                    else:  # NumPy array
                        new_tensor = np.full((prev_length, *value.shape[1:]), np.nan, dtype=value.dtype)
                    
                    # Fill assigned positions only for valid assignments
                    for i, j in zip(row_ind, col_ind):
                        if i < prev_length and j < len(value):
                            new_tensor[i] = value[j]
                    
                    # Fill unassigned positions with previous frame values
                    for i in range(prev_length):
                        if i not in row_ind:  # This position wasn't assigned
                            new_tensor[i] = prev_frame[key][i]  # Use previous value directly
                    
                    reordered_frame[key] = new_tensor
                else:
                    reordered_frame[key] = value
            else:
                reordered_frame[key] = value
        
        tracked_parameter_list.append(reordered_frame)
    print('Done!')
    return tracked_parameter_list
