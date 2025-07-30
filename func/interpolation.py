import torch

def parameters_identical(param1, param2):
    """Check if two parameter dictionaries are identical"""
    if param1 is None or param2 is None:
        return param1 is param2
    
    if set(param1.keys()) != set(param2.keys()):
        return False
    
    for key in param1.keys():
        if torch.is_tensor(param1[key]) and torch.is_tensor(param2[key]):
            if not torch.equal(param1[key], param2[key]):
                return False
        else:
            if param1[key] != param2[key]:
                return False
    
    return True


def interpolating_parameters(parameter_list: list, n: int = 4) -> list:
    print("Interpolating parameters...", end=' ')
    interpolated_parameters = parameter_list.copy()
    
    # Find segments that need interpolation (consecutive identical frames)
    segments_to_interpolate = []
    i = 0
    while i < len(parameter_list):
        if parameter_list[i] is not None:
            # Check for consecutive identical parameters
            start_idx = i
            while (i + 1 < len(parameter_list) and 
                   parameter_list[i + 1] is not None and
                   parameters_identical(parameter_list[i], parameter_list[i + 1])):
                i += 1
            
            # If we found identical consecutive frames, mark for interpolation
            if i > start_idx:
                segments_to_interpolate.append((start_idx, i))
        i += 1
    
    # Interpolate each identified segment
    for start_idx, end_idx in segments_to_interpolate:
        # Find previous valid parameter
        prev_param = None
        prev_idx = None
        for j in range(start_idx - 1, -1, -1):
            if (parameter_list[j] is not None and 
                not any(parameters_identical(parameter_list[j], parameter_list[k]) 
                       for k in range(start_idx, end_idx + 1))):
                prev_param = parameter_list[j]
                prev_idx = j
                break
        
        # Find next valid parameter
        next_param = None
        next_idx = None
        for j in range(end_idx + 1, len(parameter_list)):
            if (parameter_list[j] is not None and
                not any(parameters_identical(parameter_list[j], parameter_list[k]) 
                       for k in range(start_idx, end_idx + 1))):
                next_param = parameter_list[j]
                next_idx = j
                break
        
        # Perform interpolation for the segment
        if prev_param is not None and next_param is not None:
            # Linear interpolation across the segment
            total_frames = next_idx - prev_idx
            for i in range(start_idx, end_idx + 1):
                alpha = (i - prev_idx) / total_frames
                
                interpolated_param = {}
                for key in prev_param.keys():
                    if torch.is_tensor(prev_param[key]):
                        prev_val = prev_param[key]
                        next_val = next_param[key]
                        if torch.isnan(prev_val).any() or torch.isnan(next_val).any():
                            if not torch.isnan(prev_val).any():
                                interpolated_param[key] = prev_val
                            elif not torch.isnan(next_val).any():
                                interpolated_param[key] = next_val
                            else:
                                interpolated_param[key] = torch.zeros_like(prev_val)
                        else:
                            interpolated_param[key] = (1 - alpha) * prev_val + alpha * next_val
                    else:
                        interpolated_param[key] = prev_param[key]
                
                interpolated_parameters[i] = interpolated_param
        
        elif prev_param is not None:
            # Use previous parameter for all frames in segment
            for i in range(start_idx, end_idx + 1):
                interpolated_parameters[i] = prev_param.copy()
        
        elif next_param is not None:
            # Use next parameter for all frames in segment
            for i in range(start_idx, end_idx + 1):
                interpolated_parameters[i] = next_param.copy()
    
    # Handle None values with original logic
    for i, param in enumerate(parameter_list):
        if param is None:
            # Find available parameters for interpolation within n frames
            start_idx = max(0, i - n)
            available_params = []
            available_indices = []
            
            for j in range(start_idx, i):
                if parameter_list[j] is not None:
                    available_params.append(parameter_list[j])
                    available_indices.append(j)
            
            if not available_params:
                for j in range(0, i):
                    if parameter_list[j] is not None:
                        available_params.append(parameter_list[j])
                        available_indices.append(j)
            
            if not available_params:
                continue
            
            # Find next valid parameter
            next_param = None
            next_idx = None
            for j in range(i + 1, len(parameter_list)):
                if parameter_list[j] is not None:
                    next_param = parameter_list[j]
                    next_idx = j
                    break
            
            # Perform interpolation
            if len(available_params) == 1 and next_param is not None:
                prev_param = available_params[0]
                prev_idx = available_indices[0]
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                
                interpolated_param = {}
                for key in prev_param.keys():
                    if torch.is_tensor(prev_param[key]):
                        prev_val = prev_param[key]
                        next_val = next_param[key]
                        if torch.isnan(prev_val).any() or torch.isnan(next_val).any():
                            if not torch.isnan(prev_val).any():
                                interpolated_param[key] = prev_val
                            elif not torch.isnan(next_val).any():
                                interpolated_param[key] = next_val
                            else:
                                interpolated_param[key] = torch.zeros_like(prev_val)
                        else:
                            interpolated_param[key] = (1 - alpha) * prev_val + alpha * next_val
                    else:
                        interpolated_param[key] = prev_param[key]
                        
                interpolated_parameters[i] = interpolated_param
    
    print('Done!')
    return interpolated_parameters
