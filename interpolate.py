import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def load_npy_data(dir: str):
    npy_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
    npy_files.sort()
    return npy_files

def find_missing_objects(prev_objs, curr_objs, distance_fn=None, threshold=np.inf):
    distance_fn = distance_fn or (lambda a, b: np.linalg.norm(a - b))

    M, N = len(prev_objs), len(curr_objs)
    D = np.empty((M, N), dtype=float)
    for i, p in enumerate(prev_objs):
        for j, c in enumerate(curr_objs):
            D[i, j] = distance_fn(p, c)

    row_ind, col_ind = linear_sum_assignment(D)
    matched_pairs = [
        (i, j) for i, j in zip(row_ind, col_ind) if D[i, j] <= threshold
    ]
    matched_prev_idx = {i for i, _ in matched_pairs}
    missing_indices = [i for i in range(M) if i not in matched_prev_idx]
    return matched_pairs, missing_indices

def interpolate(mesh_dir: str, face_dir: str):
    mesh_path = load_npy_data(mesh_dir)
    face_path = load_npy_data(face_dir)

    init_num = np.load(mesh_path[0]).shape[0]
    
    for i in range(len(mesh_path)):
        mesh = np.load(mesh_path[i], allow_pickle=True)
        face = np.load(face_path[i], allow_pickle=True)
        interpolated_mesh = []
        interpolated_face = []
        
        if mesh.shape[0] >= init_num:
            continue
        else:
            pre_mesh = np.load(mesh_path[i - 1], allow_pickle=True)
            pre_face = np.load(face_path[i - 1], allow_pickle=True)
            pairs, missing = find_missing_objects(pre_mesh, mesh, )
            interpolated_mesh = [None for _ in range(pre_mesh.shape[0])]
            interpolated_face = [None for _ in range(pre_face.shape[0])]
            
            for pre_idx, cur_idx in pairs:
                interpolated_mesh[pre_idx] = mesh[cur_idx]
                interpolated_face[pre_idx] = face[cur_idx]
                
            for missing_idx in missing:
                interpolated_mesh[missing_idx] = pre_mesh[missing_idx]
                interpolated_face[missing_idx] = pre_face[missing_idx]
                
            interpolated_mesh = np.array(interpolated_mesh)
            interpolated_face = np.array(interpolated_face)
            
            print(interpolated_mesh.shape, interpolated_face.shape)
            np.save(os.path.join(mesh_dir, f"{i:08d}.npy"), interpolated_mesh, allow_pickle=True)
            np.save(os.path.join(face_dir, f"{i:08d}.npy"), interpolated_face, allow_pickle=True)

if __name__ == "__main__":
    mesh_dir = "data/working/demo_2/mesh"
    face_dir = "data/working/demo_2/face"
    interpolate(mesh_dir, face_dir)
    print("Interpolation completed.")