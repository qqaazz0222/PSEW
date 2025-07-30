import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from skspatial.objects import Vector, Plane, Points
from rich.progress import track

def load_npy_data(dir: str):
    """
    지정된 디렉토리에서 모든 .npy 파일의 경로를 찾아 정렬된 리스트로 반환
    Args:
        dir (str): .npy 파일을 검색할 디렉토리 경로
    Returns:
        list: 디렉토리 내에서 발견된 .npy 파일들의 정렬된 전체 경로 리스트
              해당 디렉토리에 .npy 파일이 없으면 빈 리스트가 반환
    """
    npy_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
    npy_files.sort()
    return npy_files

def compute_rotation(mesh_dir: str):
    """
    주어진 메쉬 데이터의 평면을 계산하고, 월드 평면과의 회전 벡터를 반환하는 함수

    Args:
        mesh_dir (str): 메쉬 데이터가 저장된 디렉토리 경로
    Returns:
        Vector: 월드 평면과의 회전 벡터
    """
    try:
        mesh_path = load_npy_data(mesh_dir)
        
        bottom_points = []
        
        for idx in range(len(mesh_path)):
            meshes = np.load(mesh_path[idx], allow_pickle=True)
            try:
                for mesh in meshes:
                    max_y_point = max(mesh, key=lambda point: point[1])
                    bottom_points.append(max_y_point)
            except:
                continue
                
        points = Points(bottom_points)
        mesh_plane = Plane.best_fit(points)
        mesh_plane_normal = mesh_plane.normal
        
        world_plane_normal = Vector([0, -1, 0])
        
        # print(f"Mesh plane normal: {mesh_plane_normal}")
        # print(f"World plane normal: {world_plane_normal}")
        
        rotation = mesh_plane_normal - world_plane_normal
        # print(f"Rotation vector: {rotation}")
        
        print(f"Mesh Plane Normal: {mesh_plane_normal}")
        print(f"World Plane Normal: {world_plane_normal}")
        print(f"Rotation Vector: {rotation}")
        return rotation
    except Exception as e:
        print(f"Error computing rotation: {e}")
        return Vector([0, 0, 0])
    
    
