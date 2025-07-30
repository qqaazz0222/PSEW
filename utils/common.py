import numpy as np
from skspatial.objects import Point, Plane

def compute_focus_and_floor(mesh_path: list):
    """
    초점과 바닥면을 계산하는 함수
    """
    mesh = np.load('npy.npy', allow_pickle=True)
    center_points = []
    bottom_points = []
    
    # 각 사람의 3D 관절 좌표에서 중심점과 바닥면을 계산
    for joints3d in mesh:
        max_y_idx = np.argmax(joints3d[:, 1])
        joint_center = np.mean(joints3d, axis=0) # 초점 계산을 위한 중심점
        joint_bottom = joints3d[max_y_idx] # 바닥면 계산을 위한 가장 아래쪽 점
        center_points.append(joint_center)
        bottom_points.append(joint_bottom)
    # 전역 초점 계산
    global_focus = np.mean(center_points, axis=0)
    # 전역 바닥면 계산
    max_bottom_point = Point(bottom_points[np.argmax([pt[1] for pt in bottom_points])])
    min_bottom_point = Point(bottom_points[np.argmin([pt[1] for pt in bottom_points])])
    mean_bottom_point = Point(np.mean(bottom_points, axis=0))
    global_plane = Plane.from_points(max_bottom_point, min_bottom_point, mean_bottom_point)
    
    return global_focus, global_plane