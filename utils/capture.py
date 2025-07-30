import os
import cv2
import json
import numpy as np
import trimesh
import pyrender
from PIL import Image
from skspatial.objects import Point, Plane
from rich.progress import track

# OpenGL 환경 설정
os.environ["PYOPENGL_PLATFORM"] = "egl"

def compute_focus_and_floor(mesh_path: list):
    """
    초점과 바닥면을 계산하는 함수
    """
    mesh = np.load(mesh_path, allow_pickle=True)
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

def capture(config: str, mesh_dir: str, scene_dir: str, capture_dir: str, draw_outline: int = 1):
    """
    GLB 파일을 렌더링하고 캡처하는 함수
    """
    def _load_data(mesh_dir: str, scene_dir: str):
        mesh_list = [os.path.join(mesh_dir, p) for p in os.listdir(mesh_dir) if p.endswith('.npy')]
        scene_list = [os.path.join(scene_dir, p) for p in os.listdir(scene_dir) if p.endswith('.glb')]
        mesh_list.sort()
        scene_list.sort()
        
        render_list = []
        for idx in range(len(mesh_list)):
            mesh_path = mesh_list[idx]
            scene_path = scene_list[idx]
            render_list.append((idx, mesh_path, scene_path))
            
        return render_list
            
    def _save_image(color, depth, save_path, draw_outline):
        # color 이미지에서 직접 컨투어 찾기
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        # 흰색 배경이 아닌 부분을 찾기 위해 임계값 적용
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        # 형태학적 연산으로 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 최소 면적 이상의 컨투어만 유지 (작은 노이즈 제거)
        min_area = 8  # 최소 면적 설정
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # depth 값에 따른 쉐도우 추가
        depth_min = np.min(depth[depth > 0])
        depth_max = np.max(depth)
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        depth_normalized = np.clip(depth_normalized, 0, 1)
        depth_grayscale = ((1 - depth_normalized) * 255).astype(np.uint8)  # 반전된 깊이 값
        depth_colored = np.stack([depth_grayscale] * 3, axis=-1)  # 3채널로 변환
        
        # 원본 색상과 깊이 기반 쉐도우 합성
        color = (0.5 * color + 0.5 * depth_colored).astype(np.uint8)
        
        # 흰색 배경 복원
        white_mask = np.all(color == [255, 255, 255], axis=-1)
        color[white_mask] = [255, 255, 255]
        
        # 컨투어를 검정색으로 두께 2로 그리기
        if draw_outline:
            cv2.drawContours(color, contours, -1, (0, 0, 0), 2)
        
        # 이미지 저장
        image = Image.fromarray(color)
        image.save(save_path)
        
    def _compute_floor(render_list: list):
        foot_points = []
        for _, _, glb_path in render_list:
            mesh = trimesh.load(glb_path, force='mesh', process=False)
            vertices = mesh.vertices
            # 각 메쉬에서 y값이 가장 작은 점 (발)을 찾습니다.
            if len(vertices) > 0:
                foot_point = vertices[np.argmin(vertices[:, 1])]
                foot_points.append(foot_point)
        
        if not foot_points:
            return None # 처리할 점이 없는 경우

        # 모든 발 지점들의 평균 y값을 계산하여 바닥면의 높이로 사용
        floor_y = np.mean([pt[1] for pt in foot_points])
        
        # 바닥면 평면 생성 (y = floor_y)
        # 평면의 법선 벡터는 (0, 1, 0)이고, 평면 위의 한 점은 (0, floor_y, 0)
        floor_plane = Plane(point=[0, floor_y, 0], normal=[0, 1, 0])
        return floor_plane
        
    
    camera_config = json.load(open(config, 'r'))
    w, h = camera_config['size']["width"], camera_config['size']["height"]
    distance = camera_config['distance']
    h_angle_start, h_angle_end, h_angle_step = camera_config["rotation"]["horizontal"]["angle_start"], camera_config["rotation"]["horizontal"]["angle_end"], camera_config["rotation"]["horizontal"]["angle_step"]
    v_angle_start, v_angle_end, v_angle_step = camera_config["rotation"]["vertical"]["angle_start"], camera_config["rotation"]["vertical"]["angle_end"], camera_config["rotation"]["vertical"]["angle_step"]
    
    # _, plane = compute_focus_and_floor(mesh_path)
    # normal_vector = plane.normal
    
    angle_list = [(v_angle, h_angle) for v_angle in range(v_angle_start, v_angle_end + 1, v_angle_step) for h_angle in range(h_angle_start, h_angle_end + 1, h_angle_step)]
    render_list = _load_data(mesh_dir, scene_dir)
    floor = _compute_floor(render_list)
    if floor is not None:
        # The normal vector of the computed floor plane
        source_normal = np.array(floor.normal, dtype=float)
        # The target normal vector (typically Y-axis up for a horizontal floor in a Y-up system)
        target_normal = np.array([0.0, 1.0, 0.0])
        if np.allclose(source_normal, target_normal):
            rot_floor = np.eye(4)
        elif np.allclose(source_normal, -target_normal):
            rot_floor = trimesh.geometry.align_vectors(source_normal, target_normal)
        else:
            rot_floor = trimesh.geometry.align_vectors(source_normal, target_normal)
    else:
        rot_floor = np.eye(4)
    
    for angle in angle_list:
        v_angle, h_angle = angle
        
        _, _, base_glb_path = render_list[0]
        base_mesh = trimesh.load(base_glb_path, force='mesh')
        base_centroid = base_mesh.centroid
    
        renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

        for frame_idx, mesh_path, glb_path in track(render_list, description=f"Capturing (v: {v_angle}, h: {h_angle})"):
            # GLB 파일 로드
            mesh = trimesh.load(glb_path, force='mesh', process=False)
            
            # 메시의 중심점을 계산
            mesh_centroid = mesh.centroid
            
            # 메시를 원점으로 이동
            mesh.apply_translation(-base_centroid)
            # 바닥면 정렬을 위한 회전 적용
            mesh.apply_transform(rot_floor)

            scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            camera_pose = np.eye(4)
            # 카메라 위치는 원점을 바라보도록 설정 (메시가 원점에 있으므로)
            camera_pose[:3, 3] = [0, 0, distance]
            
            scene.add(camera, pose=camera_pose)
            
            # 가로 회전 (원점 기준)
            rotation_matrix_horizontal = trimesh.transformations.rotation_matrix(
                angle=np.radians(180 - h_angle), direction=[0, 1, 0] # point 인자 제거 (원점 기준 회전)
            )
            # 세로 회전 (원점 기준)
            rotation_matrix_vertical = trimesh.transformations.rotation_matrix(
                angle=np.radians(-v_angle), direction=[1, 0, 0] # point 인자 제거 (원점 기준 회전)
            )
            # 두 회전 행렬을 결합
            rotation_matrix = np.dot(rotation_matrix_horizontal, rotation_matrix_vertical)
            # 메시는 이미 원점에 있으므로, 이 변환은 원점 주변 회전임
            mesh.apply_transform(rotation_matrix)
            
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(pyrender_mesh)
            color, depth = renderer.render(scene)
            
            save_path = os.path.join(capture_dir, f'{v_angle:03d}_{h_angle:03d}_{frame_idx:08d}.jpg')
            _save_image(color, depth, save_path, draw_outline)
        
        renderer.delete()
            
    return angle_list