import os 
os.environ["PYOPENGL_PLATFORM"] = "egl" # 서버 환경에서 OpenGL을 사용하기 위한 설정
os.environ['EGL_DEVICE_ID'] = '0' # EGL 디바이스 ID 설정

from argparse import ArgumentParser
import torch
import random
import trimesh
import numpy as np
from PIL import Image, ImageOps
from model import Model
from models.checker import check_smplx
from func.tracking import tracking_parameters
from func.interpolation import interpolating_parameters
from func.filtering import filtering_parameters, filtering_humans
from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, create_scene, demo_color as color, CACHE_DIR_MULTIHMR
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.video import extract_video, compress_video
from utils.visualization import visualize_loc, visualize_transl
from utils.rotate import compute_rotation
from utils.capture import capture
from utils.logger import console_banner, console_process, console_args, console_videos
from rich.console import Console
from rich.progress import track

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

"""

    ███████▙╗  ▟███████╗  ████████╗  ██╗ ██╗ ██╗
    ██╔═══██║  ██╔═════╝  ██╔═════╝  ██║ ██║ ██║
    ███████▛║  ▜██████▙╗  ████████╗  ██║ ██║ ██║
    ██╔═════╝        ██║  ██╔═════╝  ██║ ██║ ██║
    ██║        ███████▛║  ████████╗  ▜███▛▜███▛║
    ╚═╝        ╚═══════╝  ╚═══════╝  ╚════╩════╝
      P o s e  S c e n e  E v e r y  W h e r e    
      
"""

def get_args():
    """
    프레임워크 실행을 위한 인자들을 설정하고 반환하는 함수
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='multiHMR_896_L')
    parser.add_argument("--input_dir", type=str, default='data/input')
    parser.add_argument("--working_dir", type=str, default='data/working')
    parser.add_argument("--output_dir", type=str, default='data/output')
    parser.add_argument("--camera_config", type=str, default='data/camera_config.json',)
    parser.add_argument("--save_overlay", type=int, default=1, choices=[0,1])
    parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
    parser.add_argument("--det_thresh", type=float, default=0.3)
    parser.add_argument("--nms_kernel_size", type=float, default=3)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--distance", type=int, default=0, choices=[0,1])
    parser.add_argument("--unique_color", type=int, default=1, choices=[0,1])
    parser.add_argument("--use_checkpoint", type=int, default=1, choices=[0,1])
    parser.add_argument("--draw_outline", type=int, default=1, choices=[0,1])
    args = parser.parse_args()
    dict_args = vars(args)
    
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args, dict_args

def init_directories(args: ArgumentParser, video_name: str):
    """
    주어진 비디오 이름에 따라 작업 디렉토리와 하위 디렉토리를 초기화하는 함수

    Args:
        args (ArgumentParser): 명령줄 인자
        video_name (str): 비디오 이름
        
    Returns:
        tuple: 비디오 디렉토리, 오버레이 디렉토리, 메쉬 디렉토리, 얼굴 디렉토리, 장면 디렉토리, 캡처 디렉토리, 현재 출력 디렉토리
    """
    video_dir = os.path.join(args.working_dir, video_name)
    overlay_dir = os.path.join(video_dir, 'overlay')
    mesh_dir = os.path.join(video_dir, 'mesh')
    face_dir = os.path.join(video_dir, 'face')
    scene_dir = os.path.join(video_dir, 'scene')
    capture_dir = os.path.join(video_dir, 'capture')
    checkpoint_dir = os.path.join(video_dir, 'checkpoint')
    cur_output_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(capture_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(cur_output_dir, exist_ok=True)
    return video_dir, overlay_dir, mesh_dir, face_dir, scene_dir, capture_dir, checkpoint_dir, cur_output_dir

def open_image(img_path, img_size, device=torch.device('cuda')):
    """
    이미지를 열고, 크기를 조정하고 패딩하는 함수
    """
    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size))
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size))
    resize_img = np.asarray(img_pil)
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """
    카메라 파라미터를 설정하는 함수
    """
    K = torch.eye(3)
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """
    모델을 로드하는 함수
    """
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        
        try:
            os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            print("Please contact fabien.baradel@naverlabs.com or open an issue on the github repo")
            return 0

    # 가중치 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # 모델 인스턴스화
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # 모델에 가중치 적용
    model.load_state_dict(ckpt['model_state_dict'], strict=False)

    return model

def forward_model(model, input_image, camera_parameters, det_thresh=0.3, nms_kernel_size=1, pre_rotvec=None, pre_expression=None, pre_shape=None, pre_loc=None, pre_dist=None, pre_K_det=None):        
    """
    이미지와 카메라 파라미터를 입력으로 받아 모델을 통해 사람의 3D 메쉬를 예측하는 함수
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters,
                           pre_rotvec=pre_rotvec,
                           pre_expression=pre_expression,
                           pre_shape=pre_shape,
                           pre_loc=pre_loc,
                           pre_dist=pre_dist,
                           pre_K_det=pre_K_det)

    return humans

def pred_parameters(model, input_image, camera_parameters, det_thresh=0.3, nms_kernel_size=1):        
    """
    이미지와 카메라 파라미터를 입력으로 받아 모델을 통해 사람의 3D 메쉬를 예측하는 함수
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            parameters = model.pred_parameters(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters,)

    return parameters

def pred_human(model, parameters, camera_parameters):
    """
    모델과 사람의 3D 메쉬를 예측하는 함수
    """
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            humans = model.pred_human(parameters, 
                                   K=camera_parameters)
    return humans

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):
    """
    사람의 3D 메쉬를 이미지에 오버레이하는 함수
    """

    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])
    verts_list = [humans[j]['v3d'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

def load_npy_data(dir: str):
    """
    지정된 디렉토리에서 .npy 파일을 로드하고 정렬하는 함수
    """
    npy_data = []
    npy_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
    npy_files.sort()
    for n in npy_files:
        npy_data.append(np.load(n, allow_pickle=True))
    return npy_data

if __name__ == "__main__":
    console = Console()
    console = console.__class__(log_time=False)
    console_banner(console)
    
    # 인자 설정
    args, dict_args = get_args()
    console_args(console, dict_args)
    # GPU 사용 가능 여부 확인
    assert torch.cuda.is_available()
    # # SMPLX 파일 확인
    smplx_fn = check_smplx
    
    # 모델 로드
    console_process(console, "Initializing Model")
    model = load_model(args.model)
    img_size = model.img_size
    
    # 동영상 프레임 추출
    console_process(console, "Extracting Video Frames")
    frame_dict = extract_video(args.input_dir, args.working_dir, limit_frame=0)
    
    flag = True

    # 동영상별 처리
    for video_name, frame_list in frame_dict.items():
        if not flag:
            console_banner(console)
        
        console_videos(console, frame_dict.keys(), video_name)
        
        # 작업 디렉토리 설정
        video_dir, overlay_dir, mesh_dir, face_dir, scene_dir, capture_dir, checkpoint_dir, cur_output_dir = init_directories(args, video_name)
        
        parameter_list = []
        mesh_list = []
        
        human_list = []
        render_list = []
        pre_color = None
        
        scene_img_pil_visu_list = []
        scene_l_mesh_list = []
        scene_l_face_list = []
        scene_color_list = []
        
        # 카메라 파라미터 설정
        p_x, p_y = None, None
        K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
        
        console_process(console, f"Processing {video_name.upper()} Frames")
        
        flag, data = load_checkpoint(checkpoint_dir, 'parameters')
        
        if flag and args.use_checkpoint:
            parameter_list = data
            x, img_pil_nopad = open_image(frame_list[0], img_size)
        else:
            # 모델 예측 : SMPL-X 파라미터 예측
            for i, frame_path in enumerate(track(frame_list, description="Predcting Parameters")):
                # 이미지 열기 및 전처리
                x, img_pil_nopad = open_image(frame_path, img_size)
                parameters = pred_parameters(model, x, K, det_thresh=args.det_thresh, nms_kernel_size=args.nms_kernel_size)
                parameter_list.append(parameters)
                
            # 파라미터 저장(pickle)
            save_checkpoint(checkpoint_dir, 'parameters', parameter_list)
            
        # 파라미터 후처리
        parameter_list = tracking_parameters(parameter_list)
        parameter_list = interpolating_parameters(parameter_list)
        parameter_list = filtering_parameters(parameter_list)
        
        # 감지된 사람의 기준점(머리 중심) 시각화
        # visualize_loc(frame_list, parameter_list, img_pil_nopad.size)

        for i, frame_path in enumerate(track(frame_list, description="Predicting Meshes")):
            # 모델 예측 : 3D 메쉬 예측
            humans = pred_human(model, parameter_list[i], K)
            human_list.append(humans)      
            scene_img_pil_visu_list.append(img_pil_nopad)
            
        # 사람 메쉬 후처리
        human_list = filtering_humans(human_list)
        
        # 감지된 사람의 3D 위치 시각화(빨강: 머리 중심, 파랑: 골반 중심)
        # visualize_transl(human_list)
        
        # 장면 저장
        for i, humans in enumerate(track(human_list, description="Reappering Scene")):
            l_mesh = [humans[j]['v3d'].detach().cpu().numpy() for j in range(len(humans))]
            l_face = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]
            save_mesh_path = os.path.join(mesh_dir, f"{i:08d}.npy")
            save_face_path = os.path.join(face_dir, f"{i:08d}.npy")
            np.save(save_mesh_path, np.asarray(l_mesh), allow_pickle=True)
            np.save(save_face_path, np.asarray(l_face), allow_pickle=True)
        
        console_process(console, f"Computing Rotation {video_name.upper()} Meshes and Faces")
        rotation = compute_rotation(mesh_dir)
        
        scene_l_mesh_list = load_npy_data(mesh_dir)
        scene_l_face_list = load_npy_data(face_dir)
        
        for i, (img_pil_visu, l_mesh, l_face) in enumerate(zip(scene_img_pil_visu_list, scene_l_mesh_list, scene_l_face_list)):
            scene, mesh_color = create_scene(img_pil_visu, l_mesh, l_face, rot=rotation, color=pre_color, metallicFactor=0., roughnessFactor=0.5)
            scene.apply_transform(trimesh.transformations.euler_matrix(rotation[2], 0, 0, 'sxyz'))
            pre_color = mesh_color
            save_scene_path = os.path.join(scene_dir, f"{i:08d}.glb")
            scene.export(save_scene_path)
            
            render_list.append([i, save_mesh_path, save_scene_path])
            
        # 캡쳐
        console_process(console, f"Capturing {video_name.upper()} Frames")
        camera_config = args.camera_config
        angle_list = capture(camera_config, mesh_dir, scene_dir, capture_dir, args.draw_outline)
        
        # 동영상으로 변환
        console_process(console, f"Compressing {video_name.upper()} Frames")
        
        try:
            compress_video(capture_dir, cur_output_dir, angle_list)
        except:
            import json
            camera_config = json.load(open(args.camera_config, 'r'))
            angle_list = [(v_angle, h_angle) for v_angle in range(camera_config["rotation"]["vertical"]["angle_start"], camera_config["rotation"]["vertical"]["angle_end"] + 1, camera_config["rotation"]["vertical"]["angle_step"]) 
                        for h_angle in range(camera_config["rotation"]["horizontal"]["angle_start"], camera_config["rotation"]["horizontal"]["angle_end"] + 1, camera_config["rotation"]["horizontal"]["angle_step"])]
            compress_video(capture_dir, cur_output_dir, angle_list)
            
        flag = False