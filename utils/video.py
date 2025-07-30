import os
import cv2
import numpy as np
from PIL import Image
from rich.progress import track
import subprocess, shlex

def extract_video(input_dir: str, working_dir: str, limit_frame:int = 0):
    """
    동영상에서 프레임을 추출하여 이미지로 저장하고, 이미지 경로 리스트를 반환하는 함수
    """
    def _load_video(input_dir: str):
        """
        Input 디렉토리에서 동영상 파일을 로드하여 리스트로 반환하는 함수
        """
        suffixes = ('.mp4', '.avi', '.mov', '.mkv') # 지원하는 동영상 확장자
        video_list = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(suffixes) and file[0] != '.']
        
        # 동영상 파일이 없는 경우 예외 처리
        if len(video_list) == 0:
            assert NotImplementedError(f"동영상이 없습니다. (input_dir: {input_dir}, 지원 확장자: {suffixes})")
            
        return video_list
    
    def _extract_frames(video_path: str, working_dir: str):
        """
        동영상에서 프레임을 추출하여 이미지로 저장하고, 이미지 경로 리스트를 반환하는 함수
        """
        # 동영상 이름 및 디렉토리 설정
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.join(working_dir, video_name)
        frame_dir = os.path.join(video_dir, 'frames')
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        
        # 프레임 추출
        frame_list = []

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if limit_frame != 0 and frame_count >= limit_frame:
                break
            
            if not ret:
                break
            # 프레임을 00000000.jpg 형식으로 저장
            frame_path = os.path.join(frame_dir, f"{frame_count:08d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_list.append(frame_path)
            frame_count += 1

        cap.release()
        return video_name, frame_list
    
    os.makedirs(working_dir, exist_ok=True)
    
    # 입력 디렉토리에서 동영상 파일을 로드
    video_list = _load_video(input_dir)
    
    # 프레임 추출 및 저장
    frame_dict = {}
    for video_path in track(video_list, description="Extracting Frames"):
        video_name, frame_list = _extract_frames(video_path, working_dir)
        frame_dict[video_name] = frame_list
        
    return frame_dict

def compress_video(capture_dir: str, output_dir: str, angle_list: list):
    """
    캡처된 이미지들을 동영상으로 압축하는 함수
    """
    captured_list = [p for p in os.listdir(capture_dir) if p.endswith('.jpg')]
    # print(f"Captured images: {len(captured_list)} files found in {capture_dir}")
    for angle in track(angle_list, description="Compressing Videos"):
        v_angle, h_angle = angle
        image_list = [os.path.join(capture_dir, p) for p in captured_list if p.startswith(f'{v_angle:03d}_{h_angle:03d}')]
        image_list.sort()
        
        frame_list = [Image.open(image_path) for image_path in image_list]
        temp_path = os.path.join(output_dir, f"_{v_angle:03d}_{h_angle:03d}.mp4")
        output_path = os.path.join(output_dir, f"{v_angle:03d}_{h_angle:03d}.mp4")
        # smooth_output_path = os.path.join(output_dir, f"{v_angle:03d}_{h_angle:03d}_smooth.mp4")
        if frame_list:
            os.makedirs(output_dir, exist_ok=True)
            
            # 동영상 저장 설정
            width, height = frame_list[0].size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # 프레임 속도 설정
            
            video_writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            for frame in frame_list:
                frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            
        cmd = f"ffmpeg -y -i {temp_path} -c:v libx264 -pix_fmt yuv420p {output_path}"
        subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # cmd2 = f"ffmpeg -i {output_path} -vf 'minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1' -c:v libx265 -crf 20 {smooth_output_path}"
        # subprocess.run(shlex.split(cmd2), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_path)