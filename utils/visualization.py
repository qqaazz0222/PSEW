import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
import subprocess, shlex    
import shutil

def visualize_loc(frame_list: list, parameter_list: list, img_size: tuple):
    vis_loc_dir = '_vis_loc'
    os.makedirs(vis_loc_dir, exist_ok=True)
    
    task_list = zip(frame_list, parameter_list)

    for i, task in enumerate(track(task_list, description="Visualizing Locations")):
        frame_path, parameters = task
        # print(i, parameters)
        
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]
        target_width = img_size[0]
        aspect_ratio = w / h
        target_height = int(target_width / aspect_ratio)
        frame = cv2.resize(frame, (target_width, target_height))
        # Calculate center positions
        img_center_x, img_center_y = img_size[0] // 2, img_size[1] // 2
        frame_center_x, frame_center_y = target_width // 2, target_height // 2

        # Calculate top-left position to center the frame
        start_x = img_center_x - frame_center_x
        start_y = img_center_y - frame_center_y

        # Ensure the frame fits within the image bounds
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(img_size[0], start_x + target_width)
        end_y = min(img_size[1], start_y + target_height)

        # Overlay the frame onto the center of the image
        img[start_y:end_y, start_x:end_x] = frame[:end_y-start_y, :end_x-start_x]

        for loc in parameters['loc'].tolist():
            cv2.circle(img, (int(loc[0]), int(loc[1])), 5, (0, 255, 0), -1)

        vis_path = os.path.join(vis_loc_dir, f"{i:08d}.png")
        cv2.imwrite(vis_path, img)
        
    # Create video from images using ffmpeg
    ffmpeg_cmd = f'ffmpeg -y -framerate 30 -i {vis_loc_dir}/%08d.png -c:v libx264 -pix_fmt yuv420p vis_loc.mp4'
    subprocess.run(shlex.split(ffmpeg_cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists('vis_loc.mp4'):
        print(f"Video saved to: vis_loc.mp4")
        shutil.rmtree(vis_loc_dir, ignore_errors=True)
    else:
        print("Failed to create video. Please check the images in the directory.")
    

def visualize_transl(human_list: list):
    vis_transl_dir = '_vis_transl'
    os.makedirs(vis_transl_dir, exist_ok=True)

    for i, humans in enumerate(track(human_list, description="Visualizing Translations")):
        transl = [h['transl'].cpu().numpy() for h in humans]
        transl_pelvis = [h['transl_pelvis'].cpu().numpy() for h in humans]
        
        # Set labels and title
        # Create 2x2 subplot layout for different views
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        
        views = [
            {'title': 'Default View', 'elev': 20, 'azim': 225},
            {'title': 'Front View', 'elev': 0, 'azim': 180},
            {'title': 'Side View', 'elev': 0, 'azim': 270},
            {'title': 'Top View', 'elev': 90, 'azim': 180}
        ]
        
        for idx, (ax, view) in enumerate(zip(axes, views)):
            # Plot transl in red for each human
            for j, t in enumerate(transl):
                if len(t) >= 3:
                    ax.scatter(t[0], t[2], t[1], c='red', s=24, label='head' if j == 0 else "")
            
            # Plot transl_pelvis in blue for each human
            for j, tp in enumerate(transl_pelvis):
                tp = tp[0]
                if len(tp) >= 3:
                    ax.scatter(tp[0], tp[2], tp[1], c='blue', s=24, label='pelvis' if j == 0 else "")
            
            # Set view properties
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_xlim([-5, 5])
            ax.set_ylim([0, 10])
            ax.set_zlim([5, -5])
            ax.set_title(f'{view["title"]} - Frame {i}')
            ax.view_init(elev=view['elev'], azim=view['azim'])
            if idx == 0:  # Only show legend on first subplot
                ax.legend()
        
        plt.tight_layout()
        
        # Save the plot
        vis_path = os.path.join(vis_transl_dir, f"{i:08d}.png")
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create video from images using ffmpeg
    ffmpeg_cmd = f"ffmpeg -y -framerate 30 -i {vis_transl_dir}/%08d.png -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p vis_transl.mp4"
    subprocess.run(shlex.split(ffmpeg_cmd), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists('vis_transl.mp4'):
        print(f"Video saved to: vis_loc.mp4")
        shutil.rmtree(vis_transl_dir, ignore_errors=True)
    else:
        print("Failed to create video. Please check the images in the directory.")
        
