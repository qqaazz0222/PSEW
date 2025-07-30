import os
import numpy as np

def print_npy_shapes(directory_path):
    """
    Loads all .npy files in the specified directory and prints their shapes.

    Args:
        directory_path (str): The path to the directory containing .npy files.
    """
    print(directory_path)
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    filelist = os.listdir(directory_path)
    filelist.sort()  # Sort the file list for consistent output

    for filename in filelist:
        if filename.endswith(".npy"):
            file_path = os.path.join(directory_path, filename)
            try:
                data = np.load(file_path, allow_pickle=True)
                print(f"File: {filename}, Shape: {data.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

if __name__ == '__main__':
    # Replace 'your_directory_path' with the actual path to your directory
    target_directory = 'your_directory_path'
    print_npy_shapes("data/working/demo_1/mesh")
    print_npy_shapes("data/working/demo_1/face")