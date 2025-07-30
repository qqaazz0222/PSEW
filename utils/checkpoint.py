import os
import pickle

def load_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> dict:
    flag = False
    data = None
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")
    if not os.path.exists(checkpoint_path):
        return flag, data
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded from {checkpoint_path}")
    flag = True
    return flag, data

def save_checkpoint(checkpoint_dir: str, checkpoint_name: str, data: dict) -> None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved at {checkpoint_path}")
    