import os

def load_smplx_fn(smplx_dir: str):
    smplx_fn = os.path.join(smplx_dir, 'smplx', 'SMPLX_NEUTRAL.npz')
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
        print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
        print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
        print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
        print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
        assert NotImplementedError
    return smplx_fn

def load_smplx_mean(mean_parms: str):
    if not os.path.isfile(mean_parms):
        print('Start to download the SMPL mean params')
        os.system(f"wget -O {mean_parms}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
        print('SMPL mean params have been succesfully downloaded')
    return None