import os

def check_smplx(smplx_dir: str, mean_parms: str):
    # SMPLX_NEUTRAL.npz가 있는지 확인
    smplx_fn = os.path.join(smplx_dir, 'smplx', 'SMPLX_NEUTRAL.npz')
    if not os.path.isfile(smplx_fn):
        print(f"{smplx_fn} not found")
        assert NotImplementedError
        
    # SMPL mean params가 있는지 확인
    if not os.path.isfile(mean_parms):
        print(f"{mean_parms} not found, downloading SMPL mean params")
        os.system(f"wget -O {mean_parms}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
        print('SMPL mean params have been succesfully downloaded')
    return smplx_fn