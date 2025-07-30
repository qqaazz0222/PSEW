# PSEW(Pose Scene EveryWhere)

단안 카메라 영상을 활용한 행동 인식 학습용 다중 시점 데이터 생성 프레임워크

A Framework for Generating Multi-View Data for Action Recognition Training from Monocular Videos

### Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. 저장소 복제

```bash
git clone https://github.com/qqaazz0222/PSEW
cd PSEW
```

**Step 2**. Conda 가상환경 생성 및 활성화

```bash
conda env create -f env.yaml
conda activate psew
```

**Step 3**. 모델 가중치 다운로드

아래 링크에서 다운로드 받은 가중치 파일을 `root/model`디렉토리에 위치시킵니다.

-   **객체 검출 모델**: YOLO11 ( Download Link: [yolo11x.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt) )
-   **세그멘테이션 모델**: SAM 2 ( Download Link: [sam2.1_b.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt) )
