from distutils.core import setup

setup(
    name="video-segmentation",
    packages=["dataprocessing", "evaluation", "models", "training", "utils"],
    install_requires=[
        "wheel",
        "pandas>=1.2",
        "numpy>=1.20",
        "tqdm",
        "torch>=1.7",
        "dill>=0.3",
        "matplotlib>=3.3",
        "opencv-python>=4.5",
        "pycocotools>=2",
        "multiprocess>=0.70"
    ]
)

