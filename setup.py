from setuptools import setup, find_packages

setup(
    name="cross-view-localization",
    version="0.1.0",
    description="Cross-view object localization from front view to satellite imagery",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "wandb>=0.13.0",
        "albumentations>=1.3.0",
        "pyyaml>=6.0",
        "opencv-python>=4.6.0",
    ],
    python_requires=">=3.8",
)
