from setuptools import setup
from pathlib import Path

print(f"Current working directory: {Path().absolute()}")


setup(
    name="nerf",
    version="1.0",
    description="Project vanilla NeRF",
    install_requires=["thre3d_atom"],
    entry_points={
        "console_scripts": [
            f"nerf_train=train_nerf:main",
            f"nerf_demo_360=demo_360:main",
        ]
    },
)
