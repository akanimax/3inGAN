import subprocess
import sys

from pathlib import Path
from setuptools import setup

with open("README.md", "r") as file_:
    project_description = file_.read()

with open("requirements.txt", "r") as file_:
    project_requirements = file_.read().split("\n")

setup(
    name="thre3d_atom",
    version="0.1",
    description="Library for smart (deep-learning based) 3D algorithms",
    license="MIT",
    long_description=project_description,
    author="akanimax (Animesh Karnewar)",
    author_email="animeshsk3@gmail.com",
    url="https://github.com/akanimax/3d-atom",
    packages=["thre3d_atom"],
    install_requires=project_requirements,
)

# bootstrap all the projects
project_dir = Path("projects/")
for project in project_dir.iterdir():
    if project.is_dir():
        if (project / "setup.py").is_file():
            print(f"\nSetting up project {project.name}")
            subprocess.run(["python", str("setup.py"), sys.argv[-1]], cwd=str(project))
