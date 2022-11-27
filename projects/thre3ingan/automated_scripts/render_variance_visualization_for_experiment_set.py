""" Hilarious script written for automating render visualization for one set of experiments"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hilarious script written for automating the render visualization for one set of experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "--experiments_base_path",
                        action="store", type=Path, required=True,
                        help="path to the base-directory of experiment set")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def main() -> None:
    args = parse_arguments(sys.argv[1:])

    main_experiment_path = args.experiments_base_path
    all_experiments = [
        possible_exp_path
        for possible_exp_path in main_experiment_path.iterdir()
        if possible_exp_path.is_dir()
    ]

    # run the loop over the experiments and render the variance visualization for them ...
    for experiment in all_experiments:
        saved_model_path = experiment / "saved_models" / "model_stage_7_iter_2000.pth"
        render_vis_path = experiment / "variance_render"

        # obtain the file path for the variance render script:
        from projects.thre3ingan import (
            demo_thre3d_singan_with_direct_supervision_fixed_cam_variance,
        )

        render_vis_path.mkdir(exist_ok=True)
        script_to_run = Path(
            demo_thre3d_singan_with_direct_supervision_fixed_cam_variance.__file__
        ).absolute()

        log.info(f"Rendering variance visualization for experiment: {experiment}")
        subprocess.run(
            [
                "python",
                str(script_to_run),
                "-i",
                str(saved_model_path),
                "-o",
                str(render_vis_path),
            ],
            cwd=str(experiment),
        )

        log.info(f"encoding mp4 video from the frames at: {render_vis_path}")
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                "%d.png",
                "-c:v",
                "libx264",
                "-crf",
                "2",
                "-profile:v",
                "main",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "faststart",
                f"{experiment.name}_variance_video.mp4",
            ],
            cwd=str(render_vis_path),
        )

        log.info(
            f"converting the mp4 video into gif for more options: {render_vis_path}"
        )
        subprocess.run(
            [
                "convert",
                f"{experiment.name}_variance_video.mp4",
                f"{experiment.name}_variance_video.gif",
            ],
            cwd=str(render_vis_path),
        )


if __name__ == "__main__":
    main()
