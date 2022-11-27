""" Hilarious script written for automating score calculation (quality diversity metrics)
 for one set of experiments"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hilarious script written for automating the score computation "
        "(quality-diversity metrics) "
        "for one set of experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "--experiments_base_path",
                        action="store", type=Path, required=True,
                        help="path to the base-directory of experiment set")
    parser.add_argument("-i_g", "-m_g", "--gt_models_path",
                        action="store", type=Path, required=True,
                        help="path to the folder containing all "
                             "ground truth ReluField models. Note that the experiment folder names "
                             "and the model-files' names should match")
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

    # the path to the folder containing all the ground truth Relu-Field models
    gt_models_path = args.gt_models_path

    # run the loop over the experiments and render the variance visualization for them ...
    for experiment in all_experiments:
        # obtain the file paths for the metric_sample generation and
        # metric (scores) computation scripts:
        from projects.thre3ingan import (
            render_real_and_fake_view_synchronized_samples,
            compute_quality_and_diversity_metrics,
        )

        # setup paths for the subprocess to run
        saved_model_path = experiment / "saved_models" / "model_stage_7_iter_2000.pth"
        metric_samples_path = experiment / "metric_samples"
        gt_model = gt_models_path / f"{experiment.name}.pth"
        if not gt_model.is_file():
            raise ValueError(
                f"Pch pch pch ... The gt model for experiment ({experiment}) not found :("
            )

        # First render the samples needed for computing the metric:
        script_to_run = Path(
            render_real_and_fake_view_synchronized_samples.__file__
        ).absolute()
        log.info(f"Rendering metric-samples for experiment: {experiment}")
        subprocess.run(
            [
                "python",
                str(script_to_run),
                "-i",
                str(saved_model_path),
                "-i_g",
                str(gt_model),
                "-o",
                str(metric_samples_path),
            ],
            cwd=str(experiment),
        )

        # Once the samples are rendered for the metric computation, then compute the metric:
        script_to_run = Path(compute_quality_and_diversity_metrics.__file__).absolute()
        metric_log_file = experiment / "quality_diversity_scores.txt"
        with open(str(metric_log_file), "w") as log_out:
            log.info(
                f"Computing the quality-diversity metrics for experiment: {experiment}"
            )
            subprocess.run(
                [
                    "python",
                    str(script_to_run),
                    "--metric_samples_path",
                    str(metric_samples_path),
                ],
                cwd=str(experiment),
                # note that all the logs we are writing go to stderr not stdout :D
                stderr=log_out,
            )
        log.info(
            f"Metrics for experiment ({experiment.name}) are computed ... "
            f"Please check scores in: {metric_log_file}"
        )


if __name__ == "__main__":
    main()
