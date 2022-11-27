import argparse
import json

import numpy as np

from pathlib import Path
from typing import Tuple, Dict, Any, List

from colmap import read_write_model


def read_colmap(str_path: str) -> Tuple[Dict[Any, Any], List[Any]]:
    object_intrinsics = read_write_model.read_cameras_binary(str_path + "/cameras.bin")[
        1
    ]

    object_cameras = {}

    for intImage, objectImage in enumerate(
        read_write_model.read_images_binary(str_path + "/images.bin").values()
    ):
        # https://github.com/colmap/colmap/blob/master/src/ui/model_viewer_widget.cc#L71

        dblFocal = None
        dblPrincipalX = None
        dblPrincipalY = None
        dblRadial = None

        if object_intrinsics.model == "SIMPLE_RADIAL":
            dblFocal = object_intrinsics.params[0]
            dblPrincipalX = object_intrinsics.params[1]
            dblPrincipalY = object_intrinsics.params[2]
            dblRadial = object_intrinsics.params[3]

        elif object_intrinsics.model == "PINHOLE":
            dblFocal = object_intrinsics.params[1]
            dblPrincipalX = object_intrinsics.params[2]
            dblPrincipalY = object_intrinsics.params[3]
            dblRadial = 0.0

        # end

        numpyIntrinsics = np.array(
            [
                [dblFocal, 0.0, dblPrincipalX],
                [0.0, dblFocal, dblPrincipalY],
                [0.0, 0.0, 1.0],
            ],
            np.float32,
        )
        numpyExtrinsics = np.zeros([3, 4], np.float32)
        numpyExtrinsics[0:3, 0:3] = read_write_model.qvec2rotmat(
            objectImage.qvec / (np.linalg.norm(objectImage.qvec) + 0.0000001)
        )
        numpyExtrinsics[0:3, 3] = objectImage.tvec

        object_cameras[objectImage.name] = {
            "intIdent": objectImage.id,
            "strImage": objectImage.name,
            "dblFocal": dblFocal,
            "dblPrincipalX": dblPrincipalX,
            "dblPrincipalY": dblPrincipalY,
            "dblRadial": dblRadial,
            "numpyIntrinsics": numpyIntrinsics,
            "numpyExtrinsics": numpyExtrinsics,
            "intPoints": [
                intPoint for intPoint in objectImage.point3D_ids if intPoint != -1
            ],
        }
    # end

    objectPoints = []

    for intPoint, objectPoint in enumerate(
        read_write_model.read_points3D_binary(str_path + "/points3D.bin").values()
    ):
        objectPoints.append(
            {
                "intIdent": objectPoint.id,
                "numpyLocation": objectPoint.xyz,
                "numpyColor": objectPoint.rgb[::-1],
            }
        )
    # end

    intPointindices = {}

    for intPoint, objectPoint in enumerate(objectPoints):
        intPointindices[objectPoint["intIdent"]] = intPoint
    # end

    for strCamera in object_cameras:
        object_cameras[strCamera]["intPoints"] = [
            intPointindices[intPoint]
            for intPoint in object_cameras[strCamera]["intPoints"]
        ]
    # end

    return object_cameras, objectPoints


def convert_colmap_to_json(colmap_dir: Path, params_dir: Path) -> None:
    """
    take the colmap format and convert it to the json format
    from 3d-atom.
    Writes the camera_params.json file at <params_dir>
    """
    json_fn = params_dir / "camera_params.json"

    colmap_path = colmap_dir / "sparse/0"
    obj_cameras, obj_points = read_colmap(str(colmap_path))

    # This numpy file is obtained from the LLFF's code
    poses_bounds = np.load(str(colmap_dir / "poses_bounds.npy"))
    all_bounds = poses_bounds[..., -2:]
    near, far = all_bounds[:, 0].min(), all_bounds[:, 1].max()
    near = 0.1 if near < 0.0 else near

    jout = {}
    for key in obj_cameras:
        cam = obj_cameras[key]

        extrin = cam["numpyExtrinsics"]
        jcam = {}
        jcam["extrinsic"] = {}
        jcam["extrinsic"]["rotation"] = extrin[:3, :3].tolist()
        jcam["extrinsic"]["translation"] = extrin[:3, 2:3].tolist()
        jcam["intrinsic"] = {}

        # assume principal pt is center
        jcam["intrinsic"]["width"] = int(cam["dblPrincipalX"] * 2)
        jcam["intrinsic"]["height"] = int(cam["dblPrincipalY"] * 2)
        jcam["intrinsic"]["focal"] = cam["dblFocal"]
        jcam["intrinsic"]["bounds"] = [near, far]
        jout[key] = jcam

    json.dump(jout, open(json_fn, "w"), sort_keys=True, indent=4)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Script for converting the poses data from COLMAP format into thre3d_atom's format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    # fmt: off
    parser.add_argument("-i", "--colmap_dir", action="store",
                        type=Path, required=True, help="path to the colmap's directory")
    parser.add_argument("-o", "--params_dir", action="store",
                        type=Path, required=True, help="path to the output directory")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    convert_colmap_to_json(args.colmap_dir, args.params_dir)


if __name__ == "__main__":
    main(parse_arguments())
