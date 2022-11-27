""" Script to run as a blender-background-process. Inserts cameras on upper hemisphere at random locations
and renders images from those locations. Also writes the camera_params.json file to interface with 3D-atom

This script is run by Blender whose python environment has the following APIs, so ignores certain inspections

Disclaimer: This is super hacky code, and I am not going to spend time refactoring it! It works :p!
"""

# noinspection PyUnresolvedReferences,PyPackageRequirements
import bpy

# noinspection PyUnresolvedReferences
import bmesh
import numpy as np
import os
import json

# noinspection PyUnresolvedReferences,PyPackageRequirements
from mathutils import Matrix, noise, Vector

# ----------------------------------------------------------------------------------------------------
# Create a camera and Empty by whatever means that suits.
# Choose a radius to suit
# ----------------------------------------------------------------------------------------------------
# Tweakable parameters:
RADIUS = 15.0
num_cameras = 120
resolution = (1024, 1024)
scene_bounds = (10.6698729811, 19.3301270189)
focal_length = 25
sensor_size = 32
output_dir = "./rendered_images/scene_render"
hemisphere_location = Vector((0.0, 0.0, 5.0))
# ----------------------------------------------------------------------------------------------------

# uncomment the following line for rendering scenes such as the forest-hyper-real
# use_engine = "BLENDER_EEVEE"
use_engine = "CYCLES"
bpy.context.scene.render.engine = use_engine

if use_engine == "CYCLES":
    # code to make sure cycles engines uses GPU:
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(
        "\nTotal detected devices:",
        len(bpy.context.preferences.addons["cycles"].preferences.devices),
    )

    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print("\nTurned on device:", d["name"], d["use"])


# some verbose stuff:
cam_collection = bpy.data.collections.new("Camera_collection")
bpy.context.scene.collection.children.link(cam_collection)
D = bpy.data

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Translate matrix along +Z
xlate = Matrix()
xlate[2][3] = RADIUS

if not os.path.exists(output_dir):
    print(f"Creating output_directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# add the required number of cameras to the scene
for camera_num in range(num_cameras):
    cam_name = f"Kaimera_{camera_num}"

    # create the camera
    cam = bpy.data.cameras.new(cam_name)
    cam.lens = focal_length
    cam.sensor_width = sensor_size
    cam.sensor_height = sensor_size

    # create the camera object
    cam_obj = bpy.data.objects.new(cam_name, cam)

    v0 = RADIUS * noise.random_unit_vector() + hemisphere_location
    # resample the point back to positive hemisphere
    v0[-1:] = np.abs(v0[-1:])

    # Fetch tracking quaternion to align camera to orientation
    quat0 = v0.to_track_quat("Z", "Y")
    # Translate camera length of radius, orient camera with tracking quad
    cam_obj.matrix_world = quat0.to_matrix().to_4x4() @ xlate
    cam_collection.objects.link(cam_obj)

# write the camera_params.json file:
camera_params = {}
for obj in bpy.context.scene.collection.children["Camera_collection"].objects:
    if obj.type == "CAMERA":
        camera_params[str(obj.name) + ".png"] = {
            "extrinsic": {
                "rotation": np.array(obj.matrix_world)[:3, :3].astype(np.str).tolist(),
                "translation": np.array(obj.matrix_world)[:3, 3:]
                .astype(np.str)
                .tolist(),
            },
            "intrinsic": {
                "height": str(resolution[0]),
                "width": str(resolution[1]),
                "focal": str((max(resolution) * focal_length) / sensor_size),
                "bounds": [str(scene_bounds[0]), str(scene_bounds[1])],
            },
        }

# save the camera_params to a json object:
with open(os.path.join(output_dir, "camera_params.json"), "w") as dumper:
    json.dump(camera_params, dumper, indent=4)

# render image for every camera:
for obj in bpy.context.scene.collection.children["Camera_collection"].objects:
    if obj.type == "CAMERA":
        bpy.context.scene.camera = obj
        bpy.context.scene.render.filepath = os.path.join(output_dir, "images", obj.name)
        # noinspection DuplicatedCode
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.cycles.samples = 2048  # to reduce noise
        bpy.context.scene.render.dither_intensity = 0
        bpy.context.scene.render.image_settings.compression = 0
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGB"
        bpy.context.scene.render.image_settings.compression = 0
        print(f"rendering scene here now: {obj.name}")
        bpy.ops.render.render(write_still=True)


# this ensures that the added cameras are not saved in the scene
exit(0)
