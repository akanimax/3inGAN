""" Script to run as a blender-background-script. Creates a 360 degree
rotating animation.
"""
# This script is run by Blender whose python environment has the following APIs
# So ignoring the inspection.
# noinspection PyUnresolvedReferences,PyPackageRequirements
import bpy

# noinspection PyUnresolvedReferences
import bmesh
import numpy as np
import os

# noinspection PyUnresolvedReferences,PyPackageRequirements
from mathutils import Matrix, noise, Vector


cam_collection = bpy.data.collections.new("Camera_collection")
bpy.context.scene.collection.children.link(cam_collection)
D = bpy.data

# Create a camera and Empty by whatever means that suits.
# Choose a radius to suit
# ===========================================================
# Tweakable parameters:
RADIUS = 15.0
camera_pitch = 60
num_cameras = 600  # 10 seconds video at 60fps
resolution = (1024, 1024)
scene_bounds = (5.0, 60.0)
focal_length = 25
sensor_size = 32
output_dir = "./rendered_images/demo_render"
hemisphere_location = Vector((0.0, 0.0, 5.0))
# ===========================================================

# code to make sure cycles engines uses GPU:
bpy.context.scene.render.engine = "BLENDER_EEVEE"

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True


# ----------------------------------------------------------------------------------
# helper functions needed for obtaining the demo camera path:
# ----------------------------------------------------------------------------------
def _translate_z(z):
    xlate = Matrix()
    xlate[2][3] = z
    return xlate


def _rotate_theta(theta):
    return Matrix(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def _rotate_phi(phi):
    return Matrix(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )


def pose_spherical(yaw, pitch, radius):
    c2w = _translate_z(radius)
    c2w = _rotate_phi(pitch / 180.0 * np.pi) @ c2w
    c2w = _rotate_theta(yaw / 180.0 * np.pi) @ c2w
    return c2w


# ----------------------------------------------------------------------------------


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

    # logic for adding cameras with the 360degree trajectory
    camera_yaw = camera_num * (360.0 / num_cameras)
    camera_pose = pose_spherical(camera_yaw, camera_pitch, RADIUS)

    cam_obj.matrix_world = camera_pose
    cam_collection.objects.link(cam_obj)


# render image for every camera:
for obj in bpy.context.scene.collection.children["Camera_collection"].objects:
    if obj.type == "CAMERA":
        bpy.context.scene.camera = obj
        bpy.context.scene.render.filepath = os.path.join(
            output_dir, "images", obj.name.split("_")[-1] + ".png"
        )
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
