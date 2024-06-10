import argparse
import math
import os
import random
import sys
import time
import shutil
import urllib.request
from typing import Tuple
from mathutils import Vector
import numpy as np
import bpy

def free_space_too_low(min_percent: int = 15):
    """
    This function is used to raise an exception
    if free space is less than N percent of total disk space.
    """
    total, used, free = shutil.disk_usage("c:/")
    min_bytes_needed = total * (min_percent / 100)
    if free < min_bytes_needed:
        raise Exception(
            f'Not enough free space: {(free / 1000 ** 3):.2f} gigabytes free, '
            f'at least {min_bytes_needed / (1000 ** 3):.2f} gigabytes needed'
        )


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--object_path", type=str, required=True, help="Path to the object file")
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDBLENDER_EEVEE"])
parser.add_argument("--num_images", type=int, default=32)
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--delete_processed", action="store_true", help="Delete processed glb file after rendering")
parser.add_argument("--focal_length", type=float, help="Set a constant focal length for the camera")
parser.add_argument("--random_focal_length", action="store_true", help="Use random focal length between 24mm and 200mm")
parser.add_argument("--focal_per_render", action="store_true", help="Randomize focal length per render if random focal length is enabled")
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

# Blender context and scene setup
context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

# Enable transparent film
scene.render.film_transparent = True

# Set anti-aliasing filter
scene.cycles.pixel_filter_type = 'BOX'
scene.cycles.filter_width = 1.0  # Set pixel filter width


# Cycles rendering settings
scene.cycles.device = "GPU"
scene.cycles.samples = 64 #512
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.005
scene.cycles.use_denoising = True

# Enable depth and normal passes
bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_normal = True

# Set compute device type to CUDA
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = True

# Function to setup compositor nodes
def setup_compositor_nodes(output_dir, object_uid, image_index):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create input render layers node
    rl_node = tree.nodes.new(type='CompositorNodeRLayers')
    
    # Create Map Value node for normalizing depth values
    map_node = tree.nodes.new(type='CompositorNodeMapValue')
    map_node.size = [0.1]  # Adjust this value to normalize the depth map
    map_node.use_min = True
    map_node.min = [0]
    map_node.use_max = True
    map_node.max = [1]

    # Create output file nodes for depth and normal passes
    depth_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output_node.label = 'Depth Output'
    depth_output_node.base_path = os.path.join(output_dir, object_uid, 'depth')
    depth_output_node.format.file_format = 'OPEN_EXR'
    depth_output_node.file_slots[0].path = f'depth_{image_index:03d}'

    normal_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    normal_output_node.label = 'Normal Output'
    normal_output_node.base_path = os.path.join(output_dir, object_uid, 'normal')
    normal_output_node.format.file_format = 'OPEN_EXR'
    normal_output_node.file_slots[0].path = f'normal_{image_index:03d}'

    # Create Premul (premultiplied alpha) node
    premul_node = tree.nodes.new(type='CompositorNodePremulKey')

    # Link render layers to output file nodes
    tree.links.new(rl_node.outputs['Depth'], depth_output_node.inputs[0])
    tree.links.new(rl_node.outputs['Normal'], normal_output_node.inputs[0])

    # Add alpha over node for handling transparency
    alpha_over_node = tree.nodes.new(type='CompositorNodeAlphaOver')
    alpha_over_node.inputs[1].default_value = (0, 0, 0, 0)  # Transparent background

    # Link render layers to premul node, then to composite output
    tree.links.new(rl_node.outputs['Image'], premul_node.inputs[0])
    tree.links.new(premul_node.outputs[0], alpha_over_node.inputs[2])
    tree.links.new(alpha_over_node.outputs[0], rl_node.outputs['Image'])
    
     # Link render layers to Map Value node
    tree.links.new(rl_node.outputs['Depth'], map_node.inputs[0])

    # Link Map Value node to depth output file node
    tree.links.new(map_node.outputs[0], depth_output_node.inputs[0])

    # Link render layers to normal output file node
    tree.links.new(rl_node.outputs['Normal'], normal_output_node.inputs[0])
    

# Function to rename files to the expected format
def rename_files(output_dir, object_uid, image_index):
    for pass_type in ['depth', 'normal']:
        folder_path = os.path.join(output_dir, object_uid, pass_type)
        old_file = os.path.join(folder_path, f"{pass_type}_{image_index:03d}0001.exr")
        new_file = os.path.join(folder_path, f"{image_index:03d}.exr")
        if os.path.exists(old_file):
            os.rename(old_file, new_file)
        else:
            print(f"Warning: {old_file} not found for renaming")

# Function to compose RT matrix
def compose_RT(R, T):
    return np.hstack((R, T.reshape(-1, 1)))

# Function to sample a point on a sphere
def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

# Function to sample spherical coordinates
def sample_spherical(radius_min=1.9, radius_max=2.6, maxz=2.60, minz=-1):
    while True:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        vec = np.array([x, y, z])
        radius = np.random.uniform(radius_min, radius_max)
        vec *= radius
        if minz < vec[2] < maxz:
            return vec

def set_camera_location(camera, option: str, focal_length: float, bounding_box: Tuple[Vector, Vector]):
    assert option in ['fixed', 'random', 'front']

    bbox_min, bbox_max = bounding_box
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min

    # Calculate the required distance to fit the object in the camera frame
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    aspect_ratio = sensor_width / sensor_height

    # Calculate the distance required to fit the object width and height within the camera frame
    half_diagonal_bbox = (bbox_size.x**2 + bbox_size.z**2)**0.5 / 2
    fov_diagonal = 2 * math.atan((sensor_width / (2 * focal_length)))
    
    # Calculate the distance to fit the bounding box within the frame considering aspect ratio
    distance = half_diagonal_bbox / math.tan(fov_diagonal / 2)

    if option == 'fixed':
        camera.location = Vector((bbox_center.x, bbox_center.y - distance, bbox_center.z))
    elif option == 'random':
        random_vec = sample_spherical(radius_min=distance * 0.8, radius_max=distance * 2, maxz=2.60, minz=-1)
        random_vec = Vector(random_vec)  # Convert numpy array to Vector
        camera.location = bbox_center + random_vec
    elif option == 'front':
        camera.location = Vector((bbox_center.x, bbox_center.y - distance, bbox_center.z))

    # Adjust orientation
    direction = bbox_center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Set the focal length
    camera.data.lens = focal_length

    return camera




# Function to add lighting
def add_lighting(option: str) -> None:
    assert option in ['fixed', 'random']

    # Delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()

    # Add a new light
    bpy.ops.object.light_add(type="AREA")
    light = bpy.data.lights["Area"]

    if option == 'fixed':
        light.energy = 30000
        bpy.data.objects["Area"].location[0] = 0
        bpy.data.objects["Area"].location[1] = 1
        bpy.data.objects["Area"].location[2] = 0.5

    elif option == 'random':
        light.energy = random.uniform(80000, 120000)
        bpy.data.objects["Area"].location[0] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[1] = random.uniform(-2., 2.)
        bpy.data.objects["Area"].location[2] = random.uniform(1.0, 3.0)

    # Set light scale
    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200

# Function to reset the scene
def reset_scene() -> None:
    """Resets the scene to a clean state."""
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

# Function to load the 3D object
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    print(f"Loading object: '{object_path}'")
    try:
        normalized_path = object_path.lower().strip()
        print(f"PATH: '{normalized_path}'")
        if normalized_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=normalized_path, merge_vertices=True)
        elif normalized_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=normalized_path)
        else:
            raise ValueError(f"Unsupported file type: '{object_path}'")
    except Exception as e:
        print(f"Error loading object '{object_path}': {e}")


# Function to compute scene bounding box
def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("No objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

# Function to get root objects in the scene
def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

# Function to get mesh objects in the scene
def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            yield obj

# Function to normalize the scene
def normalize_scene(box_scale: float):
    bbox_min, bbox_max = scene_bbox()
    scale = box_scale / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

# Function to setup the camera
def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 24
    cam.data.sensor_width = 32
    cam.data.sensor_height = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

# Function to save rendered images
def save_images(object_file: str) -> None:
    if free_space_too_low():
        raise Exception('Free space below threshold.')
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()

    try:
        load_object(object_file)
    except Exception as e:
        print(f"Error loading object {object_file}: {e}")
        return

    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene(box_scale=2)
    add_lighting(option='random')
    camera, cam_constraint = setup_camera()

    # Create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Prepare to save
    img_dir = os.path.join(args.output_dir, object_uid, 'rgba')
    pose_dir = os.path.join(args.output_dir, object_uid, 'pose')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    if args.random_focal_length and not args.focal_per_render:
        focal_length = random.uniform(24, 200)
    else:
        focal_length = args.focal_length or 24

    bbox = scene_bbox()

    for i in range(args.num_images):
        # Setup compositor nodes for each image
        setup_compositor_nodes(args.output_dir, object_uid, i)

        # Set the camera position
        camera_option = 'random' if i > 0 else 'front'

        if args.random_focal_length and args.focal_per_render:
            focal_length = random.uniform(24, 200)
        
        camera = set_camera_location(camera, option=camera_option, focal_length=focal_length, bounding_box=bbox)

        # Render and save the main image
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(img_dir, f"{i:03d}.png")
        bpy.ops.render.render(write_still=True)

        # Rename the depth and normal files
        rename_files(args.output_dir, object_uid, i)

        # Save camera RT matrix (C2W)
        location, rotation = camera.matrix_world.decompose()[0:2]
        RT = compose_RT(rotation.to_matrix(), np.array(location))
        RT_path = os.path.join(pose_dir, f"{i:03d}.npy")
        np.save(RT_path, RT)

    # Save the Blender file for inspection
    # blend_file_path = os.path.join(args.output_dir, f"{object_uid}.blend")
    # bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)

    # Save the camera intrinsics
    intrinsics = get_calibration_matrix_K_from_blender(camera.data, return_principles=True)
    intrinsics_file_path = os.path.join(args.output_dir, object_uid, 'intrinsics.npy')

    with open(intrinsics_file_path, 'wb') as f_intrinsics:
        np.save(f_intrinsics, intrinsics)

    # Check if the intrinsics file is saved and then delete the processed glb file if the flag is set
    if os.path.exists(intrinsics_file_path) and args.delete_processed and os.path.exists(object_file):
        os.remove(object_file)

# Function to download the object file
def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    local_path = os.path.abspath(local_path)
    return local_path

# Function to get camera calibration matrix
def get_calibration_matrix_K_from_blender(camera, return_principles=False):
    """
    Get the camera intrinsic matrix from Blender camera.
    Return also numpy array of principle parameters if specified.

    Intrinsic matrix K has the following structure in pixels:
        [fx  0 cx]
        [0  fy cy]
        [0   0  1]

    Specified principle parameters are:
        [fx, fy] - focal lengths in pixels
        [cx, cy] - optical centers in pixels
        [width, height] - image resolution in pixels

    """
    render = bpy.context.scene.render
    width = render.resolution_x * render.pixel_aspect_x
    height = render.resolution_y * render.pixel_aspect_y

    focal_length = camera.lens
    sensor_width = camera.sensor_width
    sensor_height = camera.sensor_height

    focal_length_x = width * (focal_length / sensor_width)
    focal_length_y = height * (focal_length / sensor_height)

    optical_center_x = width / 2
    optical_center_y = height / 2

    K = np.array([[focal_length_x, 0, optical_center_x],
                  [0, focal_length_y, optical_center_y],
                  [0, 0, 1]])

    if return_principles:
        return np.array([
            [focal_length_x, focal_length_y],
            [optical_center_x, optical_center_y],
            [width, height],
        ])
    else:
        return K

if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
