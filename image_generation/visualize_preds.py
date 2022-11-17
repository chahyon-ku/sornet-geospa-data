import argparse
import datetime
import json
import os
import random
import h5py
import numpy
from scipy.spatial.transform.rotation import Rotation
import utils
import matplotlib.pyplot as plt
from PIL import Image
import cv2

try:
    import bpy, bpy_extras
    from bpy.types import Object as BpyObj
    from mathutils import Vector
except ImportError as e:
    print(e)
    print("\nERROR")
    print("Could not import the blender modules. Make sure blender was " + \
          "installed using the instructions from the readme.")


def parse_args():
    parser = argparse.ArgumentParser()
    # Input options
    parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                        help="Base blender file on which all scenes are based; includes " +
                             "ground plane, lights, and camera.")
    parser.add_argument('--properties_json', default='data/properties/mug_properties.json',
                        help="JSON file defining objects, materials, sizes, and colors. " +
                             "The \"colors\" field maps from CLEVR color names to RGB values; " +
                             "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                             "rescale object models; the \"materials\" and \"shapes\" fields map " +
                             "from CLEVR material and shape names to .blend files in the " +
                             "--object_material_dir and --shape_dir directories respectively.")
    parser.add_argument('--shape_dir', default='data/shapes',
                        help="Directory where .blend files for object models are stored")
    parser.add_argument('--material_dir', default='data/materials',
                        help="Directory where .blend files for materials are stored")
    parser.add_argument('--shape_color_combos_json', default=None,
                        help="Optional path to a JSON file mapping shape names to a list of " +
                             "allowed color names for that shape. This allows rendering images " +
                             "for CLEVR-CoGenT.")

    # Settings for objects
    parser.add_argument('--min_objects', default=1, type=int,
                        help="The minimum number of objects to place in each scene")
    parser.add_argument('--max_objects', default=1, type=int,
                        help="The maximum number of objects to place in each scene")
    parser.add_argument('--min_dist', default=1.1, type=float,
                        help="The minimum allowed distance between object centers")
    parser.add_argument('--margin', default=0.01, type=float,
                        help="Along all cardinal directions (left, right, front, back), all " +
                             "objects will be at least this distance apart. This makes resolving " +
                             "spatial relationships slightly less ambiguous.")
    parser.add_argument('--min_pixels_per_object', default=200, type=int,
                        help="All objects will have at least this many visible pixels in the " +
                             "final rendered images; this ensures that no objects are fully " +
                             "occluded by other objects.")
    parser.add_argument('--max_retries', default=50, type=int,
                        help="The number of times to try placing an object before giving up and " +
                             "re-placing all objects in the scene.")

    # Output settings
    parser.add_argument('--start_idx', default=0, type=int,
                        help="The index at which to start for numbering rendered images. Setting " +
                             "this to non-zero values allows you to distribute rendering across " +
                             "multiple machines and recombine the results later.")
    parser.add_argument('--num_images', default=10, type=int,
                        help="The number of images to render")
    parser.add_argument('--filename_prefix', default='CLEVR',
                        help="This prefix will be prepended to the rendered images and JSON scenes")
    parser.add_argument('--split', default='new',
                        help="Name of the split for which we are rendering. This will be added to " +
                             "the names of rendered images, and will also be stored in the JSON " +
                             "scene structure for each image.")
    parser.add_argument('--version', default='1.0',
                        help="String to store in the \"version\" field of the generated JSON file")
    parser.add_argument('--license',
                        default="Creative Commons Attribution (CC-BY 4.0)",
                        help="String to store in the \"license\" field of the generated JSON file")
    parser.add_argument('--date', default=datetime.datetime.today().strftime("%m/%d/%Y"),
                        help="String to store in the \"date\" field of the generated JSON file; " +
                             "defaults to today's date")

    # Rendering options
    parser.add_argument('--use_gpu', default=1, type=int,
                        help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                             "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                             "to work.")
    parser.add_argument('--width', default=480, type=int,
                        help="The width (in pixels) for the rendered images")
    parser.add_argument('--height', default=320, type=int,
                        help="The height (in pixels) for the rendered images")
    parser.add_argument('--key_light_jitter', default=1.0, type=float,
                        help="The magnitude of random jitter to add to the key light position.")
    parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                        help="The magnitude of random jitter to add to the fill light position.")
    parser.add_argument('--back_light_jitter', default=1.0, type=float,
                        help="The magnitude of random jitter to add to the back light position.")
    parser.add_argument('--camera_jitter', default=0.5, type=float,
                        help="The magnitude of random jitter to add to the camera position")
    parser.add_argument('--render_num_samples', default=512, type=int,
                        help="The number of samples to use when rendering. Larger values will " +
                             "result in nicer images but will cause rendering to take longer.")
    parser.add_argument('--render_min_bounces', default=8, type=int,
                        help="The minimum number of bounces to use for rendering.")
    parser.add_argument('--render_max_bounces', default=8, type=int,
                        help="The maximum number of bounces to use for rendering.")
    parser.add_argument('--render_tile_size', default=256, type=int,
                        help="The tile size to use for rendering. This should not affect the " +
                             "quality of the rendered image but may affect the speed; CPU-based " +
                             "rendering may achieve better performance using smaller tile sizes " +
                             "while larger tile sizes may be optimal for GPU-based rendering.")
    parser.add_argument('--side_camera', default=0, type=int,
                        help="The number of side cameras used.")
    parser.add_argument('--top_camera', action="store_true",
                        help="Use top camera or not, default no.",
                        default=0)
    parser.add_argument('--skip_ori_camera', action="store_true",
                        help="Use original camera or not, default yes.")
    parser.add_argument('--store_depth', action="store_true",
                        help="Redirect the depth info into the alpha channel of the output " +
                             "PNG file for later processing. This will make the images " +
                             "appear somewhat transparent to standard image viewers.")

    parser.add_argument('--h5_path', default='../output/mug_rotonly_valid.h5')
    parser.add_argument('--csv_path', default='../../cap-the-bottle/predictions/mug_rotonly.csv')
    parser.add_argument('--temp_dir', default='temp/')
    parser.add_argument('--out_dir', default='viz/mug_rotonly/')

    return parser.parse_args()


def init_scene_directions(scene_struct):
    '''
    Calculates vectors for each of the cardinal directions (behind, front, left,
    right, above) and appends them to the scene struct. Must be called after
    the camera jitter is finalized (after jitter_cam_lights())
    Args:
        scene_struct: the dictionary to add the cardinal direction vectors
    '''
    # Put a plane on the ground so we can compute cardinal directions
    if bpy.app.version < (2, 80, 0):
        bpy.ops.mesh.primitive_plane_add(radius=5)
    else:
        bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object
    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    if bpy.app.version < (2, 80, 0):
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    else:
        cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)


def add_object(self, object, args):
    '''
    Add random objects to the current blender scene
    Args:
        scene_struct - the dictionary for the scene which gets saved into json
        num_objects - the number of objects to put in the scene, a random value
                    between min_objects and max_objects
        args - the parsed command line arguments
        camera - the Blender camera objects
        output_index - the index of the scene being rendered
    '''
    x = object['3d_coords'][0]
    y = object['3d_coords'][1]
    z = object['3d_coords'][2]
    count = 0
    shape_name = object['shape']
    for obj in bpy.data.objects:
        if obj.name.startswith(shape_name):
            count += 1

    filename = os.path.join(args.shape_dir, '%s.blend' % shape_name, 'Object', shape_name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (shape_name, count)
    bpy.data.objects[shape_name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[0] = object['rotation'][0] * numpy.pi / 180
    bpy.context.object.rotation_euler[1] = object['rotation'][1] * numpy.pi / 180
    bpy.context.object.rotation_euler[2] = object['rotation'][2] * numpy.pi / 180
    bpy.ops.transform.resize(value=object['size_val'])
    bpy.context.object.location = x, y, z
    bpy.context.view_layer.update()  # Update object for new location and rotation

    colors = {
        "gray": [87, 87, 87, 255],
        "red": [173, 35, 35, 255],
        "blue": [42, 75, 215, 255],
        "green": [29, 105, 20, 255],
        "brown": [129, 74, 25, 255],
        "purple": [129, 38, 192, 255],
        "cyan": [41, 208, 208, 255],
        "yellow": [255, 238, 51, 255]
    }
    colors = {k: [vi / 255.0 for vi in v] for k, v in colors.items()}
    materials = {
        "rubber": "Rubber",
        "metal": "MyMetal"
    }

    utils.add_material(bpy.context.active_object, materials[object['material']], Color=colors[object['color']])


def render_image(args):
    ''' Renders the current scene to PNG file(s) based on the camera
    arguments, rendering a different PNG image for each camera view.
    '''
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.use_zbuffer = True
    bpy.data.cameras['Camera'].lens = 58.5
    bpy.data.cameras['Camera'].lens_unit = "FOV"
    counter = 0
    if args.store_depth:
        utils.redirect_depth_to_alpha()
    # cur = bpy.context.scene.render.filepath
    # cur = cur[:-4] + "camera"
    for ob in bpy.context.scene.objects:
        if ob.type == "CAMERA":
            if ob.name == "Camera":
                if args.skip_ori_camera:
                    continue
                # bpy.context.scene.render.filepath = cur + "_ori"
            elif ob.name == "Camera.005":
                if not args.top_camera:
                    continue
                # bpy.context.scene.render.filepath = cur + "_top"
            else:
                counter += 1
                if counter > args.side_camera:
                    continue
                # bpy.context.scene.render.filepath = cur + str(counter)
            bpy.context.scene.camera = ob
            while True:
                try:
                    bpy.ops.render.render(write_still=True, use_viewport=True)
                    break
                except Exception as e:
                    print(e)


def render_scene(image_path, objects, args):
    # Initialize the settings for scene rendering
    utils.init_scene_settings(args.base_scene_blendfile,
                              args.material_dir,
                              args.width,
                              args.height,
                              args.render_tile_size,
                              args.render_num_samples,
                              args.render_min_bounces,
                              args.render_max_bounces,
                              args.use_gpu)

    bpy.context.scene.render.filepath = image_path
    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'image_filename': os.path.basename(image_path),
        'objects': [],
        'directions': {},
    }
    init_scene_directions(scene_struct)

    for object in objects:
        add_object(scene_struct, object, args)

    render_image(args)


if __name__ == '__main__':
    args = parse_args()

    h5f = h5py.File(args.h5_path, 'r')
    csv = numpy.loadtxt(args.csv_path)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    for scene_i, scene_key in enumerate(h5f):
        objects = json.loads(h5f[scene_key]['objects'][()])
        render_scene(os.path.join(os.path.abspath(args.temp_dir), scene_key + '_gt'), objects, args)

        pose_pred = csv[scene_i]
        rotation_pred = Rotation.from_quat(pose_pred[3:])
        euler_pred = rotation_pred.as_euler('xyz', degrees=True)
        objects[0]['3d_coords'][0] = pose_pred[0]
        objects[0]['3d_coords'][1] = pose_pred[1]
        objects[0]['3d_coords'][2] = pose_pred[2]
        objects[0]['rotation'][0] = euler_pred[0]
        objects[0]['rotation'][1] = euler_pred[1]
        objects[0]['rotation'][2] = euler_pred[2]

        render_scene(os.path.join(os.path.abspath(args.temp_dir), scene_key + '_pred'), objects, args)

        img_gt = numpy.array(Image.open(os.path.join(os.path.abspath(args.temp_dir), scene_key + '_gt.png')))
        img_pred = numpy.array(Image.open(os.path.join(os.path.abspath(args.temp_dir), scene_key + '_pred.png')))
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_gt)
        axes[0].axis('off')
        axes[0].set_title('Ground Truth')
        axes[1].imshow(img_pred)
        axes[1].axis('off')
        axes[1].set_title('Predicted Pose')
        plt.savefig(os.path.join(os.path.abspath(args.out_dir), scene_key + '.png'), bbox_inches='tight')
        plt.close()

        if scene_i == 99:
            break