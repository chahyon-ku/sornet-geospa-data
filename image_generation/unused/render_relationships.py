'''
Renders the allowed relationships between every pair of objects, using the same
logic for placement as in render_images.py 

This file is useful for visualizing the different ways objects can be placed into 
scenes, and for debugging the correctness of the relationship matrices.
'''

from __future__ import print_function
import math
import sys
import random
import argparse
import json
import os
import tempfile
from datetime import datetime as dt
from collections import Counter
import numpy
import io
import utils

try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError as e:
    print(e)
    print("\nERROR")
    print("Could not import the blender modules. Make sure blender was " + \
          "installed using the instructions from the readme.")

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
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


# Output settings
parser.add_argument('--filename_prefix', default='',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/relationships/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
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
parser.add_argument('--camera_jitter', default=0.0, type=float,
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
parser.add_argument('--cam_views', default=1, type=int,
                    help="The number of different views to render for each shape.")


def main(args):
    args.output_image_dir = os.path.abspath(args.output_image_dir)
    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    render_rels(args)

def render_rels(args):
    utils.init_scene_settings(args.base_scene_blendfile, 
                                args.material_dir, 
                                args.width, 
                                args.height, 
                                args.render_tile_size, 
                                args.render_num_samples, 
                                args.render_min_bounces, 
                                args.render_max_bounces, 
                                args.use_gpu)
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba

    stdout = io.StringIO()

    # Load the support and contain matrixes from file
    con_rel_mat = numpy.loadtxt("data/rels_contain.dat", dtype=int)
    sup_rel_mat = numpy.loadtxt("data/rels_support.dat", dtype=int)
    object_placer = utils.ObjectPlacer(args.shape_dir)
    # Define the properties for the parent and child
    size, sizeval = list(properties['sizes'].items())[0]  # there's only one size
    material, materialval = list(properties['materials'].items())[0]  # use rubber material
    parent_colorval = color_name_to_rgba['red']  # red
    child_colorval = color_name_to_rgba['blue']  # blue
    for parent_shape, parent_shape_num in (properties['shapes'].items()):
        parent_dir = os.path.join(args.output_image_dir, parent_shape)
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
        fit_dir = os.path.join(parent_dir, 'fit')
        con_dir = os.path.join(parent_dir, 'contain')
        sup_dir = os.path.join(parent_dir, 'support')
        # Add the parent object to the scene
        utils.new_add_object(args.shape_dir,
                         parent_shape, sizeval, loc=(0, 0), theta=0)
        parent_obj = bpy.context.object
        utils.add_material(parent_obj, materialval, Color=parent_colorval)
        # Parent dimension info
        loc = parent_obj.location
        px = parent_obj.dimensions[0]
        py = parent_obj.dimensions[1]
        pz = parent_obj.dimensions[2]
        # Try child objects
        for child_shape, child_shape_num in properties['shapes'].items():
            sup_bool = sup_rel_mat[parent_shape_num][child_shape_num]
            con_bool = con_rel_mat[parent_shape_num][child_shape_num]
            file_name = child_shape
            fix = False
            theta = 0
            # Support
            if sup_bool and False:  # NOTE: right now only render contain relations
                save_path = os.path.join(sup_dir, file_name)
                if os.path.isfile(save_path + '.png'):
                    # print('already exists: ' + save_path)
                    pass
                else:
                    z = loc[2] + pz / 2 + 0.01
                    on_top = False
                    while not on_top:
                        dx = random.uniform(-px / 2, px / 2)
                        dy = random.uniform(-py / 2, py / 2)
                        x = loc[0] + dx
                        y = loc[1] + dy
                        on_top = above_sup((x, y, z), parent_obj)
                    utils.new_add_object(args.shape_dir, child_shape, sizeval, loc=(
                        x, y), theta=theta, zvalue=z)
                    child_obj = bpy.context.object
                    utils.add_material(child_obj, materialval,
                                    Color=child_colorval)
                    if not os.path.isdir(sup_dir):
                        print('made: ' + sup_dir)
                        os.makedirs(sup_dir)
                    render_to_file(save_path)
                    utils.delete_object(child_obj)
            # Contain
            if con_bool:
                fix = con_rel_mat[parent_shape_num][child_shape_num] == 2
                # Choose save file path
                if fix:
                    save_path = os.path.join(fit_dir, file_name)
                    if not os.path.isdir(fit_dir):
                        os.makedirs(fit_dir)
                else:
                    save_path = os.path.join(con_dir, file_name)
                    if not os.path.isdir(con_dir):
                        os.makedirs(con_dir)
                # Place object
                if os.path.isfile(save_path + '.png'):
                    # If file already exists then skip
                    pass
                else:
                    # Always place fixed in center
                    child_obj = object_placer.place_in_container(child_shape, sizeval, parent_obj, True)
                    # Check for collision between objects
                    if not fix and not utils.check_collision(child_obj, parent_obj):
                        # There will sometimes be collisions when its a fit relationship, 
                        # that's ok
                        print(f'ERROR failed overlap check: {parent_shape} <- {child_shape}')
                        output_blendfile = save_path + '.blend'
                        print(f'Saving blendfile to {output_blendfile}')
                        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
                    utils.add_material(bpy.context.active_object, materialval,
                                        Color=child_colorval)
                    render_to_file(save_path)
                    utils.delete_object(bpy.context.active_object)
        # Delete the parent object
        utils.delete_object(parent_obj)


def render_to_file(path: str):
    '''
    Renders the current blender scene to a file at the given path
    '''

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.data.cameras['Camera'].lens = 58.5
    bpy.data.cameras['Camera'].lens_unit = "FOV"

    counter = 0
    bpy.context.scene.render.filepath = path
    cur = bpy.context.scene.render.filepath
    for ob in bpy.context.scene.objects:
        if ob.type == "CAMERA":
            if ob.name == "Camera":
                # Skip original CLEVR camera
                continue
            elif ob.name == "Camera.005":
                # Skip top camera
                # continue
                bpy.context.scene.render.filepath = cur + "_top"
            else:
                counter += 1
                if counter > args.cam_views:
                    # Skip when created enough views
                    continue
                if args.cam_views > 1:
                    bpy.context.scene.render.filepath = cur + str(counter)
                else:
                    bpy.context.scene.render.filepath = cur
            bpy.context.scene.camera = ob
            while True:
                try:
                    # print("rendering object " + bpy.context.scene.render.filepath)
                    bpy.ops.render.render(write_still=True, use_viewport=True)
                    break
                except Exception as e:
                    print("RENDER ERROR: " + e)


if __name__ == '__main__': 
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        # Run normally
        args = parser.parse_args()
        main(args)
