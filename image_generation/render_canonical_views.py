# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

INSIDE_BLENDER = True
import bpy
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_canonical_view.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

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

# Setting for objects
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

# Output settings
parser.add_argument('--to_h5', default=None,
                    help="Save to h5 file rather than seperate PNGs, default no. " +
                         "The file name of the saved h5 file." )
parser.add_argument('--start_idx', default=0, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=10, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_prefix', default='',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/mug_views/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--license',
                    default="Creative Commons Attribution (CC-BY 4.0)",
                    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file; " +
                         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=1, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
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
    num_digits = 6
    prefix = '%s' % args.filename_prefix
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    img_template = os.path.join(os.path.abspath(args.output_image_dir), img_template)

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)

    #for i in range(args.num_images):
    img_path = ''#img_template % (i + args.start_idx)
    render_views(args,
                 output_index=0,
                 output_split=args.split,
                 output_image=img_path
                 )
        
def render_views(args,
                output_index=0,
                output_split='none',
                output_image='render.png',
                ):
    # Initialize scene for rendering
    IMG_WIDTH = 320
    IMG_HEIGHT = 240
    utils.init_scene_settings(args.base_scene_blendfile, 
                              args.material_dir, 
                              IMG_WIDTH, 
                              IMG_HEIGHT, 
                              args.render_tile_size, 
                              args.render_num_samples, 
                              args.render_min_bounces, 
                              args.render_max_bounces, 
                              args.use_gpu)

    # Open the properties file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba

    total_objs = len(properties['shapes'].items()) * len(color_name_to_rgba.items()) * len(properties['materials'].items())
    print("Rendering " + str(len(properties['shapes'].items())) + " objects, with " + str(len(color_name_to_rgba.items())) + " colors, and " + str(len(properties['materials'].items())) + " materials")
    print("Total of " + str(total_objs) + " distinct objects with " + str(args.cam_views) + " views, for " + str(args.cam_views * total_objs) + " total renders\n")
    # input("press enter to continue...")

    h5file = None
    if args.to_h5 is not None:
        print(f'Saving to h5 file {args.to_h5}')
        h5file = h5py.File(args.to_h5, 'w')

    for shape, _ in properties['shapes'].items():
        for color_pair in color_name_to_rgba.items():
            for material_pair in properties['materials'].items():
                for size_pair in properties['sizes'].items():
                    saved_files = render_combo(shape, color_pair, material_pair, size_pair, args.num_images)
                    if h5file is not None:
                        add_to_h5(h5_file, saved_files)

                    
    print("Done rendering scene!")

# def crop(img_pil: Image):
#     top = 70
#     left = 120
#     bottom = 166
#     right = 216
#     img_pil = img_pil.crop((left, top, right, bottom))
#     return img_pil

def add_to_h5(h5file, img_paths):
    ''' Adds the img paths into the h5 file 
    '''
    img_bytes_array = []
    for file in img_paths:
        img_pil = Image.open

        with open(path, 'rb') as img_f:
            binary_data = img_f.read()
    imgs_grp.create_dataset(str(i), data=np.asarray(binary_data))


def render_combo(shape, color_pair, material_pair, size_pair, num_images) -> list:
    ''' Renders the object with the given combination of shape, color, material,
    and size. Repeats the process num_images times
    - Returns: list of the file paths saved
    '''
    color, colorval = color_pair
    material, materialval = material_pair
    size, sizeval = size_pair
    # Create folder to store all views of object
    combination = shape + "_" + color + "_" + material + "_" + size + "/"
    render_dir = os.path.join(os.path.abspath(args.output_image_dir), combination)

    # Actually add the object to the scene
    utils.new_add_object(args.shape_dir, shape, sizeval, (0, 0))
    obj = bpy.context.object
    # shift by size / 2
    # obj.location = (obj.location[0] + obj.dimensions[0] / 2, obj.location[1] - obj.dimensions[1] / 2, obj.location[2])
    utils.add_material(obj, materialval, Color=colorval)
    saved_files = []
    for i in range(num_images):
        # Apply random rotation
        theta = 360.0 * random.random()
        obj.rotation_euler[2] = theta / 180 * math.pi
        bpy.context.view_layer.update()
        # Save images
        new_files = render_image(args.cam_views, i, render_dir)
        saved_files.extend(new_files)
    utils.delete_object(obj)
    return saved_files

def render_image(cam_views: int, index: int, render_dir: str) -> list:
    ''' Renders the current blender scene into a PNG, returning the list
    of filenames saved. 
    - cam_views: the number of camera views to render
    - index: the number of times this has been called for this combination of 
             properties, to make the saved file paths distinct
    - Returns: list of the file paths saved
    '''
    saved_files = []
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.use_zbuffer = True
    bpy.data.cameras['Camera'].lens = 58.5
    bpy.data.cameras['Camera'].lens_unit = "FOV"

    # Crop image
    #bpy.context.scene.render.use_border = True
    #bpy.context.scene.render.border_min_x = 120
    #bpy.context.scene.render.border_min_y = 70
    #bpy.context.scene.render.border_max_x = 216
    #bpy.context.scene.render.border_max_y = 166
    #bpy.context.scene.render.use_crop_to_border = True

    counter = 0
    cur = bpy.context.scene.render.filepath
    #cur = cur + str(index) + "camera"
    # cur = cur + "camera"
    for ob in bpy.context.scene.objects:
        if ob.type == "CAMERA":
            if ob.name == "Camera":
                # Skip original CLEVR camera
                continue
            elif ob.name == "Camera.005":
                # Skip top camera
                continue
            else:
                counter += 1
                if counter > cam_views:
                    # Skip when created enough views
                    continue
            bpy.context.scene.render.filepath = render_dir + '/' + str(index) + '.png'
            bpy.context.scene.camera = ob
            while True:
                try:
                    bpy.ops.render.render(write_still=True, use_viewport=True)
                    save_file = bpy.context.scene.render.filepath
                    if not save_file.endswith('.png'): save_file += '.png'
                    saved_files.append(save_file)
                    break
                except Exception as e:
                    print("RENDER ERROR: " + str(e))
    return saved_files


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        # Run normally
        args = parser.parse_args()
        main(args)
