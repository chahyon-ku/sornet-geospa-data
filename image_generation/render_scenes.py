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

import OpenEXR
import cv2
import numpy as np
import random
import logging
import utils
from image_generation import nocs

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.
This file expects to be run from Blender like this:
blender --background --python render_images.py -- [arguments to this script]
"""

# The probability of an object being placed indepentently, contained, or supported
IND_PROB = 0.01
CON_PROB = 0.99
SUP_PROB = 0.01

# The text codes for how an object is used
CONTAINER = 'con'
SUPPORTER = 'sup'
CHILD = 'cld'
INDEPENDENT = 'ind'


try:
    import bpy, bpy_extras
    from bpy.types import Object as BpyObj
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
parser.add_argument('--properties_json', default='data/properties/bunny_easy_properties.json',
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
parser.add_argument('--output_image_dir', default='../output/bunny_easy/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/bunny_easy/',
                    help="The directory where output JSON scene structures will be stored. " +
                         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
                    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='../output/bunny_easy/',
                    help="The directory where blender scene files will be stored, if the " +
                         "user requested that these files be saved using the " +
                         "--save_blendfiles flag; in this case it will be created if it does " +
                         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=1,
                    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                         "each generated image to be stored in the directory specified by " +
                         "the --output_blend_dir flag. These files are not saved by default " +
                         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")
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
                    default=1)
parser.add_argument('--skip_ori_camera', action="store_true",
                    help="Use original camera or not, default yes.")
parser.add_argument('--store_depth', action="store_true",
                    help="Redirect the depth info into the alpha channel of the output " +
                         "PNG file for later processing. This will make the images " +
                         "appear somewhat transparent to standard image viewers.")
parser.add_argument('--render_nocs', type=bool, default=True)


def main(args):
    template = os.path.join(os.path.abspath(args.output_image_dir), '%06d')
    img_template = os.path.join(os.path.abspath(args.output_image_dir), template + '.png')
    scene_template = os.path.join(args.output_scene_dir, template + '.json')
    blend_template = os.path.join(os.path.abspath(args.output_blend_dir), template)

    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_scene_dir, exist_ok=True)
    if args.save_blendfiles:
        os.makedirs(args.output_blend_dir, exist_ok=True)

    all_scene_paths = []
    scene_generator = SceneGenerator(args)
    for i in range(args.num_images):
        img_path = img_template % (i + args.start_idx)
        scene_path = scene_template % (i + args.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = blend_template % (i + args.start_idx) if args.save_blendfiles else None
        scene_generator.render_scene(output_index=(i + args.start_idx),
                                     output_split=args.split,
                                     output_image=img_path,
                                     output_scene=scene_path,
                                     output_blendfile=blend_path)


class SceneGenerator:

    args: dict
    object_mapping: dict
    color_name_to_rgba: dict
    size_mapping: list
    material_mapping: list
    shape_color_combos: list
    dou_con_mat: np.ndarray
    dou_con_mat: np.ndarray
    tri_con_mat: np.ndarray
    tri_sup_mat: np.ndarray

    def __init__(self, args):
        self.args = args
        # Load the property file
        with open(args.properties_json, 'r') as f:
            properties = json.load(f)
        self.object_mapping = dict(properties['shapes'])
        self.color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            self.color_name_to_rgba[name] = rgba
        self.size_mapping = list(properties['sizes'].items())
        self.material_mapping = [(v, k) for k, v in properties['materials'].items()]

        self.shape_color_combos = None
        if args.shape_color_combos_json is not None:
            with open(args.shape_color_combos_json, 'r') as f:
                self.shape_color_combos = list(json.load(f).items())

        # Load the relationship matrices from the paths in the properties file
        prop_dir = os.path.dirname(os.path.abspath(args.properties_json))
        def load_matrix(key: str):
            if key not in properties: return None
            path = os.path.join(prop_dir, properties[key])
            if not os.path.isfile(path):
                print(f'[ERROR] cannot find matrix {path}')
                exit(-1)
            if path.endswith('.dat'):
                return np.loadtxt(path, dtype=int)
            elif path.endswith('.npy'):
                return np.load(path)
            else:
                return None
        # 2D matrices
        self.dou_con_mat = load_matrix("contain_mat")
        self.dou_sup_mat = load_matrix("support_mat")
        # 3D matrices
        self.tri_con_mat = load_matrix("can_contain_mat")
        self.tri_sup_mat = load_matrix("can_support_mat")
        self.relation_matrices = {'dou_con_mat': self.dou_con_mat,
                                  'dou_sup_mat': self.dou_sup_mat,
                                  'tri_con_mat': self.tri_con_mat,
                                  'tri_sup_mat': self.tri_sup_mat}
        # Object placer
        self.object_placer = utils.ObjectPlacer(args.shape_dir)

    def render_scene(self,
                     output_index=0,
                     output_split='none',
                     output_image='render',
                     output_scene='render_json',
                     output_blendfile=None
                     ):
        # Initialize the settings for scene rendering
        utils.init_scene_settings(self.args.base_scene_blendfile,
                                  self.args.material_dir,
                                  self.args.width,
                                  self.args.height,
                                  self.args.render_tile_size,
                                  self.args.render_num_samples,
                                  self.args.render_min_bounces,
                                  self.args.render_max_bounces,
                                  self.args.use_gpu)
        bpy.context.scene.render.filepath = output_image
        self.jitter_cam_lights()
        # This will give ground-truth information about the scene and its objects
        scene_struct = {
            'split': output_split,
            'image_index': output_index,
            'image_filename': os.path.basename(output_image),
            'objects': [],
            'directions': {},
        }
        init_scene_directions(scene_struct)

        # Now make some random objects
        num_objects = random.randint(self.args.min_objects, self.args.max_objects)
        objects, blender_objects = self.add_random_objects(scene_struct, num_objects, output_index)

        # Render the scene and dump the scene data structure
        scene_struct['objects'] = objects
        scene_struct['relationships'] = compute_directional_relationships(scene_struct)
        spacial_relationships = compute_spacial_relationships(scene_struct, self.relation_matrices, self.object_mapping)
        if spacial_relationships is not None:
            scene_struct['relationships'].update(spacial_relationships)

        self.render_image()

        with open(output_scene, 'w') as f:
            json.dump(scene_struct, f, indent=2)

        if output_blendfile is not None:
            bpy.ops.wm.save_as_mainfile(filepath=output_blendfile + '.blend')

    # Private functions
    ############################################################################

    def jitter_cam_lights(self):
        ''' Adds jitter (random location change) to the camera and lights for
        lighting and camera diversity between scenes
        '''
        def rand(L):
            return 2.0 * L * (random.random() - 0.5)

        # Add random jitter to camera position
        if self.args.camera_jitter > 0:
            for i in range(3):
                bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

        # Add random jitter to lamp positions
        if self.args.key_light_jitter > 0:
            for i in range(3):
                bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
        if self.args.back_light_jitter > 0:
            for i in range(3):
                bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
        if self.args.fill_light_jitter > 0:
            for i in range(3):
                bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    def add_random_objects(self, scene_struct, num_objects, output_index):
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
        camera = bpy.data.objects['Camera']
        logging.debug("Scene number %d " % output_index + "*" * 20)
        logging.debug(f'\twith {num_objects} objects')

        # positions = {}
        objects = []
        blender_objects = []
        for i in range(num_objects):
            # Choose random color and shape
            num_tries = 0
            while True:
                num_tries += 1
                if num_tries > args.max_retries:
                    logging.debug("EXCEDED MAX RETRIES FOR CHOOSING OBJECT")
                    return objects, blender_objects

                # Place in container
                if self.shape_color_combos is None:
                    obj_name, obj_num = random.choice(list(self.object_mapping.items()))
                    color_name, rgba = random.choice(list(self.color_name_to_rgba.items()))
                else:
                    obj_name, color_choices = random.choice(self.shape_color_combos)
                    color_name = random.choice(color_choices)
                    obj_num = self.object_mapping[obj_name]
                    rgba = self.color_name_to_rgba[color_name]
                # Index number of this object
                # Choose a random material and size
                mat_name, mat_name_out = random.choice(self.material_mapping)
                size_name, r = random.choice(self.size_mapping)

                has_identical_object = False
                for object in objects:
                    if object['shape'] == obj_name and object['color'] == color_name\
                            and object['material'] == mat_name_out and object['size'] == size_name:
                        has_identical_object = True
                        break

                if not has_identical_object:
                    logging.debug("Object %d: %s %s %s" % (i, color_name, mat_name_out, obj_name))
                    break
            # Choose how the new object is placed into the scene
            container, fix, supporter, cur_used, parent_index = self.choose_placement_type(obj_num, objects, blender_objects)
            # Try to place the object, ensuring that we don't intersect any existing
            # objects and that we are more than the desired margin away from all existing
            # objects along all cardinal directions.
            num_tries = 0
            while True:
                num_tries += 1
                if num_tries > args.max_retries:
                    logging.debug("EXCEDED MAX RETRIES")
                    logging.debug(f'object: {obj_name}')
                    logging.debug(f'container: {container}')
                    logging.debug(f'supporter: {supporter}')
                    # logging.debug(str(objects[-1]))
                    # logging.debug(f'location: {x, y, z} rotation: {theta}')
                    logging.debug(f'fix: {fix}\n')
                    return objects, blender_objects

                new_blender_object = self.object_placer.place_with_arbitrary_pose(obj_name, r)
                # Done if successfully placed object
                if new_blender_object is not None:
                    break
                # Could not successfully place, delete and try again
                utils.delete_object(bpy.context.active_object)
            # Record placement of new object
            # positions.append((x, y, r))
            utils.add_material(new_blender_object, mat_name, Color=rgba)

            # Record data about the object in the scene data structure
            pixel_coords = utils.get_camera_coords(camera, new_blender_object.location)
            # Add the object to objects
            new_object = {
                'shape': obj_name,
                'color': color_name,
                'material': mat_name_out,
                'obj_name_out': new_blender_object.name,
                '3d_coords': tuple(new_blender_object.location),
                'rotation': [new_blender_object.rotation_euler[0]/math.pi*180,
                             new_blender_object.rotation_euler[1]/math.pi*180,
                             new_blender_object.rotation_euler[2]/math.pi*180], # convert to degrees
                'pixel_coords': pixel_coords,
                # changes by Ku for objects supported by a contained object also being contained
                'index': len(objects),
                'container_index': None,
                'supporter_index': None,
                'contained_indices': [],
                'supported_indices': [],
                # add depth
                'size': size_name,
                'size_val': r,
                'depth': None
            }

            if container is not None: # contain
                new_object['container_index'] = parent_index
                container['contained_indices'].append(len(objects))
                new_object['depth'] = container['depth'] + 1
            # elif supporter is not None and supporter['container_index'] is not None: # obj sup by con obj is also con
            #     ancester_index = supporter['container_index']
            #     ancester_blender_object = blender_objects[ancester_index]
            #     new_top = new_blender_object.location[2]# + new_blender_object.dimensions[2]/2
            #     ancester_top = ancester_blender_object.location[2] + ancester_blender_object.dimensions[2]/2
            #     if new_top <= ancester_top:
            #         new_object['container_index'] = ancester_index
            #         objects[ancester_index]['contained_indices'].append(len(objects))
            elif supporter is not None: # support
                new_object['supporter_index'] = parent_index
                supporter['supported_indices'].append(len(objects))
                new_object['depth'] = supporter['depth'] + 1
            else:
                new_object['depth'] = 1

            blender_objects.append(new_blender_object)
            objects.append(new_object)

        # Check that all objects are at least partially visible in the rendered image
        # all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
        # if not all_visible:
        #   # If any of the objects are fully occluded then start over; delete all
        #   # objects from the scene and place them all again.
        #   print('Some objects are occluded; replacing objects')
        #   for obj in blender_objects:
        #     utils.delete_object(obj)
        #   return add_random_objects(scene_struct, num_objects, args, camera)
        logging.debug(f'finished scene {output_index}')
        return objects, blender_objects

    def choose_placement_type(self, shape_num, objects, blender_objects):
        ''' Chooses how to place the new object: either independently, supported on 
        another object, or contained in another object. 
        Args:
            shape_num: the number identifier of the current shape
            objects: dict containing all objects already in the scene
        Returns:
            container: if contained was chosen, the object dict for the container,
                        None if contained was not chosen
            fix: if contained was chosen, a boolean indicating if the current child
                object perfectly fits inside of the parent container. When true
                the child must always be placed in the center of the parent 
                with the same rotation
            supporter: if supported was chosen,, the object dict for the supporter,
                        None if supported was not chosen
            cur_used: a string representing how the current object was used, 'cld'
                    if contained or supported, 'non' if placed independently
            parent_idx: the index of parent object in the list of objects
        '''
        # Check all existing objects in the scene to see if this new object 
        # could be supported or contained
        possible_containers = []
        possible_supporters = []
        for obj_i, obj in enumerate(objects):
            if len(obj['contained_indices']) > 0 or len(obj['supported_indices']) > 0:
                # Object already supporting or containing something
                continue
            if obj['depth'] >= 4: # max depth is 4
                continue
            other_num = self.object_mapping[obj['shape']]
            con_val = self.can_contain(other_num, shape_num)
            sup_val = self.can_support(other_num, shape_num)
            if con_val > 0:
                possible_containers.append(obj)
            if sup_val > 0:
                if obj['container_index'] is None or blender_objects[obj_i].location[2] + blender_objects[obj_i].dimensions[2] > \
                        blender_objects[obj['container_index']].location[2] + blender_objects[obj['container_index']].dimensions[2]:
                    possible_supporters.append(obj)
        logging.debug('Possible supporters: ')
        logging.debug(['%s %s %s' % (x['color'], x['material'], x['shape']) for x in possible_supporters])
        logging.debug('Possible containers:')
        logging.debug(['%s %s %s' % (x['color'], x['material'], x['shape']) for x in possible_containers])
        # Relationship output variables
        container = None
        fix = False
        supporter = None
        # Likelyhoods of placement
        ind_prob = IND_PROB
        con_prob = CON_PROB
        sup_prob = SUP_PROB
        # Zero probabilities if relationship impossible
        if not possible_containers:
            con_prob = 0.0
        if not possible_supporters:
            sup_prob = 0.0
        # Normalize probabilities
        total_prob = ind_prob + con_prob + sup_prob
        ind_prob /= total_prob
        con_prob /= total_prob
        sup_prob /= total_prob
        # Choose relationship for placement
        rand_val = random.random()
        if rand_val <= con_prob:
            # Place on container
            container = random.choice(possible_containers)
            assert(container is not None)
            logging.debug('Placing in container: %s %s %s' % (container['color'], container['material'], container['shape']))
            cur_used = CHILD
            container_shape_num = self.object_mapping[container['shape']]
            fix = self.can_contain(container_shape_num, shape_num) == 2
            parent_idx = objects.index(container)
        elif rand_val <= con_prob+sup_prob:
            # Place on supporter
            supporter = random.choice(possible_supporters)
            assert(supporter is not None)
            logging.debug('Placing on supporter: %s %s %s' % (supporter['color'], supporter['material'], supporter['shape']))
            cur_used = CHILD
            parent_idx = objects.index(supporter)
        else:
            # Independent
            logging.debug('Placing independently')
            cur_used = INDEPENDENT
            parent_idx = None
        return container, fix, supporter, cur_used, parent_idx

    def compute_spacial_relationships(self, objects, blender_objects):
        '''
        Computes contain and support relationships between all pairs of objects 
        in the scene. Returns a dictionary mapping string relationship names to 
        lists of lists of integers, where output[rel][i] gives a list of object
        indices that have the relationship rel with object i. For example if j is
        in output['can_contain'][i] then object i can contain object j.
        '''
        relation_matrices = [self.dou_con_mat, self.dou_sup_mat, 
                             self.tri_con_mat, self.tri_sup_mat]
        if all(matrix is None for matrix in relation_matrices):
            # No contain or relationship matrices, exit now
            return None
        all_relationships = {'can_contain': [], 'can_support': []}

        # Add the object index and value to the predicate list
        def add_val(lst: list, obj: int, val: int):
            val = int(min(1, val))
            assert(-1 <= val <= 1)
            lst.append({'obj': obj, 'val': val})
        # For each obj1, check if it can contain or support each other obj2
        for i, obj1 in enumerate(objects):
            shape_num1 = self.object_mapping[obj1['shape']]
            can_contain = []
            can_support = []
            child = None
            child_num = None
            # Find if the parent already has a child
            if obj1['used'] in [CONTAINER, SUPPORTER]:
                child_idx = obj1['child']
                child = objects[child_idx]
                child_num = self.object_mapping[child['shape']]
                # Add can contain/support predicates for the child's current
                # relation, then if the child is not a supporter it can be
                # moved, so check if the other relation would also apply
                child_supporter: bool = child['used'] == SUPPORTER
                if obj1['used'] == CONTAINER:
                    add_val(can_contain, child_idx, 1)
                    if not child_supporter:
                        sup_val = self.can_support(shape_num1, child_num)
                        add_val(can_support, child_idx, sup_val)
                else:
                    add_val(can_support, child_idx, 1)
                    if not child_supporter:
                        con_val = self.can_contain(shape_num1, child_num)
                        add_val(can_contain, child_idx, con_val)
            # Check if obj1 is already inside of a container
            obj1_contained = in_container(obj1, objects)
            obj1_sticks_out: bool = False
            if obj1_contained:
                # Check if the object sticks out of the top of the container
                obj1_sticks_out = sticks_out(obj1, objects, blender_objects)
            for j, obj2 in enumerate(objects):
                # Check not the same object or child
                if obj1 == obj2 or obj2 == child: continue
                # Check obj2 isn't already being used as a supporter (then it
                # couldn't be moved to be supported or contained by anything else)
                if obj2['used'] == 'sup': continue
                # Get the can contain value for the objects
                shape_num2 = self.object_mapping[obj2['shape']]
                con_val = self.can_contain(shape_num1, shape_num2, child_num)
                # Don't worry abt existing child for can-support relationships
                # if it's entirely inside of the parent
                if child_num is not None and in_container(child, objects) and \
                        not sticks_out(child, objects, blender_objects):
                    sup_val = self.can_support(shape_num1, shape_num2)
                else:
                    sup_val = self.can_support(shape_num1, shape_num2, child_num)
                # If the top of obj1 is inside a container, the other shape must
                # also be able to fit inside of the container
                if obj1_contained and not obj1_sticks_out:
                    parent_num = self.object_mapping[objects[obj1['parent']]['shape']]
                    sup_val = sup_val and self.can_contain(parent_num, shape_num2)
                add_val(can_contain, j, con_val)
                add_val(can_support, j, sup_val)
            all_relationships['can_contain'].append(can_contain)
            all_relationships['can_support'].append(can_support)
        return all_relationships

    def can_contain(self, parent: int, other1: int, current_child: int = None) -> int:
        ''' Uses the precomputed contain relationship matrices to determine if 
        the other1 object can be contained inside of the parent. If current_child
        is not None then the value will be computed considering the existing
        child in the scene (if both other1 and current_child can fit into the 
        parent together). 
        '''
        if current_child is None:
            if self.dou_con_mat is None: return 0 
            return self.dou_con_mat[parent][other1]
        else:
            if self.tri_con_mat is None: return 0
            return self.tri_con_mat[parent][current_child][other1]

    def can_support(self, parent: int, other1: int, current_child: int = None) -> int:
        ''' Uses the precomputed support relationship matrices to determine if 
        the other1 object can be supported on top of of the parent. If current_child
        is not None then the value will be computed considering the existing
        child in the scene (if both other1 and current_child can fit on top of 
        the parent together). 
        '''
        if current_child is None:
            if self.dou_sup_mat is None: return 0 
            return self.dou_sup_mat[parent][other1]
        else:
            if self.tri_sup_mat is None: return 0
            return self.tri_con_mat[parent][current_child][other1]

    def render_image(self):
        ''' Renders the current scene to PNG file(s) based on the camera
        arguments, rendering a different PNG image for each camera view.
        '''
        bpy.data.cameras['Camera'].lens = 58.5
        bpy.data.cameras['Camera'].lens_unit = "FOV"
        # ~.png -> ~
        cur = bpy.context.scene.render.filepath[:-4]
        for mode in range(3):
            if mode == 0:
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                suffix = ''
            elif mode == 1:
                bpy.context.scene.cycles.samples = 1
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                nocs.render_nocs()
                suffix = '_nocs'
            elif mode == 2:
                bpy.context.scene.cycles.samples = 1
                # depth.render_depth()
                bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
                bpy.context.scene.render.image_settings.use_zbuffer = True
                suffix = '_depth'
            for ob in bpy.context.scene.objects:
                if ob.type == "CAMERA":
                    # ['Camera', 'Camera.001', 'Camera.002', 'Camera.003', 'Camera.004', 'Camera.005']
                    i_camera = int(ob.name.split('.')[1]) if '.' in ob.name else 0
                    if i_camera == 0 and not self.args.skip_ori_camera:
                        bpy.context.scene.camera = ob
                        bpy.context.scene.render.filepath = cur + '_ori' + suffix
                        bpy.ops.render.render(write_still=True, use_viewport=True)
                    elif 0 < i_camera <= self.args.side_camera:
                        bpy.context.scene.camera = ob
                        bpy.context.scene.render.filepath = cur + f'_side{i_camera}' + suffix
                        bpy.ops.render.render(write_still=True, use_viewport=True)
                    elif i_camera == 5 and self.args.top_camera:
                        bpy.context.scene.camera = ob
                        bpy.context.scene.render.filepath = cur + '_top' + suffix
                        bpy.ops.render.render(write_still=True, use_viewport=True)
                    else:
                        continue

                    if mode == 2:
                        f = OpenEXR.InputFile(bpy.context.scene.render.filepath + '.exr')
                        zs = np.frombuffer(f.channel('Z'), np.float32).reshape(self.args.height, self.args.width)
                        f.close()

                        r = np.ceil(np.log2(zs))
                        g = zs / np.power(2, r)
                        b = g * 256 - np.trunc(g * 256)
                        a = b * 256 - np.trunc(b * 256)
                        rgba = np.stack([r + 128, g * 256, b * 256, a * 256], axis=-1).astype(np.uint8)

                        new_zs = 2 ** (rgba[:, :, 0] - 128) * (rgba[:, :, 1] / 256 + rgba[:, :, 2] / 256 ** 2 + rgba[:, :, 3] / 256 ** 3)
                        cv2.imwrite(bpy.context.scene.render.filepath + '.png', rgba)
                        os.remove(bpy.context.scene.render.filepath + '.exr')


# Static functions
############################################################################

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


def compute_spacial_relationships(scene_struct, relation_matrices, object_mapping):
    '''
    Computes contain and support relationships between all pairs of objects
    in the scene. Returns a dictionary mapping string relationship names to
    lists of lists of integers, where output[rel][i] gives a list of object
    indices that have the relationship rel with object i. For example if j is
    in output['can_contain'][i] then object i can contain object j.
    '''
    all_relationships = {'contain': [], 'support': [], 'can_contain': [], 'can_support': []}
    for i, obj1 in enumerate(scene_struct['objects']):
        contain = set()
        support = set()
        can_contain = set()
        can_support = set()

        for j in obj1['contained_indices']: # contain
            contain.add(j)
        for j in obj1['supported_indices']: # support
            support.add(j)
        for j, obj2 in enumerate(scene_struct['objects']): # can_contain
            if i == j: continue # same object
            if len(obj2['supported_indices']) > 0: continue # obj2 supporting others
            if j in obj1['contained_indices']: continue # obj2 already contained
            shape1 = object_mapping[obj1['shape']]
            shape2 = object_mapping[obj2['shape']]
            if len(obj1['contained_indices']) > 1: # >1 other contained
                continue
            elif len(obj1['contained_indices']) == 1: # 1 other contained
                k = obj1['contained_indices'][0]
                obj3 = scene_struct['objects'][k]
                shape3 = object_mapping[obj3['shape']]
                if relation_matrices['tri_con_mat'][shape1][shape2][shape3] >= 1:
                    can_contain.add(j)
            else: # 0 other contained
                if relation_matrices['dou_con_mat'][shape1][shape2] >= 1:
                    can_contain.add(j)
        for j, obj2 in enumerate(scene_struct['objects']): # can_support
            if i == j: continue # same object
            if len(obj2['supported_indices']) > 0: continue # obj2 supporting others
            if j in obj1['supported_indices']: continue # obj2 already supported
            shape1 = object_mapping[obj1['shape']]
            shape2 = object_mapping[obj2['shape']]
            if len(obj1['supported_indices']) > 1: # >1 other supported
                continue
            elif len(obj1['supported_indices']) == 1: # 1 other supported
                k = obj1['supported_indices'][0]
                obj3 = scene_struct['objects'][k]
                shape3 = object_mapping[obj3['shape']]
                if relation_matrices['tri_sup_mat'][shape1][shape2][shape3] >= 1:
                    can_support.add(j)
            else: # 0 other supported
                if relation_matrices['dou_sup_mat'][shape1][shape2] >= 1:
                    can_support.add(j)

        all_relationships['contain'].append(sorted(list(contain)))
        all_relationships['support'].append(sorted(list(support)))
        all_relationships['can_contain'].append(sorted(list(can_contain)))
        all_relationships['can_support'].append(sorted(list(can_support)))

    return all_relationships


def compute_directional_relationships(scene_struct, eps=0.001):
    """
    Computes relationships between all pairs of objects in the scene.
    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords1[k] - coords2[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    # print(scene_struct)
    # print(all_relationships)
    # exit(0)
    return all_relationships


def sticks_out(obj: dict, objects: list, blender_objects: list):
    ''' Check if obj sticks out of the container, returns True if the obj
    sticks out of it's parent. Checks if the top of the containee is above
    the top of the container.
    '''
    assert(in_container(obj, objects)), "%s not in container, used=%s" % (obj['obj_name_out'], obj['used'])
    i = objects.index(obj)
    this_obj = blender_objects[i]
    parent_idx = obj['parent']
    assert(parent_idx is not None), "%s has no parent, used=%s" % (obj['obj_name_out'], obj['used'])
    parent = blender_objects[parent_idx]
    this_top = this_obj.location[2] + this_obj.dimensions[2]/2
    parent_top = parent.location[2] + parent.dimensions[2]/2
    return this_top >= parent_top

def in_container(obj: dict, objects: list):
    ''' Checks the objects to see if the obj's parent is a container
    '''
    if obj['parent'] is None: return False
    return objects[obj['parent']]['used'] == CONTAINER


if __name__ == '__main__':
    log_file = 'render.log'
    if os.path.isfile(log_file):
        os.remove(log_file)
    logging.basicConfig(format='%(message)s', filename=log_file, level=logging.DEBUG)

    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        # Run normally
        args = parser.parse_args()
        main(args)


# Archived functions (not used any more)
################################################################################

def ind_overlap_ori(cur_obj, cur_pos, con_obj, sup_obj, positions, scene_struct):
    # Check to make sure the new object is further than min_dist from all
    # other objects, and further than margin along the four cardinal directions
    x, y, r = cur_pos
    dists_good = True
    margins_good = True
    for (xx, yy, rr) in positions.values():
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
            dists_good = False
            break
        for direction_name in ['left', 'right', 'front', 'behind']:
            direction_vec = scene_struct['directions'][direction_name]
            assert direction_vec[2] == 0
            margin = dx * direction_vec[0] + dy * direction_vec[1]
            if 0 < margin < args.margin:
                print(margin, args.margin, direction_name)
                print('BROKEN MARGIN!')
                margins_good = False
                return False
        if not margins_good:
            return False

    if dists_good and margins_good:
        return True


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; thisy
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.
    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3])
                          for i in range(0, len(p), 4))
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors
