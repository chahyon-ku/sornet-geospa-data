'''
This script goes through a set of objects and pre-generates the can contain and 
can support predicates. These predicates are stored in four (4) different numpy 
matrices: 2-D can contain, 2-D can support, 3-D can contain, 3-D can support.

There must be an input properties.json file so the code knows which set of objects
to use and the ID index for each object, then the four output matrices of this 
file must be appended to the JSON file before it can be used for image generation.
'''

from pyparsing import ParseElementEnhance
try:
    import bpy, bpy_extras
    from bpy.types import Object as BpyObj
except ImportError as e:
    print(e)
import argparse
import json
import os
import utils
import numpy as np
from mathutils import Vector
from typing import Tuple
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()

# File paths
parser.add_argument('set_name',
                    help="The name for the set of objects to distinguish the generated " +
                         "matrices from those generated for different sets of objects.")
parser.add_argument('--save_directory', default='data/',
                    help="Directory to save the matrices.")

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

parser.add_argument('--save_uncertain_scenes', action='store_true',
                    help="If set, will save uncertain states (when it's unclear " +
                    "if a can contain relationship is possible) into blend files " +
                    "in the output folder")

# Values to store in the 3D matrices
FALSE = 0
TRUE = 1
UNCERTAIN = -1


def main(args):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    save_file_template = '%s_rels_%%s' % args.set_name
    save_file_template = os.path.join(args.save_directory, save_file_template)
    file_endings = ['contain_dou.dat', 'support_dou.dat', 'contain_tri.npy', 'support_tri.npy']
    save_files = [save_file_template % x for x in file_endings]
    for f in save_files: print(f'Saving: {f}')
    shapes = get_shapes(args.properties_json)
    load_all_objects(shapes, args.shape_dir)
    if not shapes_orientation_ok(shapes):
        exit(-1)
    bpy.ops.wm.save_as_mainfile(filepath='../output/kitchen_blendfile.blend')

    if os.path.isfile(save_files[0]):
        print(f'skipping {save_files[0]}, already exists')
    else:
        dou_can_contain = pregenerate_pair_can_contain(shapes)
        np.savetxt(save_files[0], dou_can_contain, '%d')

    if os.path.isfile(save_files[1]):
        print(f'skipping {save_files[1]}, already exists')
    else:
        dou_can_support = pregenerate_pair_can_support(shapes)
        np.savetxt(save_files[1], dou_can_support, '%d')

    tri_can_contain, tri_can_support = pregenerate_predicates(
        shapes, save_files[0], save_files[1], args.save_uncertain_scenes)

    np.save(save_files[2], tri_can_contain)
    np.save(save_files[3], tri_can_support)

    print("\nCopy and paste the below into the properties JSON file:\n\n")

    print(f"\t\"contain_mat\": \"{file_endings[0]}\",")
    print(f"\t\"support_mat\": \"{file_endings[1]}\",")
    print(f"\t\"can_contain_mat\": \"{file_endings[2]}\",")
    print(f"\t\"can_support_mat\": \"{file_endings[3]}\"")
    print()

def get_shapes(properties_json: str) -> list:
    ''' Loads the list of shapes from the JSON properties file and returns the
    list of shape names from that file
    - properties_json: the path to the properties JSON file
    - returns: list of shape names
    '''
    with open(properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
    return properties['shapes'].items()


def load_all_objects(shapes: list, shape_dir: str) -> None:
    ''' Loads all the shapes from the list into the blender environment from 
    their .blend files.
    - shapes: the list of shape names to load
    - shape_dir: path to directory with shape .blend files
    '''
    for shape_name, shape_index in shapes:
        filename = os.path.join(shape_dir, f'{shape_name}.blend', 'Object', shape_name)
        bpy.ops.wm.append(filename=filename)
        print(bpy.data.objects[shape_name])
    print('Done adding objects')


def shapes_orientation_ok(shapes: list) -> bool:
    ''' Checks all objects to ensure the longer end of the object is oreinted
    along the x-axis. Prints and warning if not all objects meet the requirement.
    - returns: true if all objects oriented correctly, false otherwise
    '''
    all_ok = True
    for shape_name, shape_index in shapes:
        x, y, _ = bpy.data.objects[shape_name].dimensions
        if y > x:
            print(f'WARNING long end along y axis: {shape_name}')
            all_ok = False
    if not all_ok:
        print('ERROR: Not all objects oriented correctly. Fix object .blend ' +
              'files and try again')
        return False
    print('All objects oriented correctly with long end along x-axis.')
    return True


def pregenerate_pair_can_contain(shapes: list) -> np.ndarray:
    dou_can_contain = np.zeros((len(shapes), len(shapes)), dtype=np.int8)
    print('Generating can contain for pairs of objects...')
    # Iterate through all object pairs
    for parent_shape_name, parent_shape_index in shapes:
        parent_obj = bpy.data.objects[parent_shape_name]
        parent_length, parent_width, parent_height = parent_obj.dimensions
        # The depth is the measurement of the first point downward on the 
        # object, which should be the interior of the container (if it has an 
        # interior) which should be less than the bounding box of the object
        margin = 0.001 # 1mm margin on the objects 
        round_val = 6  # Round to 6 decimal points
        parent_depth: float = round(utils.get_container_depth(parent_obj), round_val)
        if abs(parent_depth) >= abs(round(parent_height, round_val)/2) - margin:
            print(f'{parent_shape_name} is NOT a container')
            continue
        else:
            print(f'{parent_shape_name} IS a container')
        
        for child_shape_name, child_shape_index in shapes:
            if parent_shape_index == child_shape_index: continue
            # Reflexive can contain relationship is impossible
            if dou_can_contain[child_shape_index, parent_shape_index] != 0: continue
            # Get child dimensions
            child_obj = bpy.data.objects[child_shape_name]
            child_length, child_width, child_height = child_obj.dimensions
            # Check child bigger
            if child_length >= parent_length or child_length * child_width >= parent_length * parent_width:
                # Child too large
                dou_can_contain[parent_shape_index, child_shape_index] = FALSE
                continue
            # Check child taller 
            elif child_height/2 > parent_depth + margin:
                # Child too tall
                dou_can_contain[parent_shape_index, child_shape_index] = FALSE
                continue
            # Try placing the child
            place_bottom_container(parent_obj, child_obj)
            bpy.context.view_layer.update()
            if utils.check_collision(parent_obj, child_obj):
                # No collision
                dou_can_contain[parent_shape_index, child_shape_index] = TRUE
                print(f'\t{parent_obj.name} can contain {child_obj.name}')
                save_scene_object_state(parent_obj, child_obj)
            else:
                # Child in collision with parent
                dou_can_contain[parent_shape_index, child_shape_index] = FALSE
    return dou_can_contain


def pregenerate_pair_can_support(shapes: list) -> np.ndarray:
    dou_can_support = np.zeros((len(shapes), len(shapes)), dtype=np.int8)
    print('Generating can support predicates for pairs of objects...')
    # Iterate through all object pairs
    for parent_shape_name, parent_shape_index in shapes:
        parent_obj = bpy.data.objects[parent_shape_name]
        parent_length, parent_width, _ = parent_obj.dimensions
        for child_shape_name, child_shape_index in shapes:
            # Get child dimensions
            child_obj = bpy.data.objects[child_shape_name]
            child_length, child_width, _ = child_obj.dimensions
            if child_length <= parent_length and child_width <= parent_width:
                dou_can_support[parent_shape_index, child_shape_index] = 1
            elif child_length <= parent_width and child_width <= parent_length:
                dou_can_support[parent_shape_index, child_shape_index] = 1
            else:
                dou_can_support[parent_shape_index, child_shape_index] = 0
    return dou_can_support


def pregenerate_predicates(shapes: list, con_path: str, sup_path: str, save_uncertain_scenes: bool) -> Tuple[np.ndarray, np.ndarray]:
    ''' Iterates through every possible combination of 3 objects, filling the "can
    contain" and "can support" relationship matrices with values to indicate the 
    possibility of those relationships for the trio of objects. A value of 0 means
    the relationship is impossible, 1 means it is possible, and -1 is uncertainty.
    For example: can_contain_mat[parent_obj, child_obj, other_obj] == 1 indicates
    that child_obj and other_obj can simultaneously fit inside of 'parent_obj'.
    - shapes: the list of shape names
    - save_uncertain_scenes: if set saves blend files when can contain relationships
    between object triples are uncertain
    - returns: 3D matrices filled with can contain and can support relationship values
    '''
    # Load existing 2D relationships matrices from file
    con_rel_mat = np.loadtxt(con_path, dtype=int)
    sup_rel_mat = np.loadtxt(sup_path, dtype=int)
    # Create 3D arrays filled with garbage to fill with predicates
    mat_shape = (len(shapes), len(shapes), len(shapes))
    can_contain_mat = np.full(mat_shape, 2, dtype=np.int8)
    can_support_mat = np.full(mat_shape, 2, dtype=np.int8)
    print('Generating can contain and can support predicates for all objects...')
    # Iterate through all object triples
    for parent_shape_name, parent_shape_index in tqdm(shapes):
        parent_obj = bpy.data.objects[parent_shape_name]
        px, py, _ = parent_obj.dimensions
        p_max_dim = utils.get_max_inside_dim(parent_obj)
        # Center the parent without rotation
        parent_obj.location = 0, 0, 0
        parent_obj.rotation_euler = 0, 0, 0
        bpy.context.view_layer.update()
        logging.debug(f'\nParent: {parent_shape_name}\t' + '*' * 40)
        logging.debug(f'\tdimensions: {px, py}')
        logging.debug(f'\tmax inside dim: {p_max_dim}')
        for child_shape_name, child_shape_index in shapes:
            # Get child dimensions
            child_obj = bpy.data.objects[child_shape_name]
            cx, cy, _ = child_obj.dimensions
            child_con_bool = con_rel_mat[parent_shape_index, child_shape_index]
            child_sup_bool = sup_rel_mat[parent_shape_index, child_shape_index]
            cshort_dim = min(cx, cy)
            # Don't check any more contain relationships for this pair if:
            # 0 means child can't be contained, 2 means child perfectly fits/fills container
            cant_con_child: bool = child_con_bool == 0 or child_con_bool == 2
            if cant_con_child:
                can_contain_mat[parent_shape_index, child_shape_index, :] = FALSE
                can_contain_mat[parent_shape_index, :, child_shape_index] = FALSE
                logging.debug(f'No possible contain for {parent_shape_name} <- ' +
                              f'{child_shape_name}: child_con_bool={child_con_bool}')
            else:
                # Optimally place the child to maximize room for other object
                child_y_dim = optimally_contain(parent_obj, child_obj)
            # Don't check any more support relationships for this pair if:
            # 0 means child can't be supported, or parent is smaller than the child
            cant_sup_child = child_sup_bool == 0 or max(px, py) < cshort_dim/2
            if cant_sup_child:
                can_support_mat[parent_shape_index, child_shape_index, :] = FALSE
                can_support_mat[parent_shape_index, :, child_shape_index] = FALSE
                logging.debug(f'No possible support for {parent_shape_index} <- ' +
                              f'{child_shape_index}: child_sup_bool={child_sup_bool} parent_max_dim ' +
                              f'={max(px, py)} half_child_short_dim={cshort_dim/2}')
            if cant_sup_child and cant_con_child:
                # Don't check any more relationships for this pair
                logging.debug(f'No possible contain or support for ' +
                              '{parent_shape} <- {child_shape}')
                continue
            for other_shape_name, other_shape_index in shapes:
                # Skip if already calculated certain relationship for this pair
                other_obj = bpy.data.objects[other_shape_name]
                ox, oy, _ = other_obj.dimensions
                oshort_dim = min(ox, oy)
                if not cant_sup_child:  # Try checking if both objects fit on parent
                    sup_val = sup_rel_mat[parent_shape_index, other_shape_index]
                    min_dist_bet_childs = (cshort_dim + oshort_dim)/2
                    sup_val = sup_val and max(px, py) > min_dist_bet_childs
                    can_support_mat[parent_shape_index, child_shape_index, other_shape_index] = sup_val
                    can_support_mat[parent_shape_index, other_shape_index, child_shape_index] = sup_val
                if not cant_con_child:
                    # Skip if relationship already reflexively defined during
                    # previous iteration (when other_obj was a child_obj)
                    if can_contain_mat[parent_shape_index, child_shape_index, other_shape_index] in [TRUE, FALSE]:
                        continue
                    logging.debug(f'other shape: {other_shape_name}')
                    if con_rel_mat[parent_shape_index, other_shape_index] != 1:
                        # Other object can't fit in parent under any circumstances
                        logging.debug(f'contain mat false for other obj ' +
                                      f'{other_shape_name}: {con_rel_mat[parent_shape_index, other_shape_index]}')
                        fit_val = FALSE
                    elif p_max_dim < child_y_dim + oshort_dim:
                        # Sum of child and other short dimensions is greater
                        # than the inside dimension of the parent
                        logging.debug('sum of child and other short dims ' +
                                      'greater than parent inside')
                        logging.debug(f'p_max_dim={p_max_dim} cshort_dim=' +
                                      f'{cshort_dim} oshort_dim={oshort_dim}')
                        fit_val = FALSE
                    else:
                        fit_val = check_fit(parent_obj, child_obj, other_obj)
                        logging.debug(f'check_fit value={fit_val}')
                    can_contain_mat[parent_shape_index, child_shape_index, other_shape_index] = fit_val
                    can_contain_mat[parent_shape_index, other_shape_index, child_shape_index] = fit_val
                    if fit_val == UNCERTAIN:
                        logging.debug(
                            f'UNCERTAIN: {parent_shape_name, child_shape_name, other_shape_name}')
                        if save_uncertain_scenes:
                            save_scene_object_state(parent_obj,
                                                    child_obj, other_obj)

    print('Finished generating 3D predicate matrices for can contain and can support.')
    # Make sure all values in the matrices have been overwritten
    cmax = can_contain_mat.max()
    smax = can_support_mat.max()
    if (cmax > TRUE):
        print(f'ERROR: can contain matrix missed assignment: {cmax}')
    if (smax > TRUE):
        print(f'ERROR: can support matrix missed assignment: {smax}')
    cmin = can_contain_mat.min()
    smin = can_support_mat.min()
    if (cmin < FALSE):
        print(f'can contain matrix has expected uncertainty: {cmin}')
    if (smin < FALSE):
        print(f'ERROR: can support matrix has uncertainty: {smin}')
    return can_contain_mat, can_support_mat

def check_fit(parent_obj: BpyObj, child_obj: BpyObj, other_obj: BpyObj) -> int:
    """ Checks if another object can fit inside the parent with the child already
    contained (i.e. checks if both child and other can fit inside the parent
    together). Assumes the parent is at the origin with no rotation, and child 
    has already been moved to the far left side of the container.
    - parent_obj: the parent object
    - child_obj: the child object already contained
    - other_obj: the other object already contained
    - returns: FALSE if the object definitely cannot fit, TRUE if the object 
               definitely can fit, and UNCERTAIN if it's not sure
    """
    # When the shapes are rectangular this function can act with more certainty
    rectangle_keywords = ('rectangle', 'cube', 'thin', 'rect')
    # Relies on the assumption that the 2nd item in the parent's name indicates
    # the inside shape of the parent
    parent_rect_interior = parent_obj.name.startswith(rectangle_keywords) or \
        parent_obj.name.split('_')[1] in rectangle_keywords
    if child_obj == other_obj:
        # If object is across the center of the parent, then can't fit another
        # instance of the object (the object is already taking up more than half)
        cx, cy, _ = child_obj.dimensions
        glob_child_x_dim = cx if child_obj.rotation_euler[2] == 0 else cy
        x_side = child_obj.location[0] + glob_child_x_dim/2
        logging.debug(
            f'objects match: child_obj.location[0]={child_obj.location[0]} glob_child_x_dim={glob_child_x_dim} x_side={x_side}')
        if x_side <= 0:
            return True
        elif child_obj.name.startswith(rectangle_keywords):
            # If child is a rectangle then definitely cannot fit
            return False
        elif not parent_rect_interior:
            # If both parent and child are elliptical then definitely cannot fit
            return False
        else:
            return UNCERTAIN
    else:
        # Place the other obj on the far right side of the parent, assuming the
        # child is already on the far left, if the objects aren't touching
        # then there's room for both
        logging.debug(f'optimally_contain({parent_obj.name}, {other_obj.name}')
        optimally_contain(parent_obj, other_obj, 'right')
        objs_clear = utils.check_collision(child_obj, other_obj)
        logging.debug(
            f'objects {child_obj.name, other_obj.name} clear of eachother: {objs_clear}')
        if objs_clear:
            logging.debug(
                f'no overlap between child {child_obj.name} and other {other_obj.name}')
            return TRUE
        elif child_obj.name.startswith(rectangle_keywords) or \
                other_obj.name.startswith(rectangle_keywords):
            logging.debug('one or both of the children are rectangles')
            return FALSE
        elif not parent_rect_interior:
            # None of the three objects are rectangles
            logging.debug('none of the objects are rectangles')
            return FALSE
        else:
            return UNCERTAIN


def optimally_contain(parent_obj: BpyObj, child_obj: BpyObj, direction: str = 'left') -> None:
    ''' Wrapper for optimally contain so that an output .blender file is saved on
    an assertion error for debugging purposes
    '''
    try:
        val = real_optimally_contain(parent_obj, child_obj, direction)
    except AssertionError as e:
        # Remove all other objects
        for obj in bpy.data.objects:
            if obj not in [parent_obj, child_obj]:
                utils.delete_object(obj)
        filepath = '../output/pregenerate_predicates_error.blend'
        if os.path.isfile(filepath):
            os.remove(filepath)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)
        print(f'ERROR: {e}')
        print(f'Saving blender file to {filepath}')
        exit(-1)
    return val


def real_optimally_contain(parent_obj: BpyObj, child_obj: BpyObj, direction: str = 'left') -> float:
    ''' Tries to optimally place the child on the far left side of the parent, 
    to leave as much room as possible for trying to place another object to see
    if both will fit in the parent container.
    - parent_obj: the parent container object
    - child_obj: the child containee object
    - returns: the y dimension of the child object in global space
    '''
    glob_child_y_dim = orient_in_container(parent_obj, child_obj)
    toward_parent = Vector(
        (-1, 0, 0)) if direction == 'left' else Vector((1, 0, 0))
    toward_child = Vector(
        (1, 0, 0)) if direction == 'left' else Vector((-1, 0, 0))
    origin = child_obj.location
    child_top = origin + Vector((0, glob_child_y_dim/2-0.01, 0))
    logging.debug(f'child_top={child_top}')
    dist_to_parent = None

    child_mat = child_obj.matrix_world
    child_mat_inv = child_obj.matrix_world.inverted()
    parent_mat = parent_obj.matrix_world
    parent_mat_inv = parent_obj.matrix_world.inverted()

    # Measure from the center of the object then from the top of the object
    for pt in [origin, child_top]:
        logging.debug(f'staring point = {pt}')
        # Get point in parent space
        pt = parent_mat_inv @ pt
        # Get distance from object to inside of parent
        result, parent_loc, normal, idx, = parent_obj.ray_cast(
            pt, toward_parent)
        assert(result), f'could not ray cast to parent from origin {child_obj.name} ' +\
            f'to parent {parent_obj.name}'

        # Get location in world coordinates
        parent_loc = parent_mat @ parent_loc
        logging.debug(f'intersect on parent in global space = {parent_loc}\n')

        # Get parent point in child space
        outside_pt = child_mat_inv @ parent_loc
        logging.debug(f'ray vector start in child space = {outside_pt}')

        # Get right direction in child space
        child_right = child_mat_inv @ toward_child
        logging.debug(f'ray vector end in child space = {child_right}')
        result, child_loc, normal, idx, = child_obj.ray_cast(
            outside_pt, child_right)
        # Convert child point to global
        logging.debug(f'intersect on child in child space  = {child_loc}')
        child_loc = child_mat @ child_loc
        logging.debug(f'intersect on child in global space  = {child_loc}')
        assert(result), f'could not ray cast to child from parent {parent_obj.name} ' +\
            f'to child {child_obj.name}'
        # Update shortest distance between the child and parent
        assert(round(parent_loc[1], 5) == round(child_loc[1], 5)), \
            f'y-axis diff for intersections {parent_obj.name} <- {child_obj.name} {parent_loc[1]} != {child_loc[1]}'
        dist = (parent_loc-child_loc).length
        logging.debug(f'distance for given starting point {dist}\n')
        dist_to_parent = dist if dist_to_parent is None \
            else min(dist_to_parent, dist)
    logging.debug(f'distance to parent={dist_to_parent}')
    assert(dist_to_parent < parent_obj.dimensions[0])/2, \
        f"point outside parent {parent_obj.name} <- {child_obj.name}"
    # Move the child and check it fits
    margin = glob_child_y_dim * -0.02  # space to leave between parent and child
    new_x_loc = dist_to_parent - 0.01
    if 'left' in direction:
        new_x_loc *= -1
        margin *= -1
    # Try moving back until the object fits
    for _ in range(5):
        child_obj.location[0] = new_x_loc
        bpy.context.view_layer.update()
        logging.debug(f'moving {child_obj.name} to x coordinate {new_x_loc}')
        if utils.check_collision(parent_obj, child_obj):
            break
        new_x_loc += margin
    bpy.context.view_layer.update()
    assert(utils.check_collision(parent_obj, child_obj)), \
        'child overlaps after being placed ' +\
        f'{parent_obj.name} <- {child_obj.name}'
    logging.debug('')
    return glob_child_y_dim


def orient_in_container(parent_obj: BpyObj, other_obj: BpyObj) -> float:
    ''' Tries to orient the other object inside of the containg parent object
    so that the long dimension of the other object is perpendicular to the long
    dimension of the parent (to minimize space taken along the long dimension 
    of the parent). Will orient the other object with the parent (long dimensions 
    of both along same axis), if it doesn't fit the short way.
    - returns: the dimension of the other obj along the global/parent y-axis
    '''
    place_bottom_container(parent_obj, other_obj)
    child_x_dim, child_y_dim, _ = other_obj.dimensions
    # Try placing child long way in parent's short dimension
    other_obj.rotation_euler = 0, 0, math.pi/2  # 90 degrees to parent
    bpy.context.view_layer.update()
    # child dimension along y-axis, different from child y-dimension
    glob_child_y_dim = child_x_dim
    if not utils.check_collision(parent_obj, other_obj):
        # Won't fit this way, return to rotation aligned with parent
        other_obj.rotation_euler[2] = 0  # Return to default rotation
        bpy.context.view_layer.update()
        glob_child_y_dim = child_y_dim
    # Sanity check the object should fit when rotation aligned with parent
    assert(utils.check_collision(parent_obj, other_obj)), \
        'could not orient child in parent' +\
        f'\n\t{parent_obj.name} <- {other_obj.name}'
    return glob_child_y_dim


def place_bottom_container(parent_obj: BpyObj, other_obj: BpyObj) -> None:
    ''' Places the other_obj so that the bottom of the object is above the bottom
    of the container and the two objects are not in collision along the z-axis.
    '''
    start = Vector((0, 0, 0))
    direction = Vector((0, 0, -1))
    result, parent_loc, _, _, = parent_obj.ray_cast(
        start, direction)
    assert(result), f'could not cast to bottom of {parent_obj.name}'
    glob_loc = parent_obj.matrix_world @ parent_loc
    height = glob_loc[2] + other_obj.dimensions[2]/2 + 0.01
    other_obj.location = 0, 0, max(height, 0)


def simple_add_obj(object_dir: str, shape_name: str, scale: float) -> BpyObj:
    ''' Adds an object to center/origin of the scene.
    - returns: the added object
    '''
    filename = os.path.join(object_dir, '%s.blend' %
                            shape_name, 'Object', shape_name)
    bpy.ops.wm.append(filename=filename)
    obj = bpy.data.objects[shape_name]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.transform.resize(value=(scale, scale, scale))
    obj.location = 0, 0, 0
    obj.rotation_euler = 0, 0, 0
    return obj


def save_scene_object_state(parent_obj: BpyObj, child_obj: BpyObj, other_obj: BpyObj=None):
    ''' Saves the current state of the objects to a blender file. Doesn't bother
    actually deleting all the other objects, just hides them. 
    - args: the objects to leave visible in the blender file
    '''
    # Hide all other objects
    for obj in bpy.data.objects:
        if obj not in [parent_obj, child_obj, other_obj]:
            obj.hide_set(True)
    # Save the blender file
    dir = '../output/generate_predicates_blenders/'
    if other_obj is None:
        file_name = f'{parent_obj.name}-{child_obj.name}.blend'
    else:
        file_name = f'{parent_obj.name}-{child_obj.name}-{other_obj.name}.blend'
    output_blendfile = os.path.join(dir, file_name)
    if not os.path.isdir('../output/generate_predicates_blenders/'):
        os.mkdir(dir)
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    # Show the objects again
    for obj in bpy.data.objects:
        if obj not in [parent_obj, child_obj, other_obj]:
            obj.hide_set(False)

################################################################################


def test_pregenerate_predicates(parent_shape: str, child_shape: str,
                                other_shape: str):
    ''' DEBUGGING: Optimally place a child in the top left side of the paren
    and render the image of the parent and child
    - parent_shape: the string indicating the parent's shape
    - child_shape: the string indicating the child's shape
    '''
    print(f'Testing with parent: {parent_shape} child: {child_shape}')
    output_file = os.path.abspath(
        '../output/test_pregenerate_predicates_%s_%s')
    utils.init_scene_settings(args.base_scene_blendfile,
                              args.material_dir,
                              args.width,
                              args.height,
                              args.render_tile_size,
                              args.render_num_samples,
                              args.render_min_bounces,
                              args.render_max_bounces,
                              args.use_gpu)
    utils.delete_object(bpy.data.objects['Ground'])  # Remove the ground
    with open(args.properties_json, 'r') as f:     # Get properties
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
    sizeval = list(properties['sizes'].items())[0][1]  # only one size
    materialval = list(properties['materials'].items())[0][1]  # rubber
    parent_colorval = color_name_to_rgba['red']  # red
    child_colorval = color_name_to_rgba['blue']  # blue
    other_colorval = color_name_to_rgba['green']  # blue
    # Add parent and child objects
    parent_obj = simple_add_obj(args.shape_dir, parent_shape, sizeval)
    utils.add_material(parent_obj, materialval, Color=parent_colorval)
    # Add child
    child_obj = simple_add_obj(args.shape_dir, child_shape, sizeval)
    utils.add_material(child_obj, materialval, Color=child_colorval)
    # Add other
    other_obj = simple_add_obj(args.shape_dir, other_shape, sizeval)
    utils.add_material(other_obj, materialval, Color=other_colorval)
    error = None
    try:
        # Place both objects on opposite sides of the container
        optimally_contain(parent_obj, child_obj, 'left')
        optimally_contain(parent_obj, other_obj, 'right')
    except AssertionError as e:
        error = e
    finally:
        # Save the blend file
        if error is None:
            print(f'SUCCEEDED: no errors!')
        else:
            print(f'ERROR: {error}')
        output_blendfile = '../output/test_pregenerate_predicates.blend'
        if os.path.isfile(output_blendfile):
            os.remove(output_blendfile)
        print(f'Saving blendfile to {output_blendfile}')
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


################################################################################

if __name__ == '__main__':
    import logging
    logging_path = '../output/pregenerate_predicates.log'
    if os.path.isfile(logging_path):
        os.remove(logging_path)
    logging.basicConfig(format='%(message)s',
                        filename=logging_path, level=logging.DEBUG)
    args = parser.parse_args()
    main(args)
    print(
        f'logged debug at: {logging.getLoggerClass().root.handlers[0].baseFilename}')
