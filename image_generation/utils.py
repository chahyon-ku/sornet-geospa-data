# -*- coding: utf-8 -*-

import sys, random, os
import bpy.types
import bpy, bpy_extras
import numpy as np
from bpy.types import Object as BpyObj
from mathutils.bvhtree import BVHTree
from mathutils import Vector
import logging

"""
Some utility functions for interacting with Blender
"""
import math


class ObjectPlacer:
    '''
	Places objects from a specific directory in a variety of ways.
	'''

    def __init__(self, shape_dir: str):
        self.shape_dir = shape_dir

    def place_independently(self, shape_name: str, scale: float) -> BpyObj:
        ''' Tries to place the shape randomly on the table. Returns the newly
		added shape on sucess, or None on failure.
		'''
        x = random.uniform(-2.2, 2.2)
        y = random.uniform(-2.2, 2.2)
        theta = 360.0 * random.random()
        new_add_object(self.shape_dir, shape_name, scale, (x, y), theta=theta)
        obj = bpy.context.active_object
        if check_all_collisions(obj):
            return obj
        else:
            return None

    def place_in_container(self, shape_name: str, scale: float, container: BpyObj, fix: bool):
        ''' Tries to place a new shape inside a container already in the blender
		scene. Returns the newly added shape on sucess, or None on failure.
		Args:
		shape_name: the name of the new shape to place inside the container
		scale: the scale/size/dimension of the new shape to place
		container: the already placed bpy object to act as the container
		fix: if the new shape perfectly fits in the container
		'''
        loc = container.location
        max_dim = get_max_inside_dim(container)
        z = loc[2] + 0.01 - get_container_depth(container)
        if fix:
            x = loc[0]
            y = loc[1]
            theta = container.rotation_euler[2] * 180 / math.pi
        else:
            in_container = False
            while not in_container:
                x = loc[0] + random.uniform(-max_dim / 2, max_dim / 2)
                y = loc[1] + random.uniform(-max_dim / 2, max_dim / 2)
                in_container = inside((x, y, z), container)
            theta = 360.0 * random.random()
        new_add_object(self.shape_dir, shape_name, scale, (x, y), theta=theta, zvalue=z)
        obj = bpy.context.active_object
        # Don't check collisions for fit is true
        if fix or check_collision(obj, container):
            return obj
        else:
            return None

    def place_on_supporter(self, shape_name: str, scale: float, supporter: BpyObj):
        '''
		Tries to place a new shape on top of a supporter already in the blender
		scene. Returns the newly added shape on sucess, or None on failure.
		Args:
			shape_name: the name of the new shape to place inside the container
			scale: the scale/size/dimension of the new shape to place
			supporter: the already placed bpy object to act as the supporter
		'''
        loc = supporter.location
        px = supporter.dimensions[0]
        py = supporter.dimensions[1]
        pz = supporter.dimensions[2]
        z = loc[2] + pz / 2 + 0.01  # Always just above the surface of the object
        # Get x and y values
        on_top = False
        while not on_top:
            dx = random.uniform(-px / 2, px / 2)
            dy = random.uniform(-py / 2, py / 2)
            x = loc[0] + dx
            y = loc[1] + dy
            on_top = above_sup((x, y, z), supporter)
        theta = 360.0 * random.random()
        new_add_object(self.shape_dir, shape_name, scale, (x, y), theta=theta, zvalue=z)
        obj = bpy.context.active_object
        if check_all_collisions(obj):
            return obj
        else:
            return None

    def place_with_arbitrary_pose(self, shape_name: str, scale):
        x = 0#random.uniform(-2.0, 2.0)
        y = 0#random.uniform(-2.0, 2.0)
        z = 0
        count = 0
        for obj in bpy.data.objects:
            if obj.name.startswith(shape_name):
                count += 1

        filename = os.path.join(self.shape_dir, '%s.blend' % shape_name, 'Object', shape_name)
        bpy.ops.wm.append(filename=filename)

        # Give it a new name to avoid conflicts
        new_name = '%s_%d' % (shape_name, count)
        bpy.data.objects[shape_name].name = new_name

        # Set the new object as active, then rotate, scale, and translate it
        bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
        bpy.context.object.rotation_euler[0] = random.random() * np.pi * 2
        bpy.context.object.rotation_euler[1] = random.random() * np.pi * 2
        bpy.context.object.rotation_euler[2] = random.random() * np.pi * 2
        bpy.ops.transform.resize(value=scale)
        dz = np.sqrt(np.sum((np.array(bpy.context.view_layer.objects.active.dimensions) / 2) ** 2))
        bpy.context.object.location = x, y, dz + z
        bpy.context.view_layer.update()  # Update object for new location and rotation
        obj = bpy.context.active_object

        return obj


def get_max_inside_dim(obj: BpyObj) -> float:
    ''' Gets the max interior dimension of the object by casting a ray outward
	from the center of the object along the x-axis and getting the distance.
	Assumes the object is centered at the origin and the long dimension is along
	the x-axis.
	- obj: the object to evaluate
	- returns: the maximum interior dimension of the object
	'''
    start = Vector((0, 0, 0))
    direction = Vector((1, 0, 0))
    result, parent_loc, normal, idx, = obj.ray_cast(start, direction)
    assert (result), f'could not cast ray to get interior of {obj.name}'
    glob_loc = obj.matrix_world @ parent_loc
    dist = (obj.location - glob_loc).length * 2
    return dist


def get_container_depth(obj: BpyObj) -> float:
    ''' Gets the distance from the center of the object to the nearest point
	directly downward on it's surface.
    Returns the distance from the center of the object to the first point reached
    on the objects surface when a ray is drawn downward from above the center of 
    the object. If the object is not a container this will be the top of the object.
	'''
    start = Vector((0, 0, obj.dimensions[2]))
    direction = Vector((0, 0, -1))
    result, parent_loc, _, _, = obj.ray_cast(
        start, direction)
    if not result:
        print(f'WARNING: could not cast ray to bottom of {obj.name}')
        return obj.dimensions[2]/2
    glob_loc = obj.matrix_world @ parent_loc
    dist = (obj.location - glob_loc).length
    assert (obj.location[:2] == glob_loc[:2])
    return dist


def inside(pt, obj: BpyObj):
    '''
	Checks that the 'pt' is inside the object
	'''
    point = Vector(pt)
    pre_dist = (obj.location - point).length  # distance between object's center and points
    smwi = obj.matrix_world.inverted()
    local_point = smwi @ point
    result, local_close, _, _ = obj.closest_point_on_mesh(local_point)
    close = obj.matrix_world @ local_close
    post_dist = (obj.location - close).length
    return pre_dist < post_dist


def above_sup(pt, obj: BpyObj) -> bool:
    '''
	Checks that 'pt' is on top of 'obj' by making sure a ray cast directly down
	from pt intersects with the obj
	- pt: the x, y, z location of the object we're trying to place
	- obj: the parent object to check that the child object is above
	- returns: true if the point is above the object, false otherwise
	'''
    loc = Vector(pt)  # get the vector form of the location
    mw = obj.matrix_world  # get the tranform matrix for the object
    origin = mw.inverted() @ loc  # convert the location into the obj's scope
    direction = (0, 0, -1)  # cast a ray directly downward
    intersection, _, _, _ = obj.ray_cast(origin, direction)
    return intersection


def init_scene_settings(base_scene_blendfile, material_dir, width,
                        height, render_tile_size, render_num_samples, render_min_bounces,
                        render_max_bounces, use_gpu):
    """
    Loads the blendfile and initializes the render settings. Initializes the
    materials to use, the image resolution, gpu usage, etc...
    """
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=base_scene_blendfile)

    # Load materials
    load_materials(material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        elif bpy.app.version < (2, 80, 0):
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.get_devices()
            cycles_prefs.compute_device_type = 'CUDA'
        else:
            cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
            cycles_prefs.get_devices()
            cycles_prefs.compute_device_type = 'OPTIX'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'


def redirect_depth_to_alpha():
    ''' Redirects the depth output of the scene into the alph channel of the
	PNG image. In a regular image viewer this makes the PNG appear transparent
	the further the object is from the camera. The depth info can be 
	easily extracted from the PNG alpha channel later. 
	'''
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    rl = tree.nodes.new('CompositorNodeRLayers')
    norm = tree.nodes.new('CompositorNodeNormalize')
    com = tree.nodes.new('CompositorNodeComposite')
    com.use_alpha = True
    links.new(rl.outputs['Image'], com.inputs['Image'])  # link Renger Image to Viewer Image
    links.new(rl.outputs['Depth'], norm.inputs['Value'])  # Normalize depth
    links.new(norm.outputs['Value'], com.inputs['Alpha'])  # link Render Z to Viewer Alpha


def check_collision(obj1, obj2):
    ''' Checks if the two objects are in collision
	- obj1, obj2: the blender objects to compare
	- returns: false if the objects overlap, true if there's no overlap
	'''
    # Get their world matrix
    mat1 = obj1.matrix_world
    mat2 = obj2.matrix_world

    # Get the geometry in world coordinates
    vert1 = [mat1 @ v.co for v in obj1.data.vertices]
    poly1 = [p.vertices for p in obj1.data.polygons]

    vert2 = [mat2 @ v.co for v in obj2.data.vertices]
    poly2 = [p.vertices for p in obj2.data.polygons]

    # Create the BVH trees
    bvh1 = BVHTree.FromPolygons(vert1, poly1)
    bvh2 = BVHTree.FromPolygons(vert2, poly2)
    # Test if overlap
    if bvh1.overlap(bvh2):
        return False
    else:
        return True


def check_all_collisions(obj: BpyObj) -> bool:
    '''Checks that the object is not in collision with any other objects in the scene
	- obj: the blender object to compare against others
	- returns: true if not in collision with any other objects, false if in collision
	'''
    for other_obj in bpy.data.objects:
        # Don't check if same object, or a camera or light
        if other_obj == obj or other_obj.type != 'MESH' or other_obj.name == 'Ground':
            continue
        if not check_collision(obj, other_obj):
            logging.debug(f'{obj.name} in collision with {other_obj.name}')
            return False
    return True


def extract_args(input_argv=None):
    """
	Pull out command-line arguments after "--". Blender ignores command-line flags
	after --, so this lets us forward command line arguments from the blender
	invocation to our own script.
	"""
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
    """ Delete a specified blender object """
    bpy.data.objects.remove(obj, do_unlink=True)


def get_camera_coords(cam, pos):
    """
	For a specified point, get both the 3D coordinates and 2D pixel-space
	coordinates of the point from the perspective of the camera.
	Inputs:
	- cam: Camera object
	- pos: Vector giving 3D world-space position
	Returns a tuple of:
	- (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
		in the range [-1, 1]
	"""
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def set_layer(obj, layer_idx):
    """ Move an object to a particular layer """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)


def new_add_object(object_dir, name, scale, loc, theta=0, zvalue=0):
    """
	Load an object from a file. We assume that in the directory object_dir, there
	is a file named "$name.blend" which contains a single object named "$name"
	that has unit size and is centered at the origin.
	- object_dir: directory with the shape .blend files
	- scale: scalar giving the size that the object should be in the scene
	- loc: tuple (x, y) giving the coordinates on the ground plane where the
	object should be placed.]
	- theta: the angle of the object
	- zvalue: the height to place the bottom of the object, default 0 means
              place on the table/ground
	"""
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta / 180 * math.pi
    bpy.ops.transform.resize(value=scale)
    dz = bpy.context.view_layer.objects.active.dimensions[2] / 2
    bpy.context.object.location = x, y, dz + zvalue
    bpy.context.view_layer.update()  # Update object for new location and rotation


def add_object(objects, object_dir, name, scale, loc, theta=0, zvalue=0, sup=False):
    """
	Load an object from a file. We assume that in the directory object_dir, there
	is a file named "$name.blend" which contains a single object named "$name"
	that has unit size and is centered at the origin.
	- scale: scalar giving the size that the object should be in the scene
	- loc: tuple (x, y) giving the coordinates on the ground plane where the
	object should be placed.
	"""
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    objects[-1]['obj_name_out'] = new_name
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.view_layer.objects.active.location[2] = 0
    bpy.context.object.rotation_euler[2] = theta / 180 * math.pi
    bpy.ops.transform.resize(value=(scale, scale, scale))

    # Move all the objects above the ground
    dz = bpy.context.view_layer.objects.active.dimensions[2] / 2
    if not sup:
        # NOTE: "convertViewVec: called in an invalid context" error here not sure why
        bpy.ops.transform.translate(value=(x, y, dz + zvalue))
    else:
        bpy.ops.transform.translate(value=(x, y, zvalue + bpy.context.view_layer.objects.active.dimensions[2] / 2))


def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'): continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def add_material(cur_obj, name, **properties):
    """
	Create a new material and assign it to the active object. "name" should be the
	name of a material that has been previously loaded using load_materials.
	"""
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    # obj = bpy.context.active_object
    obj = cur_obj
    # print("FINDING ERROR")
    # print(len(obj.data.materials))
    # for m in obj.data.materials:
    obj.data.materials.clear()
    # print("FINDING ERROR")
    # print(len(obj.data.materials))
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )
