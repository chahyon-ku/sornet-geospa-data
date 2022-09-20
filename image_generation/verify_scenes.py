'''
This file will use the output of render_images (either PNG and JSON or h5) to save
new images with the ground truth information in text on the image. This is useful
for quickly visualizing scenes with their ground truth information for verification
or debugging.
'''

import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import io
import os
import sys
import argparse
import json
from image_generation.unused.encode_ground_truth import RELATIONS, DIRECTIONAL_RELATIONS, RELATION_PHRASES, create_relation_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--num_scenes', default=100, type=int,
                    help="Total number of scenes to vizualize and save.")
parser.add_argument('--path', default='../../sornet/data/geospa_half_2view/val_default.h5',
                    help="Path to h5 file or EXR files to verify")
parser.add_argument('--save_dir', default='../../sornet/geospa_half_2view/val_default_scenes/',
                    help='Directory where to save the output images.')
parser.add_argument('--exclude_directions', action="store_true",
                    help="Exclude the left, right, behind, front predicates " +
                    "from the text output")
parser.add_argument('--save_depth', action="store_true",
                    help="Save the depth data to a seperate PNG file")


def main(args):
    # Check arguments
    num_scenes = -1
    directory_mode = False
    hf = None
    img_paths, jsn_paths = None, None
    if os.path.isdir(args.path):
        # User provided directory to PNG files
        print(f'Reading from directory: {args.path}')
        directory_mode = True
        img_paths = [f for f in os.listdir(args.path) if f.endswith('.png') and "camera_ori" in f]
        jsn_paths = [f for f in os.listdir(args.path) if f.endswith(".json")]
        assert(len(img_paths) == len(jsn_paths)), \
            f"Found {len(img_paths)} images but {len(jsn_paths)} json files"
        img_paths.sort()
        jsn_paths.sort()
        img_paths = img_paths[:args.num_scenes]
        jsn_paths = jsn_paths[:args.num_scenes]
        num_scenes = len(img_paths)
    else:
        # User provided H5 file
        print(f'Reading from H5 file: {args.path}')
        assert(os.path.isfile(args.path)), "Path argument is not a file."
        assert(args.path.lower().endswith('.h5')), "Path not to an h5 file"
        hf = h5py.File(args.path, 'r')
        scene_keys = list(hf.keys())[:args.num_scenes]
        assert(scene_keys[0] == '000000'), "Expected first scene to have 0 key"
        num_scenes = len(scene_keys)

    # Create save directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.exclude_directions:
        print("Excluding the directional predicates.")

    # For each scene
    print(f"{num_scenes} scenes")
    for scene_idx in range(num_scenes):
        # Get the image and objects
        imgs: np.ndarray = None
        relation_text: str = None
        if directory_mode:
            img_path = os.path.join(args.path, img_paths[scene_idx])
            jsn_path = os.path.join(args.path, jsn_paths[scene_idx])
            imgs, relation_text = extract_data_dir(args, img_path, jsn_path)   
        else:
            scene = hf[scene_keys[scene_idx]]
            imgs, relation_text = extract_data_h5(args, scene)
        save_path = os.path.join(args.save_dir, '%06d.png' % scene_idx)
        #depth_data = np.copy(img[:, :, 3])
        #img = img[:, :, :3]

        # Create the figure
        plt_height = max(5, len(relation_text) / 6)
        fig = plt.figure(figsize=(10, plt_height))
        # fig, (a0, a1) = plt.subplots(1, 2, figsize=(
        #     10, plt_height), gridspec_kw={'width_ratios': [2, 1]})
        # Plot 1
        # a0 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
        # Plot 2
        # a1 = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
        # Plot 3
        # fig, (a0, a1) = plt.subplots(1, 2, figsize=(
        #     10, plt_height),  gridspec_kw={'width_ratios': [2, 1]})

        # Add the scene views
        num_images = len(imgs)
        for i in range(num_images):
            img_axis = plt.subplot2grid((num_images, 2), (i, 0), rowspan=1)
            img_axis.imshow(imgs[i])
            img_axis.axis('off')

        # Add the relation text to the image
        a2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        a2.text(0, 0.5, '\n'.join(relation_text), color=(
            0, 0, 0), fontsize=10, ha='left', va='center')

        # Save the figure
        a2.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        # Save the depth data
        # if args.save_depth:
        #     depth_save_path = os.path.join(args.save_dir, '%06d_depth.png' % scene_idx)
        #     plt.imshow(depth_data)
        #     fig, ax = plt.subplots(1, 2)
        #     # fig.set_size_inches(15, 15)
        #     # Add color image plot
        #     ax[0].imshow(img)
        #     ax[0].set_title('Color')
        #     ax[0].axis('off')
        #     # Add depth image plot
        #     ax[1].imshow(depth_data)
        #     ax[1].set_title('Depth')
        #     ax[1].axis('off')
        #     plt.tight_layout()
        #     fig.savefig(depth_save_path)
        #     plt.close(fig)
    print(f'done saving verification images to directory {args.save_dir}')

def relation_text_from_matrix(objects: list, rel_data: np.ndarray, exclude_directions: bool) -> list:
    ''' Returns a list of strings describing the relationships
    between the objects in the scene.
    Args:
        objects: a list of strings, where each string describes the object
            for that index
        rel_data: the relationships numpy array of size num_relationships x 
            num_objects x num_objects
    Returns: a list of strings with the ground truth info for the scene
    '''
    tab = ' ' * 4
    relation_text = []
    for i, parent in enumerate(objects):
        relation_text.append(parent)
        relation_exists = False
        for k, rel in [(k, rel) for k, rel in enumerate(RELATIONS[:rel_data.shape[0]])]:
            if exclude_directions and rel in DIRECTIONAL_RELATIONS: continue
            for j, child in enumerate(objects):
                if i == j or rel_data[k, i, j] == 0: 
                    continue
                text = tab + f'{RELATION_PHRASES[rel]} {child}'
                relation_text.append(text)
                relation_exists = True
        if not relation_exists:
            relation_text.append(
                tab + f'no relations')
    return relation_text

def extract_data_h5(args, scene: dict):
    ''' Gets the image and ground truth information from an h5 file for a single
    scene.
    Args:
        scene: the scene dictionary from the h5 file containing both the image 
               and ground truth info
    Returns:
        img: the numpy array of the images RGB values
        relation_text: list of strings with the ground truth information 
    '''
    idatas = [np.array(scene['images'][str(i)]) for i in range(2)]
    imgs = [mpimg.imread(io.BytesIO(idata)) for idata in idatas]
    objects = np.array(scene['objects'])
    objects = str(objects)[2:-1].split(',')
    # Get the relations
    rel_data = np.array(scene['relations'])
    relation_text = relation_text_from_matrix(objects, rel_data, args.exclude_directions)
    return imgs, relation_text


def extract_data_dir(args, img_path: str, jsn_path: str):
    ''' Gets the image and ground truth text for a single scene from a directory.
    Uses the image and JSON file with ground truth data seperately.
    Args:
        img_path: the path to the PNG image
        jsn_path: the path to the JSON ground truth file
    Returns:
        img: the numpy array of the images RGB values
        relation_text: list of strings with the ground truth information
    '''
    with open(jsn_path) as json_f:
        json_data = json.load(json_f)
    objects = ['%s_%s_%s' % (
        obj['color'], obj['material'], obj['shape']) for obj in json_data['objects']]
    rel_data = create_relation_matrix(json_data)
    relation_text = relation_text_from_matrix(objects, rel_data, args.exclude_directions)
    
    if img_path.lower().endswith('.png'):
        img = mpimg.imread(img_path)
    else:
        print('\nERROR')
        print(f'could not recognize file type {img_path}')
        exit()
    imgs = [img]
    # Try to find other views of the same scene index
    scene_id: str = img_path[-20:-14]
    other_paths = []
    for path in os.listdir(args.path):
        if 'camera_ori' not in path and path.endswith('.png') and scene_id in path:
            other_paths.append(os.path.join(args.path, path))
    for other_p in other_paths:
        imgs.append(mpimg.imread(other_p))
    return imgs, relation_text

if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        args = parser.parse_args()
        main(args)
