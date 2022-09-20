import subprocess
import sys
import os
import argparse
import json
from tkinter.tix import REAL
import numpy as np
import h5py
from encode_ground_truth import create_relation_matrix

# Relations to include in the h5 output
# Same order as in the h5 file, i.e. the 2nd relation here will be the 2nd matrix
RELATIONS = ['right', 'front', 'contain',
             'support', 'can_contain', 'can_support']

OUT_JSON_PATH = '../output/scenes/'
OUT_IMG_PATH = '../output/images/'
OUT_H5_PATH = '../output/h5outputfile.h5'

IMG_WIDTH = 480
IMG_HEIGHT = 320

PY_FILE = '../render_scenes.py'

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=None, type=int,
                    help="The number of scenes to render each time that a blender" +
                         "subprocess is called i.e. number of scene metadata" +
                         " files that exist before being cleaned up.")
parser.add_argument('--num_scenes', default=10, type=int,
                    help="Total number of scenes to render.")


def main(args):
    # Stop if files already exist
    if check_existing_files():
        return
    # Create an h5 file
    if os.path.isfile(OUT_H5_PATH):
        os.remove(OUT_H5_PATH)
    h5file = h5py.File(OUT_H5_PATH, 'a')
    if args.batch_size is None:
        args.batch_size = min(10, args.num_scenes)

    process_parameters = ['python', PY_FILE]
    for scene_index in range(0, args.num_scenes, args.batch_size):
        batch_size = min(args.batch_size, args.num_scenes-scene_index)
        python_args = [
            '--num_images', str(batch_size), '--width', str(IMG_WIDTH), '--height', str(IMG_HEIGHT)]
        print(
            f"Scenes {scene_index} to {scene_index+batch_size}: {process_parameters+python_args}")
        subprocess.call(process_parameters+python_args)
        encode_output(h5file, scene_index, batch_size)
        delete_files()
    h5file.close()

    print(f"FINISHED GENERATING {args.num_scenes} SCENES")
    h5_size = os.path.getsize(OUT_H5_PATH)
    print('hdf5 file size:\t\t\t%dkB' % (h5_size/1000))
    print("SUCCESS")


def check_existing_files() -> bool:
    '''
    Checks if existing images or json files exist. Returns true if any do,
    false otherwise.
    '''
    # Check no existing output files
    imgs_exist = os.path.isdir(OUT_IMG_PATH) and len(
        os.listdir(OUT_IMG_PATH)) > 0
    json_exist = os.path.isdir(OUT_JSON_PATH) and len(
        os.listdir(OUT_JSON_PATH)) > 0
    if imgs_exist:
        print("There are images in the output folder:")
        print(OUT_IMG_PATH)
    if json_exist:
        print("There are json files in the output folder:")
        print(OUT_JSON_PATH)
    if imgs_exist or json_exist:
        print("These files will be overwritten during this process. Please delete " +
              "or move them to start.")
        if input('\'Y\' to delete files now...') == 'Y':
            delete_files()
            return False
        return True
    return False


# Keywords for encoding scene_information
SCENE_KEY = '%06d'
IMAGE_KEY = 'image'
OBJECTS_KEY = 'objects'
RELATIONS_KEY = 'relations'


def encode_output(h5file: h5py.File, start_idx: int, num_scenes: int):
    '''
    Looks through the output image and json files, encoding them into a 
    single h5 file. Takes num_scenes: the expected number of scenes which were 
    generated.
    '''
    # Get image and json scene paths
    img_paths = os.listdir(OUT_IMG_PATH)
    jsn_paths = os.listdir(OUT_JSON_PATH)
    # Check expected number of images and scenes
    assert(len(img_paths) ==
           num_scenes), f'was {len(img_paths)} images, expected {num_scenes}'
    assert(len(jsn_paths) ==
           num_scenes), f'was {len(jsn_paths)} images, expected {num_scenes}'

    img_paths.sort()
    jsn_paths.sort()
    for i in range(num_scenes):
        scene_id = SCENE_KEY % (i + start_idx)
        scene_grp = h5file.create_group(scene_id)
        img_path = os.path.join(OUT_IMG_PATH, img_paths[i])
        json_path = os.path.join(OUT_JSON_PATH, jsn_paths[i])
        # Check the image and json file match
        batch_idx = SCENE_KEY % i
        assert(
            batch_idx in img_path), f'image path {img_path} missing scene index {batch_idx}'
        assert(
            batch_idx in json_path), f'json path {json_path} missing scene index {batch_idx}'
        # Add the image
        with open(img_path, 'rb') as img_f:
            binary_data = img_f.read()
        scene_grp.create_dataset(IMAGE_KEY, data=np.asarray(binary_data))
        # Get predicates and objects from json
        with open(json_path) as json_f:
            json_data = json.load(json_f)
        # Add the objects
        objects = []
        for obj in json_data['objects']:
            objects.append('%s_%s_%s' %
                           (obj['color'], obj['material'], obj['shape']))
        obj_str = ','.join(objects)
        print(f'objects: {obj_str}')
        scene_grp.create_dataset(
            OBJECTS_KEY, data=np.array(obj_str, dtype='S'))
        # Add the relations
        rel_arr = create_relation_matrix(json_data)
        scene_grp.create_dataset(RELATIONS_KEY, data=rel_arr)


def delete_files():
    '''
    Deletes all files in the output directories
    '''
    for dir in [OUT_IMG_PATH, OUT_JSON_PATH]:
        for file in os.listdir(dir):
            os.remove(os.path.join(dir, file))


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        args = parser.parse_args()
        main(args)
