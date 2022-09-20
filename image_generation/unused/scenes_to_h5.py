'''
Contains the logic for converting the JSON and PNG outputs of the image_generation.py
into a single h5 file containing multiple scenes. This is required because SORNet
is designed to use h5 files rather than the JSON and PNG output of the CLEVR.
'''

import io
import os
import json
import time
from PIL import Image
import numpy as np
import h5py

OUT_DIR_PATH = '../../geospa_scenes_50/'
OUT_H5_PATH = '../../geospa_scenes_50/train.h5'

IMG_WIDTH = 480
IMG_HEIGHT = 320

# Keywords for encoding scene_information
INCLUDE_RELATIONS = ['front', 'right']


def encode_output(f: h5py.File):
    '''
    Looks through the output image and json files, encoding them into a 
    single h5 file. Takes num_scenes: the expected number of scenes which were 
    generated.
    '''
    # Verify output files
    img_paths = []
    jsn_paths = []
    for dir in os.listdir(OUT_DIR_PATH):
        if dir[-3:] == 'png':
            img_path = os.path.join(OUT_DIR_PATH, dir)
            jsn_path = os.path.join(OUT_DIR_PATH, dir[:-14] + '.json')
            time_since_modify = time.time() - os.path.getmtime(img_path)
            if time_since_modify > 10:
                img_paths.append(img_path)
                jsn_paths.append(jsn_path)

    n_scenes = len(f.keys())
    print(n_scenes)

    # Add info for each scene
    img_paths.sort()
    jsn_paths.sort()
    for i, (img_path, jsn_path) in enumerate(zip(img_paths, jsn_paths)):
        scene_key = '%06d' % (i + n_scenes)
        scene_grp = f.create_group(scene_key)
        # Check the image and json file match

        # Add the image
        with Image.open(img_path) as img_pil:
            buf = io.BytesIO()
            img_pil.save(buf, 'png')
        scene_grp.create_dataset('image', data=np.array(buf.getvalue()))

        # Get predicates and objects from json
        with open(jsn_path) as json_f:
            json_data = json.load(json_f)
        # Add the objects
        objects = []
        object_names = []
        for obj in json_data['objects']:
            object_names.append(obj['obj_name_out'])
            objects.append('%s_%s_%s' %
                           (obj['color'], obj['material'], obj['shape']))
        obj_str = ','.join(objects)
        #print(f'objects: {obj_str}')
        scene_grp.create_dataset('objects', data=np.array(obj_str, dtype='S'))
        # Add the relations
        num_objs = len(json_data['objects'])
        rel_arr_shape = (4, 9, 10)  # shape of relations array
        # First front, back, right, left relations
        rel_arr = np.zeros(rel_arr_shape, dtype=np.int8)
        for arr_idx, rel in enumerate(INCLUDE_RELATIONS):
            rel_list = json_data['relationships'][rel]
            for first_obj_idx, other_idxs in enumerate(rel_list):
                rel_arr[arr_idx, other_idxs, first_obj_idx] = 1
        # Then contain support relations
        for obj_idx, obj_dict in enumerate(json_data['objects']):
            if obj_dict['child'] is None:
                continue
            child_idx = object_names.index(obj_dict['child']['obj_name_out'])
            if obj_dict['used'] == 'sup':
                rel_arr[-1, obj_idx, child_idx] = 1
            elif obj_dict['used'] == 'con':
                rel_arr[-2, obj_idx, child_idx] = 1
        scene_grp.create_dataset('relations', data=rel_arr)

    for i, (img_path, json_path) in enumerate(zip(img_paths, jsn_paths)):
        os.remove(img_path)
        os.remove(json_path)


if __name__ == '__main__':
    with h5py.File(OUT_H5_PATH, 'w') as f:
        while True:
            encode_output(f)
            f.flush()
            time.sleep(1)



    # if '--help' in sys.argv or '-h' in sys.argv:
    #     parser.print_help()
    # else:
    #     args = parser.parse_args()
    #     main(args)
