
import argparse
import io
import os
import json
import time
from PIL import Image
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument('--in-dir', type=str)  # ../output/train/
parser.add_argument('--h5-path', type=str)  # ../output/train.h5
args = parser.parse_args()


def encode_output(f: h5py.File):
    '''
    Looks through the output image and json files, encoding them into a
    single h5 file. Takes num_scenes: the expected number of scenes which were
    generated.
    '''
    # Verify output files
    img_paths = []
    jsn_paths = []
    for dir in os.listdir(args.in_dir):
        if dir[-4:] == 'json':
            jsn_path = os.path.join(args.in_dir, dir)
            img_path = [os.path.join(args.in_dir, dir[:-5] + 'camera_ori.png'),
                        os.path.join(args.in_dir, dir[:-5] + 'camera_top.png')]
            if not os.path.exists(jsn_path) or not os.path.exists(img_path[0]) or not os.path.exists(img_path[1]):
                continue
            time_since_modify = time.time() - os.path.getmtime(jsn_path)
            if time_since_modify > 10:
                img_paths.append(img_path)
                jsn_paths.append(jsn_path)

    n_scenes = len(f.keys())
    print(n_scenes, len(img_paths))

    # Add info for each scene
    img_paths.sort()
    jsn_paths.sort()
    for i, (img_path, jsn_path) in enumerate(zip(img_paths, jsn_paths)):
        # if i + n_scenes >= 10000:
        #     break
        print(i, img_path, jsn_path)
        scene_key = '%06d' % (i + n_scenes)
        scene_grp = f.create_group(scene_key)
        images_grp = scene_grp.create_group('images')
        # Check the image and json file match

        # Add the image
        for path_i, path in enumerate(img_path):
            with Image.open(path) as img_pil:
                buf = io.BytesIO()
                img_pil.save(buf, 'png')
            images_grp.create_dataset(f'{path_i}', data=np.array(buf.getvalue()))

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
        scene_grp.create_dataset('objects', data=np.array(obj_str, dtype='S'))
        # Add the relations
        rel_arr_shape = (len(json_data['relationships']), 9, 10)  # shape of relations array
        # First front, back, right, left relations
        rel_arr = np.zeros(rel_arr_shape, dtype=np.half)
        for rel_idx, rel in enumerate(json_data['relationships']):
            rel_list = json_data['relationships'][rel]
            for first_obj_idx, other_idxs in enumerate(rel_list):
                rel_arr[rel_idx, first_obj_idx, other_idxs] = 1

        scene_grp.create_dataset('relations', data=rel_arr)

    # for i, (img_path, json_path) in enumerate(zip(img_paths, jsn_paths)):
    #     os.remove(img_path)
    #     os.remove(json_path)


if __name__ == '__main__':
    with h5py.File(args.h5_path, 'w') as f:
        while True:
            encode_output(f)
            f.flush()
            time.sleep(1)
            break
