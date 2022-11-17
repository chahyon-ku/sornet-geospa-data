
import argparse
import io
import os
import json
import time
from PIL import Image
import numpy as np
import h5py
from scipy.spatial.transform.rotation import Rotation

parser = argparse.ArgumentParser()
parser.add_argument('--in-dir', type=str, default='../output/mug_train/')
parser.add_argument('--include_top', type=bool, default=False)
parser.add_argument('--h5-path', type=str, default='../output/mug_train_5000.h5')
args = parser.parse_args()


def encode_output(f: h5py.File, include_top):
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
            img_path = [os.path.join(args.in_dir, dir[:-5] + 'camera_ori.png')]#,
            if include_top:
                img_path.append(os.path.join(args.in_dir, dir[:-5] + 'camera_top.png'))
            if not os.path.exists(jsn_path) or not os.path.exists(img_path[0]):# or not os.path.exists(img_path[1]):
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
        pose_arrs = []
        for object_data in json_data['objects']:
            object = '%s_%s_%s_%s' % (object_data['shape'],
                                         object_data['color'],
                                         object_data['material'],
                                         object_data['size'])
            objects.append(object)
            # Add pose
            r = Rotation.from_euler('xyz', object_data['rotation'], degrees=False)
            pose_arr = np.concatenate([np.array(object_data['3d_coords']), np.array(r.as_quat())])
            pose_arrs.append(pose_arr)
        poses = np.stack(pose_arrs)
        scene_grp.create_dataset('poses', data=poses)

        obj_str = ','.join(objects)
        scene_grp.create_dataset('obj_classes', data=np.array([0]))

        scene_grp.create_dataset('objects', data=json.dumps(json_data['objects']))

    # for i, (img_path, json_path) in enumerate(zip(img_paths, jsn_paths)):
    #     os.remove(img_path)
    #     os.remove(json_path)


if __name__ == '__main__':
    with h5py.File(args.h5_path, 'w') as f:
        while True:
            encode_output(f, args.include_top)
            f.flush()
            time.sleep(1)
            break
