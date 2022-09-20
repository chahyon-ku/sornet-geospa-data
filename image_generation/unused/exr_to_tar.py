import tarfile
import io
import os
import json
import time


OUT_DIR_PATH = '../output/geospa_exr_split2'
OUT_H5_PATH = '../output/geospa_exr_split2_train.tar'

def encode_output(f: tarfile.TarFile):
    '''
    Looks through the output image and json files, encoding them into a
    single h5 file. Takes num_scenes: the expected number of scenes which were
    generated.
    '''
    # Verify output files
    img_paths = []
    jsn_paths = []
    for dir in os.listdir(OUT_DIR_PATH):
        if dir[-3:] == 'exr':#'png':
            img_path = os.path.join(OUT_DIR_PATH, dir)
            jsn_path = os.path.join(OUT_DIR_PATH, dir[:-14] + '.json')
            time_since_modify = time.time() - os.path.getmtime(img_path)
            if time_since_modify > 10:
                img_paths.append(img_path)
                jsn_paths.append(jsn_path)


    # Add info for each scene
    img_paths.sort()
    jsn_paths.sort()
    for i, (img_path, jsn_path) in enumerate(zip(img_paths, jsn_paths)):
        f.add(img_path, arcname=img_path.split('/')[-1])
        f.add(jsn_path, arcname=jsn_path.split('/')[-1])
        print(img_path, jsn_path)

    for i, (img_path, json_path) in enumerate(zip(img_paths, jsn_paths)):
        os.remove(img_path)
        os.remove(json_path)


if __name__ == '__main__':
    while True:
        with tarfile.open(OUT_H5_PATH, 'a') as f:
            encode_output(f)
        time.sleep(10)
