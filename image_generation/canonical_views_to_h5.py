import argparse
import numpy
import io
import os
from PIL import Image
import h5py


def crop(img_pil: Image):
    top = 70
    left = 120
    bottom = 166
    right = 216
    img_pil = img_pil.crop((left, top, right, bottom))
    return img_pil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, default='../output/objects/')
    parser.add_argument('--h5-path', type=str, default='../output/objects.h5')
    args = parser.parse_args()

    with h5py.File(args.h5_path, 'a') as f:
        for object_dir in os.listdir(args.in_dir):
            if os.path.isfile(os.path.join(args.is_dir, object_dir)):
                continue
            print(object_dir)
            object_name = object_dir
            tokens = object_name.split('_01_')
            if len(tokens) > 1:
                object_name = tokens[1].split('_one-size')[0] + '_' + tokens[0] + '_01'
                if object_name.startswith('large_'):
                    object_name = object_name[6:] + '_large'
                elif object_name.startswith('small_'):
                    object_name = object_name[6:] + '_small'
            else:
                tokens = object_name.split('_')
                object_name = '_'.join(tokens[2:-1]) + '_' + '_'.join(tokens[0:2])
            print(object_name)
            img_bytes_array = []
            for i, file in enumerate(os.listdir(os.path.join(args.is_dir, object_dir))):
                if file[-3:] != 'png':
                    continue
                img_pil = Image.open(os.path.join(args.is_dir, object_dir, file))
                img_pil = crop(img_pil)
                img_pil = img_pil.resize((32, 32))
                buf = io.BytesIO()
                img_pil.save(buf, 'png')
                img_bytes = buf.getvalue()
                img_bytes = numpy.array(img_bytes)
                img_bytes_array.append(img_bytes)
            f.create_dataset(object_name, shape=(len(img_bytes_array), ), data=img_bytes_array)