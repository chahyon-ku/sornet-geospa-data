import argparse
import numpy
import io
import os
from PIL import Image
import h5py


def crop(img_pil: Image):
    top = 70
    left = 110
    bottom = 170
    right = 210
    img_pil = img_pil.crop((left, top, right, bottom))
    return img_pil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', type=str, default='../output/mug_views/')
    parser.add_argument('--out-dir', type=str, default='../output/mug_cropped_views/')
    parser.add_argument('--h5-path', type=str, default='../output/mug_views.h5')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.h5_path, 'w') as f:
        for object_dir in os.listdir(args.in_dir):
            if os.path.isfile(os.path.join(args.in_dir, object_dir)):
                continue
            print(object_dir)
            img_bytes_array = []
            os.makedirs(os.path.join(args.out_dir, object_dir), exist_ok=True)
            for i, file in enumerate(os.listdir(os.path.join(args.in_dir, object_dir))):
                if file[-3:] != 'png':
                    continue
                img_pil = Image.open(os.path.join(args.in_dir, object_dir, file))
                img_pil = crop(img_pil)
                img_pil = img_pil.resize((32, 32))
                buf = io.BytesIO()
                img_pil.save(os.path.join(args.out_dir, object_dir, file))
                img_pil.save(buf, 'png')
                img_bytes = buf.getvalue()
                img_bytes = numpy.array(img_bytes)
                img_bytes_array.append(img_bytes)
            f.create_dataset(object_dir, shape=(len(img_bytes_array), ), data=img_bytes_array)