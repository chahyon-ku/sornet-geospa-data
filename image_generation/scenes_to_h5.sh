#!/bin/bash

python exr_to_scenes.py --in-dir=../output/train/ --h5-path=../output/train.h5
python exr_to_scenes.py --in-dir=../output/val_default/ --h5-path=../output/val_default.h5
python exr_to_scenes.py --in-dir=../output/val_color/ --h5-path=../output/val_color.h5
python exr_to_scenes.py --in-dir=../output/val_size/ --h5-path=../output/val_size.h5
python exr_to_scenes.py --in-dir=../output/val_shape/ --h5-path=../output/val_shape.h5
python exr_to_scenes.py --in-dir=../output/val_material/ --h5-path=../output/val_material.h5
python exr_to_scenes.py --in-dir=../output/val_op/ --h5-path=../output/val_op.h5
