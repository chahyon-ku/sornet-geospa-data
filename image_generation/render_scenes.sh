#!/bin/bash
python -u render_scenes.py --properties_json 'data/properties/train_properties.json'\
	--output_image_dir '../output/train/'\
	--output_scene_dir '../output/train/'\
	--num_images 80000\
	--start_idx 0

python -u render_scenes.py --properties_json 'data/properties/train_properties.json'\
	--output_image_dir '../output/val_default/'\
  --output_scene_dir '../output/val_default/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/val_op_properties.json'\
  --output_image_dir '../output/val_op/'\
  --output_scene_dir '../output/val_op/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/val_color_properties.json'\
	--output_image_dir '../output/val_color/'\
  --output_scene_dir '../output/val_color/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/kitchen_properties.json'\
  --output_image_dir '../output/val_kitchen/'\
  --output_scene_dir '../output/val_kitchen/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/val_size_properties.json'\
	--output_image_dir '../output/val_size/'\
  --output_scene_dir '../output/val_size/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/val_shape_properties.json'\
	--output_image_dir '../output/val_shape/'\
  --output_scene_dir '../output/val_shape/'\
  --num_images 10000\
  --start_idx 0

python -u render_scenes.py --properties_json 'data/properties/val_material_properties.json'\
  --output_image_dir '../output/val_material/'\
  --output_scene_dir '../output/val_material/'\
  --num_images 10000\
  --start_idx 0
