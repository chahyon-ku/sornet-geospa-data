python -u render_scenes.py --properties_json data/properties/mug_train_properties.json^
	--output_image_dir ../output/mug_train/^
	--output_scene_dir ../output/mug_train/^
	--num_images 2000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/mug_val_properties.json^
	--output_image_dir ../output/mug_val/^
	--output_scene_dir ../output/mug_val/^
	--num_images 2000^
	--start_idx 0