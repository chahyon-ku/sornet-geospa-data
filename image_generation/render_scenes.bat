python -u render_scenes.py --properties_json data/properties/mug_easy_properties.json^
	--output_image_dir ../output/mug_1drot_train/^
	--output_scene_dir ../output/mug_1drot_train/^
	--num_images 2000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/mug_easy_properties.json^
	--output_image_dir ../output/mug_1drot_valid/^
	--output_scene_dir ../output/mug_1drot_valid/^
	--num_images 2000^
	--start_idx 0
@REM
@REM python -u render_scenes.py --properties_json data/properties/mug_train_properties.json^
@REM 	--output_image_dir ../output/mug_train2/^
@REM 	--output_scene_dir ../output/mug_train2/^
@REM 	--num_images 5000^
@REM 	--start_idx 0