python -u render_scenes.py --properties_json data/properties/mug_easy_properties.json^
	--output_image_dir ../output/mug_big_train/^
	--output_scene_dir ../output/mug_big_train/^
	--num_images 1000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/mug_easy_properties.json^
	--output_image_dir ../output/mug_big_valid/^
	--output_scene_dir ../output/mug_big_valid/^
	--num_images 1000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/bunny_easy_properties.json^
	--output_image_dir ../output/bunny_train/^
	--output_scene_dir ../output/bunny_train/^
	--num_images 1000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/bunny_easy_properties.json^
	--output_image_dir ../output/bunny_valid/^
	--output_scene_dir ../output/bunny_valid/^
	--num_images 1000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/dragon_easy_properties.json^
	--output_image_dir ../output/dragon_train/^
	--output_scene_dir ../output/dragon_train/^
	--num_images 1000^
	--start_idx 0

python -u render_scenes.py --properties_json data/properties/dragon_easy_properties.json^
	--output_image_dir ../output/dragon_valid/^
	--output_scene_dir ../output/dragon_valid/^
	--num_images 1000^
	--start_idx 0
@REM
@REM python -u render_scenes.py --properties_json data/properties/mug_train_properties.json^
@REM 	--output_image_dir ../output/mug_train2/^
@REM 	--output_scene_dir ../output/mug_train2/^
@REM 	--num_images 5000^
@REM 	--start_idx 0