@REM python scenes_to_h5.py --in-dir=../output/mug_valid/ --h5-path=../output/mug_valid_5000.h5
@REM python scenes_to_h5.py --in-dir=../output/mug_train/ --h5-path=../output/mug_train_5000.h5
@REM python scenes_to_h5.py --in-dir=../output/mug_train2/ --h5-path=../output/mug_train2_5000.h5

python scenes_to_h5.py --in-dir=../output/mug_valid_easy/ --h5-path=../output/mug_easy_valid.h5
python scenes_to_h5.py --in-dir=../output/mug_train_easy/ --h5-path=../output/mug_easy_train.h5