@REM python scenes_to_h5.py --in-dir=../output/mug_valid/ --h5-path=../output/mug_valid_5000.h5
@REM python scenes_to_h5.py --in-dir=../output/mug_train/ --h5-path=../output/mug_train_5000.h5
@REM python scenes_to_h5.py --in-dir=../output/mug_train2/ --h5-path=../output/mug_train2_5000.h5

@REM python scenes_to_h5.py --in-dir=../output/mug_big_train/ --h5-path=../output/mug_big_train.h5 --include_top True
@REM python scenes_to_h5.py --in-dir=../output/mug_big_valid/ --h5-path=../output/mug_big_valid.h5 --include_top True
@REM python scenes_to_h5.py --in-dir=../output/bunny_train/ --h5-path=../output/bunny_train.h5
@REM python scenes_to_h5.py --in-dir=../output/bunny_valid/ --h5-path=../output/bunny_valid.h5
python scenes_to_h5.py --in-dir=../output/dragon_train/ --h5-path=../output/dragon_train.h5
python scenes_to_h5.py --in-dir=../output/dragon_valid/ --h5-path=../output/dragon_valid.h5