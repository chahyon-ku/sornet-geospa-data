**Object Placement**

Each object is positioned randomly, but before actually adding the object to the scene we ensure that its center is at least --min_dist units away from the centers of all other objects. We also ensure that between each pair of objects, the left/right and front/back distance along the ground plane is at least --margin units; this helps to minimize ambiguous spatial relationships. If after --max_retries attempts we are unable to find a suitable position for an object, then all objects are deleted and placed again from scratch.

**Image Resolution**

By default images are rendered at 320x240, but the resolution can be customized using the --height and --width flags.

**Rendering Quality**

You can control the quality of rendering with the --render_num_samples flag; using fewer samples will run more quickly but will result in grainy images. I've found that 64 samples is a good number to use for development; The --render_min_bounces and --render_max_bounces control the number of bounces for transparent objects; I've found the default of 8 to work well for these options.

When rendering, Blender breaks up the output image into tiles and renders tiles sequentialy; the --render_tile_size flag controls the size of these tiles. This should not affect the output image, but may affect the speed at which it is rendered. For CPU rendering smaller tile sizes may be optimal, while for GPU rendering larger tiles may be faster.

With default settings, rendering a 320x240 image takes about 4 seconds on a Pascal Titan X. It's very likely that these rendering times could be drastically reduced by someone more familiar with Blender, but this rendering speed was acceptable for our purposes.

**Saving Blender Scene Files**

You can save a Blender .blend file for each rendered image by adding the flag --save_blendfiles 1. These files can be more than 5 MB each, so they are not saved by default.

**Output Files**

Rendered images are stored in the --output_image_dir directory, which is created if it does not exist. The filename of each rendered image is constructed from the --filename_prefix, the --split, and the image index.

A JSON file for each scene containing ground-truth object positions and attributes is saved in the --output_scene_dir directory, which is created if it does not exist. After all images are rendered the JSON files for each individual scene are combined into a single JSON file and written to --output_scene_file. This single file will also store the --split, --version (default 1.0), --license (default CC-BY 4.0), and --date (default today).

When rendering large numbers of images, I have sometimes experienced random Blender crashes; saving JSON files for each scene as they are rendered ensures that you do not lose information for scenes already rendered in the event of a crash.

If saving Blender scene files for each image (--save_blendfiles 1) then they are stored in the --output_blend_dir directory, which is created if it does not exist.

**Object Properties**

The file --properties_json file (default data/properties.json) defines the allowed shapes, sizes, colors, and materials used for objects.
Each shape (cube, sphere, cylinder) is stored in its own .blend file in the --shape_dir (default data/shapes); the file X.blend contains a single object named X centered at the origin with unit size. The shapes field of the JSON properties file maps human-readable shape names to .blend files in the --shape_dir.

The colors field of the JSON properties file maps human-readable color names to RGB values between 0 and 255; most of our colors are adapted from Wad's Optimum 16 Color Palette.

The sizes field of the JSON properties file maps human-readable size names to scaling factors used to scale the object models from the --shape_dir.

Each material is stored in its own .blend file in the --material_dir (default data/materials). The file X.blend should contain a single NodeTree item named X, and this NodeTree item must have a single Color input that accepts an RGBA value so that each material can be used with any color. The materials field of the JSON properties file maps human-readable material names to .blend files in the --material_dir.
