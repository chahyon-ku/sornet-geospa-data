# Rendering Images

## Setup
Clone the repo using:
`git clone https://github.com/Nanami18/Summer_research.git`

**Operating system:** The code is tested on Ubuntu 20.04, it is expected to work on other Ubuntu and OSX system

**Python Version**: The main module used to render images is [bpy](https://pypi.org/project/bpy/). As of now (May 2022), the newest version is bpy 2.82 which relies on **Python version 3.7**


**Modules**: 

Install the required modules from *requirements.txt*:

```
cd image_generation
pip install -r requirements.txt
```

**Install bpy**:

Download the wheel file corresponding to your platform from this link https://github.com/TylerGubala/blenderpy/releases, then install it:

```
pip install *.whl
```
Then configure the Blender python module:
```
bpy_post_install
```

> Important to install `bpy-cuda` opposed to regular `bpy` in order to use the CUDA GPU rendering. If you only plan to use CPU rendering feel free to just download regular `bpy`. GPU rendering is generally much faster than CPU.



Confirm  bpy was installed properly and check the Blender version (in Python console):

```
Python 3.7.13
Type "help", "copyright", "credits" or "license" for more information.
>>> import bpy
>>> bpy.app.version_string
'2.82 (sub 7)'
>>>        
```

---

**Sofware Installation (optional)**: It is not required to have a version of Blender installed, since the python `bpy` module is standalone, but it can be helpful to have a version of the Blender GUI to open `.blend` files or even create new objects.
* As shown above, you can check the Blender version of bpy in a Python console to
ensure that the version of Blender you download is compatible with bpy
* As of now, (May 2022) the newest verison of bpy 2.82 uses Blender version 2.82
* Blender 3.1.2 (newest as of May 2022) is able to open any .blend files from older versions
* To modify the base_scene blend file you need to have the same version of blender
as `bpy`. [link to Blender 2.82 release](https://download.blender.org/release/Blender2.82/)
* Blender's compatibility was broken between versions 2.7 and 2.8, so as long as
Blender and bpy are both >= 2.8 everything should work 

**Blender download page is here:** https://www.blender.org/download/

---

## Generating Images

After finishing the **Setup** steps above you should be ready to generate scene images.

Running the image generation script: 
```
cd image_generation
python render_images.py
```
By default without any additional command line arguments, this script will use the GPU to create 10 images, with 3 to 6 objects per scene. For each scene, an image will be rendered from the original camera perspective with a corresponding JSON file containing the ground truth information of the scene. The scenes' images will be 480 pixels wide by 320 pixels high. 

### Image Generation Arguments
To see a full list of arguments avaiable for `render_images.py` run:
```
python render_images.py --help
```


**Handy arguments**
* num_images - an integer, number of images to generate
* use_gpu- an integer, available if you have a NVIDIA GPU
* min_object/max_object - an integer, set the min/max number of objects in the scene
* skip_ori_camera - if specified, don't use original camera provided by CLEVR
* top_camera - if specificed, use the top camera
* side_camera - an integer, set the number of side camera to use, max 4

Example usage, generating 10 scenes, with a top view and 2 side views, using the gpu:
```
python render_images.py --num_images 10 --top_view --side_camera 2 --use_gpu 1
```
(total of 10 scenes with 1 top view, 2 side views, and one original view = 
10 * 4 = 40 total images rendered)

> For more information about the underlying mechanisms of scene setup and usage
of the command line arguments for rendering check out the 
[moreinfo markdown file](./moreinfo.md).

---

## Generating Canonical Views
To generate canonical views for each possible attribute combination(shape + color + size + material) of each object, run the following command from the main directory
```
cd image_generation
python canonical_view.py --cam_views 4 --num_images 1
```

* `cam_views`  is the number of angles, so `1` would be just a front view, and the max `4` would be front, back, left, and right views. 
* `num_images` is the number of times to repeat the process, so for example a value of `3` would repeat the process 3 times for each possible shape. 

(The total number of output images will be 672 (total number of distinct possible shapes) * `cam_views` * `num_images`)

Then crop the canonical views into desired size with the following command:
<code><pre>
python crop_views.py
</code></pre>

---

## Getting Ground Truth
The ground truth will be stored in the format of .json file. Generating ground truth by running the following commands in the terminal(starting from the main directory of this repo):
<code><pre>
cd image_generation
blender --background --python render_images.py -- --num_images 10 --use_gpu 1
</code></pre>

This will generate one view for all possible combinations from each camera(4 in total), specifying --num_images
will control how many iterations gets run(the number of views would be num_images*4 for each combination)

The output will the ground truth of all of the images that was created and it will be stored in one single file called data.json. It will be saved in the output folder. 

## Common Issue
When generate a large amount of data, you might get the following error: *Two many open files*

To solve this, increase the open file limit on your system. For example, on Linux, run the following from the command line:
 <code><pre>
ulimit -n       # Check the current open file limit
ulimit -n num   # Set the open file limit to num
ulimit -Hn      # Check the maximum size you can set
</code></pre>
It's recommended to set this number to 4*num_images if you are using the default value for argument under **Settings for objects** in render_images.py.

 Set a higher limit if you are generating scenes with stricter constraints(for example, increased the min_pixels_per_object) 

 If the open file limit you want to set is above the maximum size, you will need root access to modify that upper bound

 Notice the change will only affect the current session
 
 ## Getting Predicates
Once the images has been generated, there will be an output folder that will contain all of the ground-truth. This is where you will run the predicates.py script. Once you run the predicate.py script, a new file called data.json will appear. This file will contain the predicates for each image:
<code><pre>
cd output
python3 predicates.py
</code></pre>
It will state the scene, the object that is being examined, the object's attributes, and the possible relationship it could possible have.

## TODO List
<ul>
 <li> The support relationship doesn't check for outside collision yet(i.e.: check for the collision between the child and surrounding objects)</li>
 <li> Currently only check for a few rotation, not an exhaustive search for possible poses</li>
</ul>
