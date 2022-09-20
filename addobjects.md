# Adding New Objects


## Creating a new blender object

Blender objects are stored in the folder `image_generation/data/shapes` where each object is a single `.blend` file containting a single object.

![example blend file contents](refrence_images/example_blend_file_contents.jpg)

Above is an example image for a blender object named *kitchen_glass*. The name of the object in the blendfile and the name for the file should match. In this example the blendfile should have the path:

`image_generation/data/shapes/kitchen_glass.blend`

**It's important that both the `.blend` file and the object within the file have the same name.**

## Object Properties

![example blender file](refrence_images\blender_kitchen_glass.jpg)

Above is an example of the *kitchen_glass* blender file. There's a couple of things which are important for every file:

* Objects should be oriented such that they're upright and the longest side is along the x-axis in blender (the red axis)
* Circled in BLUE is the properties panel of the 3D view. This can be opened by pressing `n` on the keyboard. The typical dimensions of objects we use is around 1 meter, but this could vary depending on your set of objects. 
* Circled in RED is the object transform properties. All of the *location* and *rotation* values should be zero, and the *scale* values should be 1, but if the values are wrong fix them in the next step:
* If the object is oriented correctly but the transform values are incorrect, then you need to **apply transforms** for the object. With the object selected, `ctrl-A` will open the *Apply* menu, then select *All Transforms*: ![apply transforms](refrence_images\apply_transforms.jpg) 

---

## Updating the properties.json file

The **properties** file is a JSON file located in `image_generation/data` it contains information about how to generate each scene. The main sections are:

Shapes
: the names of shapes to include in the scenes, where each shape name corresponds to the shape file name and name within the blend file. For example to add the kitchen glass object, you would add `"kitchen_glass": idx` to the `shapes` dictionary, where `idx` is an arbitrary integer identifier for the object distinct from all other objects in the JSON file in the range [0, number of objects). The object id is important for the relationship matrices discussed lower.

Colors
: a dictionary of colors, where the keys are the name of the color (ex: red, blue, green) and the values are the RGB values for the color (ex: something like `[20, 105, 20]` for green)

Materials
: dictonary of material names to their file name (excluding .blend extension) in the folder `image_generation/data/materials`

Sizes
: dictionary of size names to their float scale value. For our purposes we only use one size value where every object is scaled by 0.5

Matrices
: the last four elements are the relative file paths of numpy matrices storing the possible relationships between objects. These files must be independently generated for each set of objects before scene images can be generated. The next section talks more about these matrices.

---

## Relationship Matrices

The `contain_mat` and `support_mat` must have shapes of N x N

The `can_contain_mat` and `can_support_mat` must have shapes of N x N x N

Where **N** is the number of objects in the `shapes` dictionary from the properties.json file. These matrices are important when generating scenes so the code knows which objects can be contained in others or supported above others.

**Example** 

JSON file:
```
{
    "shapes": {
        "kitchen_plate": 0,
        "kitcen_glass": 1,
        "kitchen_mug": 2,
        "kitchen_fork": 3
    },
    ...
}
```

Matrix values:

* `support_mat[3, 0]` would be `0` because the fork can't support a plate
* `contain_mat[1, 3]` would be `1` because the glass CAN contain a fork
* `can_support_mat[0, 1, 2]` would be `1` because the plate could support both a glass and a mug

## Generating Relationship Matrices

**Any time a new object is added, all four of the relationship matrices must be regenerated**

### Relationship logic

**Supported**
an object can be supported by another if the child supportee has a smaller x and y dimensions than the parent supporter.

**Contained** 
an object can be contained if the child containee can be placed in the parent container such that the center of the child fits inside the bounding box of the parent, i.e. more than half the child is inside the parent.

### Generating

To generate relationship matrices run:

```
python pregenerate_predicates.py <set_name> --properties_json <file_path>
```

**set_name**
: the name of the set of objects, the saved matrices will all start with this string

**file_path** 
: path to the properties JSON file with the `shapes` dictionary of shape names to their identifying integer indices

**EXAMPLE**

Generating for set of kitchen objects, run:
```
python pregenerate_predicates.py kitchen --properties_json data/kitchen_properties.json
```

And these four files will be generated:

```
data/kitchen_rels_contain_dou.dat
data/kitchen_rels_support_dou.dat
data/kitchen_rels_contain_tri.npy
data/kitchen_rels_support_tri.npy
```













