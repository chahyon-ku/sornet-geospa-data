# Util functions for determining relationships between objects
# load and show an image with Pillow

from PIL import Image
from numpy import asarray
import numpy
import cv2 as cv

def is_container(object_image):
    object_np = asarray(object_image)
    print(object_np.shape)

    object_np = numpy.where(object_np < 100, 0, object_np)
    object_np = numpy.where((object_np >= 100) & (object_np < 230), 120, object_np)
    object_np = numpy.where(object_np >= 230, 255, object_np)

    u_values = numpy.unique(object_np[:, :, 1])
    print(u_values)
    if 120 in u_values:
        print("Object is container")
        return True
    else:
        print("Object is not a container")
        return False


def image2nptemplate(object_image, parent):
    object_np = asarray(object_image)

    if parent:
        object_np = numpy.where(object_np < 100, 0, object_np)
        object_np = numpy.where((object_np >= 100) & (object_np < 230), 120, object_np)
        object_np = numpy.where(object_np >= 230, 255, object_np)
    else:
        object_np = numpy.where(object_np <= 250, 0, object_np)
        object_np = numpy.where(object_np > 250, 255, object_np)

    return object_np

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def inside_sup(dx, dy, theta, shape_name, scale=117.2):
    directory = "../top_view_images/"
    parent_image = cv.imread(directory + shape_name + ".png")
    rotated_np = image2nptemplate(rotate_image(parent_image, theta), True)
    centerx, centery = rotated_np.shape[0]//2, rotated_np.shape[1]//2
    x,y = int(centerx + dx * scale), int(centery + dy * scale)

    if x >= rotated_np.shape[0] or y >= rotated_np.shape[1]:
        return False
    if rotated_np[x][y][0]:
        return True
    else:
        return False

    num_to_shape = {
        0: "cube_01",
        1: "cube_01_small",
        2: "cube_01_large",
        3: "cylinder_01_small",
        4: "cylinder_01",
        5: "cylinder_01_large",
        6: "cube_container_01_small",
        7: "cube_container_01",
        8: "cube_container_01_large",
        9: "cube_cylinder_01_small",
        10: "cube_cylinder_01",
        11: "cube_cylinder_01_large",
        12: "long_cylinder_01_small",
        13: "long_cylinder_01",
        14: "long_cylinder_01_large",
        15: "rectangle_01_small",
        16: "rectangle_01",
        17: "rectangle_01_large",
        18: "rectangle_cont_01_small",
        19: "rectangle_cont_01",
        20: "rectangle_cont_01_large",
        21: "rectangle_cylinder_01_small",
        22: "rectangle_cylinder_01",
        23: "rectangle_cylinder_01_large",
        24: "short_cylinder_01_small",
        25: "short_cylinder_01",
        26: "short_cylinder_01_large",
        27: "cylinder_container_01_small",
        28: "cylinder_container_01",
        29: "cylinder_container_01_large",
        30: "cylinder_rect_cont_01_small",
        31: "cylinder_rect_cont_01",
        32: "cylinder_rect_cont_01_large",
        33: "ellipse_container_01_small",
        34: "ellipse_container_01",
        35: "ellipse_container_01_large",
        36: "ellipse_rect_cont_01_small",
        37: "ellipse_rect_cont_01",
        38: "ellipse_rect_cont_01_large",
        39: "thin_rectangle_01_small",
        40: "thin_rectangle_01",
        41: "thin_rectangle_01_large"
    }