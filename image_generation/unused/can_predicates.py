# load and show an image with Pillow
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import matplotlib.patches as mpatches
from numpy import asarray
import numpy
import cv2 as cv
import math
from scipy import signal

# Open the image form working directory
parent_image = Image.open('../../top_view_images/cube_container_01_large.png')
child_image = Image.open('../../top_view_images/cube_01_large.png')

# summarize some details about the shapes
print(parent_image.size)
print(child_image.size)

# # show the parent image
# pyplot.imshow(parent_image)
# pyplot.show()
#
# # show the child image
# pyplot.imshow(child_image)
# pyplot.show()


# Is parent container ?
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


is_container(parent_image)
is_container(child_image)
