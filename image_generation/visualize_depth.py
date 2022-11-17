import cv2
import numpy


if __name__ == '__main__':
    depth_img = numpy.array(cv2.imread('../output/bunny_easy/000000_ori_depth.png', cv2.IMREAD_UNCHANGED))
    # depth_img = depth_img[:, :, ::-1]
    print(depth_img)
    depth_img = depth_img.astype('float') / 256.0
    norm_depth_img = (2.0 ** (depth_img[:, :, 0] * 256 - 128)) * (depth_img[:, :, 1] + depth_img[:, :, 2] / 256.0 + depth_img[:, :, 3] / 256.0 ** 2)
    print(norm_depth_img)
    # norm_depth_img = (norm_depth_img - numpy.min(norm_depth_img)) / (numpy.max(norm_depth_img) - numpy.min(norm_depth_img))
    # norm_depth_img = numpy.clip(norm_depth_img, 0, 1)
    # print(depth_img, norm_depth_img)
    cv2.imshow('image', norm_depth_img / numpy.max(norm_depth_img))
    cv2.waitKey()