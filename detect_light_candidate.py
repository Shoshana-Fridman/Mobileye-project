from math import sqrt

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from skimage.feature import peak_local_max

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

print("All imports okay. Yay!")

LIGHT_CANDIDATES_NUM = 10


def find_lights(c_image: np.ndarray, kernel):
    c_image = sg.convolve2d(c_image, kernel)
    coordinates = peak_local_max(c_image, min_distance=30, num_peaks=LIGHT_CANDIDATES_NUM)
    return coordinates[:, 1] + 5, coordinates[:, 0] - 5


def get_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def run(image_path: str):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    red_kernel = np.array([[-0.5, -0.5, -0.5],
                           [-0.5, -0.5, -0.5],
                           [-0.5, -0.5, -0.5],
                           [1, 1, -0.5],
                           [1, 2, 1],
                           [1, 1, 1]])

    green_kernel = np.array([[1, 1, 1],
                             [1, 2, 1],
                             [1, 1, -0.5],
                             [-0.5, -0.5, -0.5],
                             [-0.5, -0.5, -0.5],
                             [-0.5, -0.5, -0.5]])

    c_image = np.array(Image.open(image_path))
    x_red, y_red = find_lights(c_image[:, :, 0], red_kernel)
    x_green, y_green = find_lights(c_image[:, :, 1], green_kernel)
    red_candidates = list(zip(x_red, y_red))
    green_candidates = list(zip(x_green, y_green))
    for red_candidate in red_candidates:
        green_candidates = list(filter(lambda x: get_distance(red_candidate, x) > 15, green_candidates))

    return red_candidates, green_candidates




'''tests for detect light candidates'''

# def show_image_and_gt(image, objs, fig_num=None):
#     plt.figure(fig_num).clf()
#     plt.imshow(image)
#     labels = set()
#     if objs is not None:
#         for o in objs:
#             poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
#             plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
#             labels.add(o['label'])
#         if len(labels) > 1:
#             plt.legend()
#
#
# def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
#     """
#     Run the attention code
#     """
#     if json_path is None:
#         objects = None
#     else:
#         gt_data = json.load(open(json_path))
#         what = ['traffic light']
#         objects = [o for o in gt_data['objects'] if o['label'] in what]
#     image_array = np.array(Image.open(image_path))
#     show_image_and_gt(image_array, objects, fig_num)
#
#     red_candidates, green_candidates = run(image_path)
#     plt.plot(np.array(red_candidates)[:,0], np.array(red_candidates)[:,1], 'ro', color='r', markersize=4)
#     plt.plot(np.array(green_candidates)[:,0], np.array(green_candidates)[:,1], 'ro', color='g', markersize=4)
#
#
# def main(argv=None):
#     """It's nice to have a standalone tester for the algorithm.
#     Consider looping over some images from here, so you can manually exmine the results
#     Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
#     :param argv: In case you want to programmatically run this"""
#     parser = argparse.ArgumentParser("Test TFL attention mechanism")
#     parser.add_argument('-i', '--image', type=str, help='Path to an image')
#     parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
#     parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
#     args = parser.parse_args(argv)
#     default_base = 'images'
#     if args.dir is None:
#         args.dir = default_base
#     flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
#     for image in flist:
#         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
#         if not os.path.exists(json_fn):
#             json_fn = None
#         test_find_tfl_lights(image, json_fn)
#     if len(flist):
#         print("You should now see some images, with the ground truth marked on them. Close all to quit.")
#     else:
#         print("Bad configuration?? Didn't find any picture to show")
#     plt.show(block=True)
#
#
# if __name__ == '__main__':
#     main()

