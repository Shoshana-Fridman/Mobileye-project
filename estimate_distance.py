import numpy as np
import matplotlib.pyplot as plt

import SFM


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]


def run(curr_img_path, prev_img_path, curr_tfls, prev_tfls, EM, focal, pp):
    prev_container = FrameContainer(prev_img_path)
    curr_container = FrameContainer(curr_img_path)
    prev_container.traffic_light = np.array(prev_tfls)
    curr_container.traffic_light = np.array(curr_tfls)
    curr_container.EM = EM
    curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
    return curr_container


