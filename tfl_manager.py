import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

import detect_light_candidate
import predict_tfl
import estimate_distance


class TFL_manager:
    def __init__(self, focal, pp):
        self.model = self.get_model()
        self.focal = focal
        self.pp = pp
        self.prev_tfl_points = []
        self.prev_image_path = ""

    def get_model(self):
        loaded_model = load_model("model.h5")
        return loaded_model

    def visualize(self, image, red_candidates, green_candidates, red_tfls, green_tfls, curr_container):
        fig, (distance_sec, tfl_sec, suspicious_sec) = plt.subplots(1, 3, figsize=(12, 6))
        fig.canvas.set_window_title('Mobileye Project 2020')
        plt.suptitle(f"Frame {image}")

        '''part 1'''
        suspicious_sec.set_title('Suspicious candidates')
        suspicious_sec.imshow(plt.imread(image))
        suspicious_sec.plot(red_candidates[:, 0], red_candidates[:, 1], 'ro', color='r', markersize=4)
        suspicious_sec.plot(green_candidates[:, 0], green_candidates[:, 1], 'ro', color='g', markersize=4)

        '''part 2'''
        tfl_sec.set_title('Traffic light candidates')
        tfl_sec.imshow(plt.imread(image))
        tfl_sec.plot(red_tfls[:, 0], red_tfls[:, 1], 'ro', color='r', markersize=4)
        tfl_sec.plot(green_tfls[:, 0], green_tfls[:, 1], 'ro', color='g', markersize=4)

        '''part 3'''
        distance_sec.set_title('tfl distances')
        if curr_container:
            distance_sec.imshow(curr_container.img)
            curr_p = curr_container.traffic_light
            for i in range(len(curr_p)):
                if curr_container.valid[i]:
                    distance_sec.text(curr_p[i, 0], curr_p[i, 1],
                                      r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')

        plt.show()

    def run(self, image, EM=None):
        '''part 1'''
        red_candidates, green_candidates = detect_light_candidate.run(image)

        '''part 2'''
        red_tfls, green_tfls = predict_tfl.run(image, red_candidates, green_candidates, self.model)

        '''sanity: check if part 1 return no more candidates then part 2'''
        try:
            assert len(red_candidates) >= len(red_tfls) or len(green_candidates) >= len(green_tfls)
        except AssertionError:
            red_tfls, green_tfls = red_candidates, green_candidates

        '''part 3'''
        curr_container = None
        if self.prev_image_path:
            curr_container = estimate_distance.run(image, self.prev_image_path, red_tfls + green_tfls,
                                                   self.prev_tfl_points, EM, self.focal, self.pp)

        self.visualize(image, np.array(red_candidates), np.array(green_candidates), np.array(red_tfls),
                       np.array(green_tfls), curr_container)
        self.prev_tfl_points = red_tfls + green_tfls
        self.prev_image_path = image
