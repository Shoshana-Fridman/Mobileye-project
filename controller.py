import ast
import glob
import os
import sys
import pickle
import numpy as np

from tfl_manager import TFL_manager
from variables import FRAME_EXTENSION_LENGTH


def init_pls():
    pkl_file_path = glob.glob(os.path.join("dusseldorf", '*.pkl'))
    path_images = glob.glob(os.path.join("dusseldorf", '*_leftImg8bit.png'))
    with open(r"pls.txt", "w") as pls_file:
        pls_file.write(str(pkl_file_path + path_images))


def get_frame_id(frame_path):
    return int(frame_path[-FRAME_EXTENSION_LENGTH - 6:-FRAME_EXTENSION_LENGTH])


class Controller:
    def __init__(self, pls_file):
        self.pls = pls_file
        self.pkl, self.frame_list = self.get_pathes()
        self.data = self.load_data()
        self.focal = self.data['flx']
        self.pp = self.data['principle_point']
        self.tfl_manager = TFL_manager(self.focal, self.pp)

    def get_pathes(self):
        pkl = ""
        frame_list = []
        with open(self.pls, "r") as pls_file:
            pathes_list = ast.literal_eval(pls_file.read())
            for path in pathes_list:
                if path[-3:] == "pkl":
                    pkl = path
                else:
                    frame_list.append(path)
        return pkl, frame_list

    def load_data(self):
        with open(self.pkl, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        return data

    def run(self):
        if self.frame_list:
            '''handel first image'''
            self.tfl_manager.run(self.frame_list[0])
            '''handel other images'''
            for image_path in self.frame_list[1:]:
                EM = np.eye(4)
                for index in range(get_frame_id(image_path) - 1, get_frame_id(image_path)):
                    EM = np.dot(self.data['egomotion_' + str(index) + '-' + str(index + 1)], EM)
                self.tfl_manager.run(image_path, EM)


if __name__ == '__main__':
    control = Controller(sys.argv[1])
    control.run()
