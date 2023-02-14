import glob
import os

from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase


class RobotData(GraspDatasetBase):
    """
    Dataset wrapper for the Jacquard dataset.
    """

    def __init__(self, file_path, ds_rotate=False, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(RobotData, self).__init__(**kwargs)
        print(os.getcwd())
        
        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        # self.depth_files = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in self.grasp_files]
        self.depth_files = sorted(glob.glob('outputs/depth/*.png'))
        self.rgb_files = sorted(glob.glob('outputs/rgb/*.png'))
        self.grasp_files = self.depth_files.copy()
        self.length = len(self.depth_files)

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.img = depth_img.img[500:, :]
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.img = rgb_img.img[500:, :]
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])
