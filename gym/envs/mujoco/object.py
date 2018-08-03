import numpy as np
import random
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
from PIL import Image

class ObjectEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file=None, distractors=False):
        utils.EzPickle.__init__(self)
        print(xml_file)
        if xml_file is None:
            xml_file = 'object.xml'
        self.include_distractors = distractors
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def _step(self, a):

        ob = self._get_obs()
        done = False
        return ob, 0, done, None

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[1] = 0.5
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = random.uniform(-180,180)
        self.viewer.cam.elevation = random.uniform(-30,-30)  # -60 lower range  # used to be (-60,-30)

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.obj_pos = np.concatenate([
                    np.random.uniform(low=-0.4, high=0.4, size=1),
                    np.random.uniform(low=-0.4, high=0.4, size=1)])
            if np.linalg.norm(self.obj_pos - self.goal_pos) > 0.17:
                break

        if self.include_distractors:
            if self.obj_pos[1] < 0:
                y_range = [0.0, 0.2]
            else:
                y_range = [-0.2, 0.0]
            while True:
                self.distractor_pos = np.concatenate([
                        np.random.uniform(low=-0.3, high=0, size=1),
                        np.random.uniform(low=y_range[0], high=y_range[1], size=1)])
                if np.linalg.norm(self.distractor_pos - self.goal_pos) > 0.17 and np.linalg.norm(self.obj_pos - self.distractor_pos) > 0.1:
                    break
            qpos[-6:-4] = self.distractor_pos

        self.obj_angle = random.uniform(0, 2*np.pi)
        qpos[-5:-3] = self.obj_pos
        qpos[-3] = self.obj_angle
        qpos[-2:] = self.goal_pos


        setattr(self.model.data, 'qpos', qpos)
        self.model._compute_subtree()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self._get_obs()

    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, self.obj_pos, self.obj_angle


    def _get_obs(self):
        if self.include_distractors:
            return np.concatenate([
                self.get_body_com("distractor"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ])
        else:
            return np.concatenate([
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ])

