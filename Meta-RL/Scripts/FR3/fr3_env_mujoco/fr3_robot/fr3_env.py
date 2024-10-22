"""
Modified from gymnasium-robotics FrankaRobotEnv to use the Franka FR3 robot instead of the older Panda robot.
"""

# TODO: add argument in constructor for cartesian vs joint control; 
# and in any case, add xyz position and orientation of EE using forward kinematics (possibly also with noise)

from os import path
import numpy as np

import mujoco

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces



DEFAULT_CAMERA_CONFIG = {
    "distance": 2,
    "azimuth": 90.0, 
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.4]), #np.array([-0.2, 0.5, 2.0]),
}

class FrankaFR3Robot(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        model_path="fr3_w_hand.xml",
        frame_skip=50,   # default 50 steps = 100ms = 10Hz  (ok for gripper manipulation; perhaps aim at 20-30Hz for Tilburg Hand)
        robot_noise_ratio: float = 0.01,   # TODO: later on we will need to calibrate this
        time_limit=10.0, # 10 seconds
        control_mode='joint', # 'joint' or 'cartesian'
        **kwargs,   # render_mode, width, heigth, camera_id, camera_name
    ):
        self.xml_file_path = path.join(
            path.dirname(__file__),
            model_path,
        )

        # TODO: add argument for tilburg-hand vs franka-gripper;  in case, the default model path is different, tasks need a different file as well, and joint/actuator names will be different
        self.control_mode = control_mode

        self.frame_skip = frame_skip
        self.robot_noise_ratio = robot_noise_ratio
        self.time_limit_steps = int(time_limit * 1000 / frame_skip / 2 )
        self.current_step = 0

        ## TODO: check joint names and actuator names automatically
        self.joint_names = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7', 'finger_joint1']
        self.actuator_names = ['actuator_joint1', 'actuator_joint2', 'actuator_joint3', 'actuator_joint4', 'actuator_joint5', 'actuator_joint6', 'actuator_joint7', 'actuator_gripper']

        # HACKY: temporarily pre-load the model to get joint limits and observation
        self.model = mujoco.MjModel.from_xml_path(self.xml_file_path)
        self.data = mujoco.MjData(self.model)

        self.robot_pos_bound = np.zeros([len(self.joint_names), 2], dtype=float)
        #self.robot_vel_bound = np.ones([len(self.joint_names), 2], dtype=float)
        self.robot_pos_noise_amp = 1 * np.ones(len(self.joint_names), dtype=float) ### TODO: CHECK
        self.robot_vel_noise_amp = 1 * np.ones(len(self.joint_names), dtype=float) ### TODO: CHECK

        #self.robot_vel_bound[:, 0] = -10.0
        #self.robot_vel_bound[:, 1] = 10.0

        for i in range(len(self.actuator_names)):
            self.robot_pos_bound[i, 0] = self.model.actuator(name=self.actuator_names[i]).ctrlrange[0]
            self.robot_pos_bound[i, 1] = self.model.actuator(name=self.actuator_names[i]).ctrlrange[1]

        self.act_mid = np.mean(self.robot_pos_bound, axis=1)
        self.act_rng = 0.5 * (self.robot_pos_bound[:, 1] - self.robot_pos_bound[:, 0])

        super().__init__(
            self.xml_file_path,
            self.frame_skip,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=self._get_obs().shape, dtype=np.float32),
            **kwargs,
        )
        self.action_space = spaces.Box(low=-1., high=1., shape=self.action_space.shape, dtype=np.float32)


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Denormalize the input action from [-1, 1] range to the each actuators control range
        action = self.act_mid + action * self.act_rng

        self.do_simulation(action, self.frame_skip)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()

        terminated = False
        truncated = False
    
        self.current_step += 1
        if self.current_step >= self.time_limit_steps:
            truncated = True

        return obs, 0.0, terminated, truncated, {}


    def _get_obs(self):
        if self.control_mode == 'joint':
            # Gather simulated observation
            if self.data.qpos is not None and self.joint_names:
                robot_qpos = np.squeeze(np.array([self.data.joint(name).qpos for name in self.joint_names]))
                robot_qvel = np.squeeze(np.array([self.data.joint(name).qvel for name in self.joint_names]))
            else:
                robot_qpos = np.zeros(8) # find info at https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#data
                robot_qvel = np.zeros(8)

            # Simulate observation noise
            robot_qpos += (
                self.robot_noise_ratio
                * self.robot_pos_noise_amp
                * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qpos.shape)
            )
            robot_qvel += (
                self.robot_noise_ratio
                * self.robot_vel_noise_amp
                * self.np_random.uniform(low=-1.0, high=1.0, size=robot_qvel.shape)
            )
            return np.concatenate((robot_qpos.copy(), robot_qvel.copy()))
        
        elif self.control_mode == 'cartesian':
            ee_pos = self.data.body("right_finger").xpos.copy() # Gets the EE position: 3, 
            ee_quat = self.data.body("right_finger").xquat.copy()  # Gets the EE orientation as a quaternion: 4,
            # Convert quaternion to Euler angles if needed
            ee_euler = mujoco.mjlib.mju_quat2Euler(ee_quat)

            ee_pose = np.concatenate([ee_pos, ee_euler])
            #simulate noise
            noise = self.robot_noise_ratio * self.np_random.uniform(low=-0.01, high=0.01, size=ee_pose.shape)
            return ee_pose + noise

    def reset_model(self):
        # TODO: introduce randomizations in initial pos and vel

        self.current_step = 0

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=(9,))
        qpos[-1] = qpos[-2] # 2 fingers; only for fr3's default gripper
        qvel = np.zeros(9)
        self.set_state(qpos, qvel)
        obs = self._get_obs()

        return obs
    
if __name__ == "__main__":
    env = FrankaFR3Robot(render_mode='human')
    import time
    st = time.time()
    while time.time()-st < 5.0:
        env.step(env.action_space.sample())
        env.render()
    env.close()
