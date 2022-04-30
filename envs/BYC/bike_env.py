import pybullet as p
import pybullet_data
import numpy as np
import gym


class BikeEnv(gym.Env):
    def __init__(self, render=True):
        self.render = render
        self.action_space = gym.spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = gym.spaces.Box(np.array([-100] * 12), np.array([100] * 12))
        p.connect(p.GUI if self.render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=50,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
        self._timestep = 1 / 240
        # p.changeDynamics(self.plane_id, -1, lateralFriction=0, spinningFriction=0, rollingFriction=0.15)

        self.robot_id = p.loadURDF("../envs/BYC/urdf/byc.urdf")
        self.front_fork_ind = 0
        self.front_wheel_ind = 1
        self.rear_fork_ind = 2
        self.rear_wheel_ind = 3
        self.available_joints_indexes = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]

        # coefficients of rewards
        self.FR = 1
        self.HR = 1
        self.CC = 0.5

    def get_current_observation(self):
        if not hasattr(self, "robot_id"):
            assert Exception("robot hasn't been loaded in!")
        base_pos, base_ori = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_angvel = p.getBaseVelocity(self.robot_id)
        base_euler = p.getEulerFromQuaternion(base_ori)

        roll = base_euler[0]  # q1
        d_roll = base_angvel[0]  # dq1
        yaw = base_euler[2]  # q2
        d_yaw = base_angvel[2]  # dq2
        front_fork, d_front_fork, _1, _2 = p.getJointState(self.robot_id, self.front_fork_ind)  # q3 and dq3
        front_wheel, d_front_wheel, _1, _2 = p.getJointState(self.robot_id, self.front_fork_ind)  # q4 and dq4
        rear_fork, d_rear_fork, _1, _2 = p.getJointState(self.robot_id, self.front_fork_ind)  # q5 and dq5
        rear_wheel, d_rear_wheel, _1, _2 = p.getJointState(self.robot_id, self.front_fork_ind)  # q6 and dq6

        return roll, d_roll, yaw, d_yaw, front_fork, d_front_fork, front_wheel, d_front_wheel, rear_fork, d_rear_fork, rear_wheel, d_rear_wheel

    def step(self, action):
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=[self.front_fork_ind, self.front_wheel_ind, self.rear_fork_ind, self.rear_wheel_ind],
            controlMode=p.TORQUE_CONTROL,
            forces=action
        )
        p.stepSimulation()
        roll, d_roll, yaw, d_yaw, front_fork, d_front_fork, front_wheel, d_front_wheel, rear_fork, d_rear_fork, rear_wheel, d_rear_wheel = self.get_current_observation()
        self.state = np.array([roll, d_roll, yaw, d_yaw, front_fork, d_front_fork, front_wheel, d_front_wheel, rear_fork, d_rear_fork, rear_wheel, d_rear_wheel])
        state = self.state

        self.last_action = action
        done = bool(abs(roll) > 30 * np.pi / 180)
        reward = self.get_reward(state, action)
        if done:
            reward -= 1
        info = []
        self.simtime += self._timestep
        return state, reward, done, info

    def reset(self):
        self.simtime = 0
        p.setGravity(0, 0, -9.81)

        base_euler = np.array([np.random.uniform(-15 * np.pi / 360, 15 * np.pi / 360), 0, 0])
        base_pos = np.array([0, 0, 0.8249*np.cos(base_euler[0])])
        base_ori = p.getQuaternionFromEuler(base_euler)
        base_vel = np.random.normal(np.zeros(3), 0.012 * np.ones(3))
        base_angvel = np.random.normal(np.zeros(3), 0.4 * np.ones(3))
        p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_ori)
        p.resetBaseVelocity(self.robot_id, base_vel, base_angvel)

        front_fork = np.clip(np.random.normal(0, 0.01), -np.pi / 2, np.pi / 2)
        d_front_fork = np.random.normal(0, 2)
        front_wheel = 0
        d_front_wheel = np.random.normal(0, 0.15)
        rear_fork = np.clip(np.random.normal(0, 0.01), -np.pi / 2, np.pi / 2)
        d_rear_fork = np.random.normal(0, 0.15)
        rear_wheel = 0
        d_rear_wheel = np.random.normal(0, 0.15)
        p.resetJointState(self.robot_id, self.front_fork_ind, front_fork, d_front_fork)
        p.resetJointState(self.robot_id, self.front_wheel_ind, front_wheel, d_front_wheel)
        p.resetJointState(self.robot_id, self.rear_fork_ind, rear_fork, d_rear_fork)
        p.resetJointState(self.robot_id, self.rear_wheel_ind, rear_wheel, d_rear_wheel)


        roll = base_euler[0]
        d_roll = base_angvel[0]
        yaw = base_euler[2]
        d_yaw = base_angvel[2]
        self.state = np.array([roll, d_roll, yaw, d_yaw, front_fork, d_front_fork, front_wheel, d_front_wheel, rear_fork, d_rear_fork, rear_wheel, d_rear_wheel])
        self.last_action = np.zeros(4)
        self.desired_vel = np.array([np.random.uniform(3, 5), 0, 0])
        return self.state

    def get_reward(self, state, action):
        base_vel, _ = p.getBaseVelocity(self.robot_id)
        roll, d_roll, yaw, d_yaw, front_fork, d_front_fork, front_wheel, d_front_wheel, rear_fork, d_rear_fork, rear_wheel, d_rear_wheel = state
        forward_reward = self.FR * base_vel[0]
        healthy_reward = self.HR * 1
        control_cost = self.CC * np.linalg.norm(action) ** 2
        reward = forward_reward + healthy_reward - control_cost
        return reward

    def close(self):
        pass