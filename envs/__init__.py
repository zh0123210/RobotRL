from gym.envs.registration import register

register(
    id='Bike-v0',
    entry_point='RobotRL_pytorch.envs.BYC:BikeEnv',
)
