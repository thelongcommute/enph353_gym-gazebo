import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Gazebo
# ----------------------------------------

# cart pole
register(
    id='GazeboCartPole-v0',
    entry_point='gym_gazebo.envs.gazebo_cartpole:GazeboCartPolev0Env',
)

register(
    id='Gazebo_Lab06-v0',
    entry_point='gym_gazebo.envs.gazebo_lab06:Gazebo_Lab06_Env',
    max_episode_steps=3000,
)

