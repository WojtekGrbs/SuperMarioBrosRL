import gym
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation


RENDER_MODE = 'human'
SUPER_MARIO_BROS_VERSION = 'SuperMarioBros-v3'
FRAMES_TO_SKIP = 6
STACKED_FRAMES = 4
ENVIRONMENT_SIZE = 32

def generate_env():
    # Creating the env
    env = gym_super_mario_bros.make(SUPER_MARIO_BROS_VERSION, render_mode=RENDER_MODE, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = ResizeObservation(env, shape=ENVIRONMENT_SIZE)
    env = FrameSkippingWrapper(env, frames_to_skip=FRAMES_TO_SKIP)
    env = GrayScaleObservation(env)
    return FrameStack(env, num_stack=STACKED_FRAMES)


class FrameSkippingWrapper(Wrapper):
    def __init__(self, env, frames_to_skip: int):
        super().__init__(env)
        self.frames_to_skip = frames_to_skip

    def step(self, action):

        state, reward_sum, done, truncated, info = None, 0, False, False, None

        for _ in range(self.frames_to_skip):
            state, reward, done, truncated, info = self.env.step(action)

            # If the game ends before the whole skip
            if done is True:
                break

        # Returning the final state of mario, with sum of rewards as reward
        return state, reward_sum, done, truncated, info
