import gymnasium as gym
import unittest
from gymnasium.envs.registration import register

register(
    id='simglucose-adult1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adult#001'}
)


class TestSeed(unittest.TestCase):
    def test_changing_seed_generates_different_results(self):
        env = gym.make('simglucose-adult1-v0')

        observation_seed0, _ = env.reset(seed=0)
        start_time_seed0 = env.env.scenario.start_time

        observation_seed1, _ = env.reset(seed=1000)
        start_time_seed1 = env.env.scenario.start_time

        self.assertNotEqual(start_time_seed0, start_time_seed1)
        self.assertNotEqual(observation_seed0, observation_seed1)


if __name__ == '__main__':
    unittest.main()
