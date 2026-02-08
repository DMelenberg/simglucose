import gymnasium as gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController


class TestGym(unittest.TestCase):
    def test_gym_random_agent(self):
        from gymnasium.envs.registration import register
        register(
            id='simglucose-adolescent2-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={'patient_name': 'adolescent#002'}
        )

        env = gym.make('simglucose-adolescent2-v0')
        ctrller = BBController()

        reward = 0
        terminated = False
        truncated = False
        info = {'sample_time': 3,
                'patient_name': 'adolescent#002',
                'meal': 0}

        observation, info = env.reset()
        for t in range(200):
            env.render()
            print(observation)
            # action = env.action_space.sample()
            ctrl_action = ctrller.policy(observation, reward, terminated, **info)
            action = ctrl_action.basal + ctrl_action.bolus
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
