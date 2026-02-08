import gymnasium as gym
import unittest
from simglucose.controller.basal_bolus_ctrller import BBController


def custom_reward(BG_last_hour):
    if BG_last_hour[-1] > 180:
        return -1
    elif BG_last_hour[-1] < 70:
        return -2
    else:
        return 1


class TestCustomReward(unittest.TestCase):
    def test_custom_reward(self):
        from gymnasium.envs.registration import register
        register(
            id='simglucose-adolescent3-v0',
            entry_point='simglucose.envs:T1DSimEnv',
            kwargs={
                'patient_name': 'adolescent#003',
                'reward_fun': custom_reward
            })

        env = gym.make('simglucose-adolescent3-v0')
        ctrller = BBController()

        reward = 1
        terminated = False
        truncated = False
        observation, info = env.reset()
        for t in range(200):
            env.render()
            print(observation)
            # action = env.action_space.sample()
            ctrl_action = ctrller.policy(observation, reward, terminated, **info)
            action = ctrl_action.basal + ctrl_action.bolus
            observation, reward, terminated, truncated, info = env.step(action)
            print("Reward = {}".format(reward))
            if observation.CGM > 180:
                self.assertEqual(reward, -1)
            elif observation.CGM < 70:
                self.assertEqual(reward, -2)
            else:
                self.assertEqual(reward, 1)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                break


if __name__ == '__main__':
    unittest.main()
