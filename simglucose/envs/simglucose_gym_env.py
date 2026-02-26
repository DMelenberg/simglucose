from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import hashlib
from datetime import datetime
from pathlib import Path

PATIENT_PARA_FILE = Path(__file__).parent.parent / "params" / "vpatient_params.csv"


def _hash_seed(seed, max_bytes=8):
    seed = int(seed)
    hash_bytes = hashlib.sha512(str(seed).encode("utf-8")).digest()
    return int.from_bytes(hash_bytes[:max_bytes], "big")


def _np_random(seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    return np.random.RandomState(seed), seed


class T1DSimEnv(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    """

    metadata = {"render_modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = _np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        # This gym only controls basal insulin
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError("Expected action to contain a single value.")
        action_value = float(action_array.reshape(-1)[0])
        act = Action(basal=action_value, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        truncated = False
        return obs, reward, done, truncated, info

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        step = self.env.reset()
        return step.observation, step.info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        obs, info = self._reset()
        return obs, info

    def _seed(self, seed=None):
        self.np_random, seed1 = _np_random(seed=seed)
        seed_rng = np.random.RandomState(seed1)
        seed2 = _hash_seed(seed_rng.randint(0, 1000)) % 2**31
        seed3 = _hash_seed(seed2 + 1) % 2**31
        seed4 = _hash_seed(seed3 + 1) % 2**31
        return [seed1, seed2, seed3, seed4]

    def seed(self, seed=None):
        return self._seed(seed=seed)

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = _hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = _hash_seed(seed2 + 1) % 2**31
        seed4 = _hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(0, 24)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super().close()
        self.env._close_viewer()

    def render(self):
        self._render()

    def close(self):
        self._close()

    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]

    @property
    def scenario(self):
        return self.env.scenario

    @property
    def sensor(self):
        return self.env.sensor


class T1DSimGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Truncated will be controlled by TimeLimit wrapper when registering the env.
        # For example,
        # register(
        #     id="simglucose/adolescent2-v0",
        #     entry_point="simglucose.envs:T1DSimGymnasiumEnv",
        #     max_episode_steps=10,
        #     kwargs={"patient_name": "adolescent#002"},
        # )
        # Once the max_episode_steps is set, the truncated value will be overridden.
        return np.array([obs.CGM], dtype=np.float32), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        return np.array([obs.CGM], dtype=np.float32), info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()

