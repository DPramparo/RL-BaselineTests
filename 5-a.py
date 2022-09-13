#En este archivo, se utiliza el entorno creado a mano para ver como se comporta.
from EnvGoLeft import GoLeftEnv, GoGrid
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import DQN, PPO

env = GoGrid(width=4, height= 4)
env = make_vec_env(lambda: env, n_envs=1)

model = PPO('MlpPolicy', env, verbose=1).learn(20000)


obs = env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='console')
    if dones[0]:
        obs = env.reset()
