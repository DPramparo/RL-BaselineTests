#En este archivo, hago una entrada en calor de las librerias propuestas por la catedra.

import os
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
#############################PRUEBAS PILOTO
os.makedirs('logs', exist_ok=True)

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

cwd = os.getcwd()

env = gym.make('CartPole-v1')

# MlpPolicy es una política "estándar" que aprende con perceptron multicapa
# (es decir sin capas convolucionales o demás variantes),
# 2 capas ocultas con 64 neuronas cada una
model = DQN('MlpPolicy', env)
model.learn(total_timesteps=10000)


obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()

venv = make_vec_env(lambda: gym.make('CartPole-v1'), n_envs=1)

model = DQN('MlpPolicy', venv)
model.learn(total_timesteps=100000)


env = gym.make('CartPole-v1')
env = Monitor(env, 'logs/')  # reemplazamos env por su monitor

model = DQN('MlpPolicy', env, )
model.learn(total_timesteps=10000)

env = gym.make('CartPole-v1')

callbacks = []  # lista de callbacks a usar, pueden ser varios

# callback para detener entrenamiento al alcanzar recompensa de 9.8
# (a fines demostrativos, es una recompensa baja)
stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=9.8)

# al crear EvalCallback, se asocia el mismo con stop_training_callback
callbacks.append(EvalCallback(env,
                              eval_freq=1000,
                              callback_on_new_best=stop_training_callback))

# la semilla aleatoria hace que las ejecuciones sean determinísticas
model = DQN('MlpPolicy', env, seed=42)
model.learn(total_timesteps=10000, callback=callbacks)

#################FIN PRUEBAS


