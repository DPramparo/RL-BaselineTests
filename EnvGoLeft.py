

import numpy as np


import gym
from gym import spaces


class GoLeftEnv(gym.Env):
  """
  Ambiente personalizado que sigue la interfaz de gym.
  Es un entorno simple en el cuál el agente debe aprender a ir siempre
  hacia la izquierda.
  """
  # Dado que estamos en colab, no podemos implementar la salida por interfaz
  # gráfica ('human' render mode)
  metadata = {'render.modes': ['console']}
  # Definimos las constantes
  LEFT = 0
  RIGHT = 1

  def __init__(self, grid_size=10):
    super(GoLeftEnv, self).__init__()

    # Tamaño de la grilla de 1D
    self.grid_size = grid_size
    # Inicializamos en agente a la derecha de la grilla
    self.agent_pos = grid_size - 1

    # Definimos el espacio de acción y observaciones
    # Los mismos deben ser objetos gym.spaces
    # En este ejemplo usamos dos acciones discretas: izquierda y derecha
    n_actions = 2
    self.action_space = spaces.Discrete(n_actions)
    # La observación será la coordenada donde se encuentra el agente
    # puede ser descrita tanto por los espacios Discrete como Box
    self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                        shape=(1,), dtype=np.float32)

  def reset(self):
    """
    Importante: la observación devuelta debe ser un array de numpy
    :return: (np.array)
    """
    # Se inicializa el agente a la derecha de la grilla
    self.agent_pos = self.grid_size - 1
    # convertimos con astype a float32 (numpy) para hacer más general el agente
    # (en caso de que querramos usar acciones continuas)
    return np.array([self.agent_pos]).astype(np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.agent_pos -= 1
    elif action == self.RIGHT:
      self.agent_pos += 1
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # Evitamos que el agente se salga de los límites de la grilla
    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

    # Llegó el agente a su estado objetivo (izquierda) de la grilla?
    done = bool(self.agent_pos == 0)

    # Asignamos recompensa sólo cuando el agente llega a su objetivo
    # (recompensa = 0 en todos los demás estados)
    reward = 1 if self.agent_pos == 0 else 0

    # gym también nos permite devolver información adicional, ej. en atari:
    # las vidas restantes del agente (no usaremos esto por ahora)
    info = {}

    return np.array([self.agent_pos]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # en nuestra interfaz de consola, representamos el agente como una cruz, y
    # el resto como un punto
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size - self.agent_pos))

  def close(self):
    pass

###Entorno de grilla con diferentes obstaculos
class GoGrid(gym.Env):
  """
  Ambiente personalizado que sigue la interfaz de gym.

  """
  # Dado que estamos en colab, no podemos implementar la salida por interfaz
  # gráfica ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Definimos las constantes
  LEFT = 0
  RIGHT = 1
  UP = 2
  DOWN = 3

#Aumento el tamaño respecto al original
  def __init__(self, width, height):
    super(GoGrid, self).__init__()
    self.width = width
    self.height = height
    self.layers = ('#', 'P')
    self.grid_size = width * height

    # Definimos el espacio de acción y observaciones
    # Los mismos deben ser objetos gym.spaces
    # Tendra 4 acciones, arriba, abajo, izquierda y derecha
    n_actions = 4
    self.action_space = spaces.Discrete(n_actions)

    self.observation_space = spaces.Box(
      low=0, high=1,
      shape=(self.width, self.height, len(self.layers)),
      dtype=np.int32
    )

  def reset(self):
    """
    Importante: la observación devuelta debe ser un array de numpy
    :return: (np.array)
    """
    # Se inicializa el agente a la derecha de la grilla
    self.agent_pos = self.grid_size - 1
    # convertimos con astype a float32 (numpy) para hacer más general el agente
    # (en caso de que querramos usar acciones continuas)
    return np.array([self.agent_pos]).astype(np.float32)

  def step(self, action):
    if action == self.LEFT:
      self.agent_pos -= 1
    elif action == self.RIGHT:
      self.agent_pos += 1
    elif action == self.UP:
      self.agent_pos += 2
    elif action == self.DOWN:
      self.agent_pos -=2
    else:
      raise ValueError("Received invalid action={} which is not part of the action space".format(action))

    # Evitamos que el agente se salga de los límites de la grilla
    self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

    # Llegó el agente a su estado objetivo (izquierda) de la grilla?
    done = bool(self.agent_pos == 0)
    #Castigamos al agente cuando se choca una pared en la posicion 5
    reward = -100 if self.agent_pos == 5 else 0
    # Asignamos recompensa sólo cuando el agente llega a su objetivo
    # (recompensa = 0 en todos los demás estados)
    reward = 1
    if self.agent_pos == 0:
      reward = 1
    elif self.agent_pos != 0:
      reward = 0
    elif self.agent_pos == 5:
      reward = -100


    # gym también nos permite devolver información adicional, ej. en atari:
    # las vidas restantes del agente (no usaremos esto por ahora)
    info = {}

    return np.array([self.agent_pos]).astype(np.float32), reward, done, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # en nuestra interfaz de consola, representamos el agente como una cruz, y
    # el resto como un punto
    print("." * self.agent_pos, end="")
    print("x", end="")
    print("." * (self.grid_size))

  def close(self):
    pass