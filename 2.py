#Realizacion de los diferentes algoritmos
#Sarsa
import matplotlib.pyplot as plt
import gym
import numpy as np
from funciones_utiles import plot_reward_per_episode, plot_steps_per_episode, plot_steps_per_episode_smooth, draw_value_matrix

# se crea el diccionario que contendrá los valores de Q para cada tupla (estado, acción)
q = {}

# definimos los híper-parámetros básicos

alpha = 0.5
gamma = 1
epsilon = 0.1
tau = 25

episodes_to_run = 500

env = gym.make("CliffWalking-v0")
actions = range(env.action_space.n)

# se declara una semilla aleatoria
random_state = np.random.RandomState(42)


#Funcion para tomar una accion
def choose_action(state):
    """
    Elige una acción según el aprendizaje realizado, usando una
    política de exploración épsilon-greedy
    """
    q_values = [q.get((state, a), 0.0) for a in actions]  # ej: para 4 acciones inicializa en [0,0,0,0]
    max_q = max(q_values)

    if random_state.uniform() < epsilon:  # sorteamos un número: es menor a épsilon?
        return random_state.choice(actions)  # sí: se selecciona una acción aleatoria

    count = q_values.count(max_q)

    # hay más de un máximo valor de estado-acción?
    if count > 1:
        # sí: seleccionamos uno de ellos aleatoriamente
        best = [i for i in range(len(actions)) if q_values[i] == max_q]
        i = random_state.choice(best)
    else:
        # no: seleccionamos el máximo valor de estado-acción
        i = q_values.index(max_q)

    return actions[i]

#Funcion que realiza las iteracioens
def run():
    """
    Corre el agente de RL
    """
    timesteps_of_episode = []  # registro de la cantidad de pasos que le llevó en cada episodio
    reward_of_episode = []  # cantidad de recompensa que recibió el agente en cada episodio

    for i_episode in range(episodes_to_run):
        # se ejecuta una instancia del agente hasta que el mismo llega a la salida
        # o tarda más de 2000 pasos

        # reinicia el ambiente, obteniendo el estado inicial del mismo
        state = env.reset()

        episode_reward = 0
        done = False
        t = 0

        # elige una acción basado en el estado actual
        action = choose_action(state)

        while not done:

            # el agente ejecuta la acción elegida y obtiene los resultados
            next_state, reward, done, info = env.step(action)

            next_action = choose_action(next_state)

            episode_reward += reward
            learn(state, action, reward, next_state, next_action)

            if not done and t < 2000:
                state = next_state
                action = next_action
            else:
                # el algoritmo no ha podido llegar a la meta antes de dar 2000 pasos
                done = True  # se establece manualmente la bandera done
                timesteps_of_episode = np.append(timesteps_of_episode, [int(t + 1)])
                reward_of_episode = np.append(reward_of_episode, max(episode_reward, -100))

            t += 1
            if i_episode % 100 == 0:
                print('[Episode {}] - Mean rewards {}.'.format(i_episode, np.mean(reward_of_episode)))
    return reward_of_episode.mean(), timesteps_of_episode, reward_of_episode

####Funcion de aprendizaje
def learn(state, action, reward, next_state, next_action):
    """
    Dado un (estado, acción, recompensa, estado siguiente),
    realiza una actualización SARSA de Q(s,a)
    """

    q_back = q.get((state,action), 0.0)
    q_next = q.get((next_state, next_action),0.0)
    q[(state,action)] = q_back + alpha * ( reward + gamma  * q_next - q_back)

    #q[(state, action)] = 0  # TODO - completa con tu código aquí


#Corro el script para ver su funcionamiento
avg_reward_per_episode, timesteps_ep, reward_ep = run()

plot_reward_per_episode(reward_ep)

plot_steps_per_episode(timesteps_ep)
plot_steps_per_episode_smooth(timesteps_ep)
draw_value_matrix(q)
env.close()
