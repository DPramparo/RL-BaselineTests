import os
from subprocess import Popen, PIPE
cwd = os.getcwd()
os.chdir('rl-baselines3-zoo/')

args = [
    '-n', str(100000),
    '--algo', 'ppo',
    '--env', 'SpaceInvaders-v0'
]

p = Popen(['python', 'train.py'] + args,
                           stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()
rc = p.returncode
os.chdir(cwd)


os.chdir('rl-baselines3-zoo/')

args = [
    '--algo', 'ppo',
    '--env', 'SpaceInvaders-v0',
    '--folder', 'logs/'
]

p = Popen(['python', 'enjoy.py'] + args,
                               stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate()
rc = p.returncode
os.chdir(cwd)
print(rc)

os.chdir('rl-baselines3-zoo/')

args = [
    '--algo', 'ppo',
    '--env', 'SpaceInvaders-v0',
    '--exp-folder', 'logs/'
]

p = Popen(['python', '-m', 'scripts.plot_train'] + args, stdout=PIPE)
output, err = p.communicate()
rc = p.returncode
os.chdir(cwd)


