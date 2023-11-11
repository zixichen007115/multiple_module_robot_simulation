import numpy as np
from run_simulation_dynamic import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='p', choices=['p', 'o'])
config = parser.parse_args()

ctrl_step = 500

# trajectory p
if config.task == 'p':
    ang = 0.8
    x_tar = np.zeros(ctrl_step)
    y_tar = np.zeros(ctrl_step)
    ax_tar = np.cos(np.linspace(0, 2 * np.pi, ctrl_step))
    ay_tar = np.sin(np.linspace(0, 2 * np.pi, ctrl_step))
    az_tar = np.ones(ctrl_step)*ang

# trajectory o
else:
    ang = 1
    x_tar = np.zeros(ctrl_step)
    y_tar = np.zeros(ctrl_step)
    ax_tar = np.cos(np.linspace(0, 2 * np.pi, ctrl_step))
    ay_tar = np.sin(np.linspace(0, 2 * np.pi, ctrl_step))
    az_tar = np.ones(ctrl_step)*ang

    part = int(ctrl_step/5)

    rad = 0.4
    x_tar[:part] = np.linspace(0, rad, part)
    y_tar[:part] = 0
    x_tar[part:] = np.cos(np.linspace(0, 2*np.pi, ctrl_step-part))*rad
    y_tar[part:] = np.sin(np.linspace(0, 2*np.pi, ctrl_step-part))*rad

    ax_tar[:part] = 1
    ay_tar[:part] = 0
    ax_tar[part:] = np.cos(np.linspace(0, 2*np.pi, ctrl_step-part))
    ay_tar[part:] = np.sin(np.linspace(0, 2*np.pi, ctrl_step-part))

rest = 1-np.square(ang)
if rest < 1e-10:
    ax_tar = np.zeros(ctrl_step)
    ay_tar = np.zeros(ctrl_step)
else:
    for i in range(ctrl_step):
        ax_tar[i] = np.sqrt(np.square(ax_tar[i])*rest)*np.sign(ax_tar[i])
        ay_tar[i] = np.sqrt(np.square(ay_tar[i])*rest)*np.sign(ay_tar[i])

# simulation
pos_list, dir_list, act_list, real_list, shape_list = main(ctrl_step=ctrl_step,
                                                           tar_list=(x_tar, y_tar, ax_tar, ay_tar, az_tar))
# save robot motion
np.savez('../0_files/data_follow_'+str(config.task), pos_list=pos_list, dir_list=dir_list, act_list=act_list,
         real_list=real_list, shape_list=shape_list)
