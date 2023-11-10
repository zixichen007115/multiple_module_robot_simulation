"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import sys

import numpy as np
from tqdm import tqdm

from set_arm_environment import ArmEnvironment
import torch

def restore_model(input_size=13, hidden_size=128, num_layers=4, num_classes=8):
    from model import LSTM
    LSTM = LSTM(input_size, hidden_size, num_layers, num_classes, torch.device('cpu'))
    LSTM_path = '../0_files/LSTM-1.ckpt'
    LSTM.load_state_dict(torch.load(LSTM_path, map_location=lambda storage, loc: storage))
    return LSTM

class Environment(ArmEnvironment):
    def get_data(self):
        return [self.rod_parameters_dict]
    def setup(self):
        self.set_arm()

def main(ctrl_step=1, tar_list=None):
    """ Create simulation environment """
    # final_time = 15.001
    time_step = 2.5e-4
    controller_Hz = 1
    final_time = int(ctrl_step/controller_Hz)
    par_young = 1
    par_LM = 1
    youngs_modulus = par_young * 10_000
    LM_ratio_muscle_position_parameter = par_LM * 0.0075

    env = Environment(final_time, time_step=time_step, youngs_modulus=youngs_modulus,
                      LM_ratio_muscle_position_parameter=LM_ratio_muscle_position_parameter)
    total_steps, systems = env.reset()
    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(np.zeros(env.muscle_groups[m].activation.shape))

    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    num_seg  = 4
    t_step   = 10
    ctrl_num = 0
    ori_vec = np.zeros([3, 1])
    ori_vec[2] = 1

    x_tar,y_tar,ax_tar,ay_tar,az_tar = tar_list
    pos_list = np.zeros((num_seg, 3, len(x_tar)))
    shape_list = np.zeros((21, 3, len(x_tar)))
    dir_list = np.zeros((num_seg, 3, 3, len(x_tar)))
    real_list = np.zeros((5, len(x_tar)))

    lstm = restore_model()
    seg_input = np.zeros([t_step, 13])
    act = np.zeros(8)
    act_list = np.zeros((len(x_tar), 8))
    pre_act   = np.zeros(8)

    for k_sim in tqdm(range(total_steps)):
        if (k_sim % controller_step_skip) == 0:
            for i in range(21):
                shape_list[i, :, ctrl_num] = env.shearable_rod.position_collection[:, 5*i]
            for i_col in range(num_seg):
                pos_list[i_col, :, ctrl_num] = env.shearable_rod.position_collection[:, 25+i_col*25]
                dir_list[i_col, :, :, ctrl_num] = env.shearable_rod.director_collection[:, :, 24+i_col*25]

            for i_st in range(t_step - 1):
                seg_input[i_st] = np.copy(seg_input[i_st + 1])

            seg_input[-1, :8] = pre_act
            seg_input[-2, 8:10] = pos_list[-1, :2, ctrl_num]*10
            vec = np.matmul(dir_list[-1, :, :, ctrl_num].T, ori_vec)[:, 0]
            seg_input[-2, 10:] = vec

            seg_input[-1, 8:] = [x_tar[ctrl_num], y_tar[ctrl_num], ax_tar[ctrl_num], ay_tar[ctrl_num], az_tar[ctrl_num]]

            real_list[:, ctrl_num] = seg_input[-2, 8:]

            NN_input_tensor = torch.Tensor(np.array([seg_input]))

            with torch.no_grad():
                NN_input_tensor.to(torch.device('cpu'))
                out = lstm(NN_input_tensor)
                out = out.cpu().numpy()

            act = out[0, -1, :]

            act_max = 1
            for i in range(8):
                if np.abs(act[i]) > act_max:
                    act[i] = act_max*act[i]/np.abs(act[i])

            act_list[ctrl_num, :] = act
            pre_act = np.copy(act)

            activations[0] = np.concatenate(
                (np.ones(25) * np.max([0, act[0]]), np.ones(25) * np.max([0, act[2]]),
                 np.ones(25) * np.max([0, act[4]]), np.ones(25) * np.max([0, act[6]])))
            activations[1] = np.concatenate(
                (np.ones(25) * np.max([0, act[1]]), np.ones(25) * np.max([0, act[3]]),
                 np.ones(25) * np.max([0, act[5]]), np.ones(25) * np.max([0, act[7]])))
            activations[2] = np.concatenate(
                (np.ones(25) * np.max([0, -act[0]]), np.ones(25) * np.max([0, -act[2]]),
                 np.ones(25) * np.max([0, -act[4]]), np.ones(25) * np.max([0, -act[6]])))
            activations[3] = np.concatenate(
                (np.ones(25) * np.max([0, -act[1]]), np.ones(25) * np.max([0, -act[3]]),
                 np.ones(25) * np.max([0, -act[5]]), np.ones(25) * np.max([0, -act[7]])))

            ctrl_num = ctrl_num+1

        time, systems, done = env.step(time, activations)
    return pos_list, dir_list, act_list, real_list, shape_list
