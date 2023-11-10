"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import sys

import numpy as np
from tqdm import tqdm
from elastica._rotations import _inv_rotate
from coomm.algorithms import ForwardBackwardMuscle
from coomm.objects import PointTarget
from coomm.callback_func import AlgorithmMuscleCallBack

from set_arm_environment import ArmEnvironment

def get_algo(rod, muscles):
    algo = ForwardBackwardMuscle(
        rod=rod,
        muscles=muscles,
        algo_config = dict(
            stepsize=1e-8,
            activation_diff_tolerance=1e-12
        ),
    )

    return algo

class Environment(ArmEnvironment):
    
    def get_data(self):
        return [self.rod_parameters_dict]

    def setup(self):
        self.set_arm()

def main(ctrl_step=1, par_young=1, par_LM=1, act_list=None):

    """ Create simulation environment """
    # final_time = 15.001
    time_step = 2.5e-4
    controller_Hz = 1
    final_time = int(ctrl_step/controller_Hz)
    # final_time = 50

    youngs_modulus = par_young * 10_000
    LM_ratio_muscle_position_parameter = par_LM * 0.0075
    # 1.501

    env = Environment(final_time, time_step=time_step, youngs_modulus=youngs_modulus,LM_ratio_muscle_position_parameter=LM_ratio_muscle_position_parameter)
    total_steps, systems = env.reset()
    
    controller_step_skip = int(1.0 / (controller_Hz * env.time_step))

    """ Initialize algorithm """
    algo = get_algo(
        rod=systems[0],
        muscles=env.muscle_groups
    )
    algo_callback = AlgorithmMuscleCallBack(step_skip=env.step_skip)


    """ Read arm params """
    activations = []
    for m in range(len(env.muscle_groups)):
        activations.append(
            np.zeros(env.muscle_groups[m].activation.shape)
        )


    """ Start the simulation """
    print("Running simulation ...")
    time = np.float64(0.0)

    segment = 4

    pos_list = np.zeros((segment, 3, len(act_list[:,0])))

    dir_list = np.zeros((segment, 3, 3, len(act_list[:, 0])))

    ctrl_num = 0



    for k_sim in tqdm(range(total_steps)):

        if (k_sim % controller_step_skip) == 0:
            activations[0] = np.concatenate((np.ones(25)*np.max([0,act_list[ctrl_num,0]]),np.ones(25)*np.max([0,act_list[ctrl_num,2]]),
                np.ones(25)*np.max([0,act_list[ctrl_num,4]]),np.ones(25)*np.max([0,act_list[ctrl_num,6]])))
            activations[1] = np.concatenate((np.ones(25)*np.max([0,act_list[ctrl_num,1]]),np.ones(25)*np.max([0,act_list[ctrl_num,3]]),
                np.ones(25)*np.max([0,act_list[ctrl_num,5]]),np.ones(25)*np.max([0,act_list[ctrl_num,7]])))
            activations[2] = np.concatenate((np.ones(25)*np.max([0,-act_list[ctrl_num,0]]),np.ones(25)*np.max([0,-act_list[ctrl_num,2]]),
                np.ones(25)*np.max([0,-act_list[ctrl_num,4]]),np.ones(25)*np.max([0,-act_list[ctrl_num,6]])))
            activations[3] = np.concatenate((np.ones(25)*np.max([0,-act_list[ctrl_num,1]]),np.ones(25)*np.max([0,-act_list[ctrl_num,3]]),
                np.ones(25)*np.max([0,-act_list[ctrl_num,5]]),np.ones(25)*np.max([0,-act_list[ctrl_num,7]])))
            # activations[4] = np.ones(100)*np.max([0,act_list[ctrl_num,0]])
            # activations[5] = np.ones(100)*np.max([0,act_list[ctrl_num,1]])
            # activations[6] = np.ones(100)*np.max([0,-act_list[ctrl_num,0]])
            # activations[7] = np.ones(100)*np.max([0,-act_list[ctrl_num,1]])

            for i_col in range(4):
                pos_list[i_col,:,ctrl_num] = env.shearable_rod.position_collection[:,25+i_col*25]
                dir_list[i_col,:,:,ctrl_num] = env.shearable_rod.director_collection[:,:,24+i_col*25]


            # first_pos = env.shearable_rod.position_collection[:,25]
            # first_pos_list[:,ctrl_num] = first_pos
            # mid_pos = env.shearable_rod.position_collection[:,50]
            # mid_pos_list[:,ctrl_num] = mid_pos
            # third_pos = env.shearable_rod.position_collection[:,75]
            # third_pos_list[:,ctrl_num] = third_pos
            # end_pos = env.shearable_rod.position_collection[:,100]
            # end_pos_list[:,ctrl_num] = end_pos


            # dir = env.shearable_rod.director_collection
            # pos = env.shearable_rod.position_collection
            #
            # print(np.shape(dir))
            # print(np.shape(pos))
            # sys.exit()


            # first_dir = env.shearable_rod.director_collection[:, :, 0]
            # first_dir_list[:, :, ctrl_num] = (first_dir)
            # mid_dir = env.shearable_rod.director_collection[:, :, 49]
            # mid_dir_list[:, :, ctrl_num] = (mid_dir)
            # third_dir = env.shearable_rod.director_collection[:, :, 74]
            # third_dir_list[:, :, ctrl_num] = (third_dir)
            # end_dir = env.shearable_rod.director_collection[:, :, 99]
            # end_dir_list[:, :, ctrl_num] = (end_dir)
            
            ctrl_num = ctrl_num+1

            




        algo_callback.make_callback(algo, time, k_sim)
        time, systems, done = env.step(time, activations)
    print(ctrl_num)

    # env.save_data(
    #     filename='sim',
    #     algo=algo_callback.callback_params,
    # )

    return pos_list, dir_list, act_list
