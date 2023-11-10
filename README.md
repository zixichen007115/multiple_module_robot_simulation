# multiple_module_robot_simulation

## Dependencies
`pip install -r requriements.txt`

# 0_files
All the files (NN dataset, NN, trajectory, etc.) generated during training and control

# 1_sim
The four-module robot is actuated randomly to generate a dataset for NN training.

`python data_generation.py --data_kind pseran`

# 2_mapping
To train an LSTM controller for end pose control

`python main.py --mode train`

To test

`python main.py --mode test`

# 3_ctrl
To control the four-module robot to keep the end orientation/position invariant and change the end position/orientation

`python ctrl.py --task o`/`python ctrl.py --task p`

For visualization

`python show.py --task o`/`python plot.py --task o`

`python show.py --task p`/`python plot.py --task p`
