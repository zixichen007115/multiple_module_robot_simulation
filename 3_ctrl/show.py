import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='p', choices=['p', 'o'])
config = parser.parse_args()

data = np.load("../0_files/data_follow_" + str(config.task) + ".npz")
data_ws = np.load("../0_files/data_pseran.npz")

pos_list = data_ws["pos_list"]*10
act_list = data["act_list"].T
real_list = data["real_list"]

# act_list: 2, segment, steps
# real_list: segment, 2, steps

print(np.shape(act_list))
print(np.shape(real_list))

step = 10
ctrl_step = 500

if config.task == 'p':
    ang = 0.8
    x_tar = np.zeros(ctrl_step)
    y_tar = np.zeros(ctrl_step)
    ax_tar = np.cos(np.linspace(0, 2 * np.pi, ctrl_step))
    ay_tar = np.sin(np.linspace(0, 2 * np.pi, ctrl_step))
    az_tar = np.ones(ctrl_step)*ang

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
if rest<1e-10:
    ax_tar = np.zeros(ctrl_step)
    ay_tar = np.zeros(ctrl_step)
else:
    for i in range(ctrl_step):
        ax_tar[i] = np.sqrt(np.square(ax_tar[i])*rest)*np.sign(ax_tar[i])
        ay_tar[i] = np.sqrt(np.square(ay_tar[i])*rest)*np.sign(ay_tar[i])

x_real, y_real, ax_real, ay_real, az_real  = real_list

plt.figure(figsize=(8,8))
plt.plot(x_tar,y_tar,c='b',label='target')
plt.plot(x_real,y_real,c='r',label='real')
plt.scatter(pos_list[-1,0],pos_list[-1,1],c='orange',alpha=0.25)
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.legend(fontsize=15)
plt.show()

plt.figure(figsize=(12,8))
plt.subplot(221)
plt.plot(x_tar,c='b')
plt.plot(x_real,c='r')
plt.ylim([-1.1, 1.1])
plt.ylabel('x')
plt.subplot(222)
plt.plot(y_tar,c='b')
plt.plot(y_real,c='r')
plt.ylim([-1.1, 1.1])
plt.ylabel('y')
plt.subplot(223)
plt.plot(ax_tar,c='b')
plt.plot(ax_real,c='r')
plt.ylim([-1.1, 1.1])
plt.ylabel('cos')
plt.subplot(224)
plt.plot(az_tar,c='b')
plt.plot(az_real,c='r')
plt.ylim([-1.1, 1.1])
plt.ylabel('bending')
plt.show()
