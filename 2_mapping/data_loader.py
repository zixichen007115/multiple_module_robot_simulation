import sys

from torch.utils import data
import random
import numpy as np

class Data_sim(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, mode, t_step):
        """Initialize and preprocess the CelebA dataset."""
        self.mode = mode
        self.t_step = t_step
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.preprocess()

    def preprocess(self):

        data = np.load("../0_files/data_pseran.npz")
        pos_list = data["pos_list"]
        dir_list = data["dir_list"]
        act_list = data["act_list"].T
        # pos_list: segment, 3, steps
        # dir_list: segment, 3, 3, steps
        # act_list: segment*2, steps

        print("position  list shape:{}".format(np.shape(pos_list)))
        print("direction list shape:{}".format(np.shape(dir_list)))
        print("action    list shape:{}".format(np.shape(act_list)))
        print("action range:%.3f_%.3f"%(np.min(act_list),np.max(act_list)))

        random.seed(1)
        list_length = np.shape(act_list)[1]
        val_test_sample = random.sample(range(list_length),int(list_length*0.3))
        val_sample = val_test_sample[:int(list_length*0.1)]
        test_sample = val_test_sample[int(list_length*0.1):]

        ori_vec = np.zeros([3, 1])
        ori_vec[2] = 1

        vec = np.zeros([list_length, 3])
        for i in range(list_length):
            vec[i] = np.matmul(dir_list[-1, :, :, i].T, ori_vec)[:, 0]

        t_step = self.t_step

        for i in range(1,list_length-t_step-5):
            seg_input = np.zeros([t_step, 13])
            # 8 for actuation, 2 for end position, 3 for end orientation
            output = np.zeros([t_step, 8])
            for k in range(t_step):
                seg_input[k, :8] = act_list[:, i + k]
                seg_input[k, 8:10] = pos_list[-1, :2, i + k + 2] * 10
                seg_input[k, 10:] = vec[i+k+2]

                output[k] = act_list[:, i + k+1]

            if i in val_sample:
                self.val_dataset.append([seg_input.transpose(), output.transpose()])
            elif i in test_sample:
                self.test_dataset.append([seg_input.transpose(), output.transpose()])
            else:
                self.train_dataset.append([seg_input.transpose(), output.transpose()])

        print('Finished preprocessing the dataset...')
        print('train sample number: %d.'%len(self.train_dataset))
        print('validation sample number: %d.'%len(self.val_dataset))
        print('test sample number: %d.'%len(self.test_dataset))

    def __getitem__(self, index):
        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'test':
            dataset = self.test_dataset
        else:
            dataset = self.val_dataset
        seg_input, output = dataset[index]

        return seg_input.transpose(), output.transpose()


    def __len__(self):
        """Return the number of images."""
        if self.mode == 'train':
            return len(self.train_dataset)
        elif self.mode == 'test':
            return len(self.test_dataset)
        else:
            return len(self.val_dataset)

def get_loader(batch_size=32, mode='train',num_workers=1, t_step=10):
    """Build and return a data loader."""
    dataset = Data_sim(mode=mode, t_step=t_step)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True if mode=='train' else False,
                                  num_workers=num_workers)
    return data_loader