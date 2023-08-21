from torch.utils.data import Dataset

import scipy.io
import os


class SurrealDataset(Dataset):
    def __init__(self, root, split="train", overlap="50%"):
        super(SurrealDataset, self).__init__()
        self.root = root

        runs = ["run0", "run1", "run2"]
        if overlap == "50%":
            runs = [runs[0]]
        elif overlap == "30%":
            runs = [runs[1]]
        elif overlap == "70%":
            runs = [runs[2]]

        self.samples = []
        for run in runs:
            run_dir = os.path.join(root, split, run)
            for seq_name_dir_name in os.listdir(run_dir):
                seq_name_dir = os.path.join(run_dir, seq_name_dir_name)
                for seq_name in os.listdir(seq_name_dir):
                    if seq_name.endswith(".mp4"):
                        self.samples.append(
                            (
                                os.path.join(seq_name_dir, seq_name),
                                scipy.io.loadmat(
                                    os.path.join(
                                        seq_name_dir, seq_name[:-4] + "_info.mat"
                                    )
                                ),
                            )
                        )

        self.num_joints = 16

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
