import os
import pickle

if __name__ == '__main__':
    root_dir = "/home/xlju/Project/OpenPCDet/drop_out_test_results"
    all_pkl = os.listdir(root_dir)
    all_pkl.sort()
    all_pkl_paths = [os.path.join(root_dir, i) for i in all_pkl]
    sample = all_pkl_paths[0]
    with open(sample, 'rb') as f:
        res = pickle.load(f)
        pass