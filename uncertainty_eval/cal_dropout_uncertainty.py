import os
import pickle
import pcdet.utils.object3d_kitti as object3d_kitti


label_folder = "/home/xlju/data/data_openpcdet/training/label_2"


def grab_all_pkl(test_log_dir):
    sub_dirs = [i for i in os.listdir(test_log_dir) if os.path.isdir(os.path.join(test_log_dir, i))]
    result_pkl_file_list = []
    for i in sub_dirs:
        result_file = os.listdir(os.path.join(test_log_dir, i))
        if "result.pkl" in result_file:
            result_pkl_file_list.append(os.path.join(test_log_dir, i, "result.pkl"))
    return result_pkl_file_list


def get_corresponding_gt(frm_idx):
    label_file = os.path.join(label_folder, '{:0>6d}.txt'.format(frm_idx))
    return object3d_kitti.get_objects_from_label(label_file)


if __name__ == '__main__':
    tmp = get_corresponding_gt(4)
    log_dir = "/home/xlju/Project/OpenPCDet/output/kitti_models/pv_rcnn/default/eval_dropout/epoch_8369/val/default"
    pkl_list = grab_all_pkl(log_dir)
    for i in pkl_list:
        with open(i, 'rb') as f:
            cur_test_res = pickle.load(f)
            print(cur_test_res[0])
            print("======================================")
            pass
        pass
