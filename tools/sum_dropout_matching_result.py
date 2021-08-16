import os
import pickle
import argparse
import numpy as np
import copy
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def grab_all_pkl(test_log_dir):
    sub_dirs = [i for i in os.listdir(test_log_dir) if os.path.isdir(os.path.join(test_log_dir, i))]
    result_pkl_file_list = []
    for i in sub_dirs:
        result_file = os.listdir(os.path.join(test_log_dir, i))
        if "gt_dt_matching_res.pkl" in result_file:
            result_pkl_file_list.append(os.path.join(test_log_dir, i, "gt_dt_matching_res.pkl"))
    return result_pkl_file_list

def compress_test_result(res):
    res.pop('truncated')
    res.pop('occluded')
    pass

def get_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def dump_pkl(pkl_file, obj):
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    log_file = "tmp.txt"
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    log_dir = "../output/kitti_models/pv_rcnn/default/eval_dropout/epoch_8369/val/default"
    pkl_list = grab_all_pkl(log_dir)

    _, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    cnt = 0
    # save result by frm! notation: io with high frequency...
    gt_annos = [copy.deepcopy(info['annos']) for info in test_loader.dataset.kitti_infos]
    results_dir = "/home/xlju/Project/OpenPCDet/drop_out_test_results"
    frm_dict_list = []
    for idx, frm_gt in enumerate(gt_annos):
        frm_dict = {"gt": frm_gt,
                    'dropout_results': [],
                    'gt_dt_corrspondence': {
                        'camera': [],
                        'bev': [],
                        '3d': []
                    }
                    }
        frm_dict_list.append(frm_dict)

    for i in pkl_list:
        print("processing idx = ", cnt, "pickle_name = ", i)
        cur_test_res = None
        cur_matching_res = None
        with open(i, 'rb') as f:
            cur_matching_res = pickle.load(f)
        cur_test_res_file = os.path.join("/".join(i.split('/')[:-1]), "result.pkl")
        with open(cur_test_res_file, 'rb') as f:
            cur_test_res = pickle.load(f)
        pass
        for k, v in cur_matching_res.items():
            assert len(v) == len(cur_test_res) == len(gt_annos)
        for idx, frm_tst in enumerate(cur_test_res):
            frm_dict_list[idx]['dropout_results'].append(frm_tst)
        for idx, frm_tst in enumerate(cur_matching_res['camera']):
            frm_dict_list[idx]['gt_dt_corrspondence']['camera'].append(frm_tst)
        for idx, frm_tst in enumerate(cur_matching_res['bev']):
            frm_dict_list[idx]['gt_dt_corrspondence']['bev'].append(frm_tst)
        for idx, frm_tst in enumerate(cur_matching_res['3d']):
            frm_dict_list[idx]['gt_dt_corrspondence']['3d'].append(frm_tst)
        cnt += 1
    print("saving dataset..")
    for frm in frm_dict_list:
        kitti_frm_id_str = frm['dropout_results'][0]['frame_id']
        pkl_file = os.path.join(results_dir, kitti_frm_id_str + ".pkl")
        print("saving:", pkl_file)
        dump_pkl(pkl_file, frm)
