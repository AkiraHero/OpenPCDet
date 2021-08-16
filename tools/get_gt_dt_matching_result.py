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
        if "result.pkl" in result_file:
            result_pkl_file_list.append(os.path.join(test_log_dir, i, "result.pkl"))
    return result_pkl_file_list


def get_matching_list(det_annos, gt_annos, overlap_metric, overlap_thres, valid_classes):
    num_parts = 100
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    rets = kitti_eval.calculate_iou_partly(det_annos, gt_annos, overlap_metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    # have to select gt data with designated classes
    dt_correspondence_list = []
    for frm_idx, frm in enumerate(gt_annos):
        frm_overlap = overlaps[frm_idx]
        dt_correspondence = np.array([-1] * total_gt_num[frm_idx])
        for gt_idx in range(total_gt_num[frm_idx]):
            cur_gt_class = frm['name'][gt_idx]
            if cur_gt_class not in valid_classes:
                continue
            max_overlap = -1
            dt_idx = -1
            for det_idx, iou in enumerate(frm_overlap[:, gt_idx]):
                if det_annos[frm_idx]['name'][det_idx] == cur_gt_class:
                    if iou > overlap_thres[overlap_metric, name_to_class[cur_gt_class]]:
                        if iou > max_overlap:
                            max_overlap = iou
                            dt_idx = det_idx
            if max_overlap != -1:
                dt_correspondence[gt_idx] = dt_idx
        dt_correspondence_list.append(dt_correspondence)
    return dt_correspondence_list

def get_matching_dict(model_out, data_loader):
    dataset = data_loader.dataset
    # todo: why deep copy?
    det_annos = copy.deepcopy(model_out)
    gt_annos = [copy.deepcopy(info['annos']) for info in data_loader.dataset.kitti_infos]
    assert len(gt_annos) == len(det_annos)
    class_names = dataset.class_names


    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7,
                             0.5, 0.7], [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7]])
    # overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7,
    #                          0.5, 0.5], [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
    #                         [0.5, 0.25, 0.25, 0.5, 0.25, 0.5]])

    matching_dict = {}
    min_overlaps = overlap_0_7
    matching_dict['camera'] = get_matching_list(det_annos, gt_annos, 0, min_overlaps, class_names)
    matching_dict['bev'] = get_matching_list(det_annos, gt_annos, 1, min_overlaps, class_names)
    matching_dict['3d'] = get_matching_list(det_annos, gt_annos, 2, min_overlaps, class_names)
    return matching_dict


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
    for i in pkl_list:
        print("processing idx = ", cnt, "pickle_name = ", i)
        with open(i, 'rb') as f:
            cur_test_res = pickle.load(f)
            matching_dict = get_matching_dict(cur_test_res, test_loader)
            matching_dict_path = os.path.join("/".join(i.split('/')[:-1]), "gt_dt_matching_res.pkl")
            with open(matching_dict_path, 'wb') as f_out:
                pickle.dump(matching_dict, f_out)
        cnt += 1
