import argparse
import logging
import time

import numpy as np
import torch.utils.data
import torch
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

logging.basicConfig(level=logging.INFO)
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true',
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()

    # if jacquard_output and dataset != 'jacquard':
    #     raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    # if jacquard_output and augment:
    #     raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()
    import os
    import shutil
    try:
        shutil.rmtree('results/')
    except:
        pass
    os.makedirs('results/', exist_ok=True)
    # Get the compute device
    # device = get_device(force_cpu)\
    device = torch.device('cuda:0')

    # Load Dataset
    dataset_path = 'outputs/'
    input_size = 224
    ds_rotate = False
    augment = False
    use_depth = True
    include_rgb= True
    split = 1.0
    ds_shuffle = False
    num_workers = 0
    random_seed = 0
    # logging.info('Loading {} Dataset...'.format(dataset.title()))
    Dataset = get_dataset(None)
    test_dataset = Dataset(dataset_path,
                           output_size=input_size,
                           ds_rotate=ds_rotate,
                           random_rotate=augment,
                           random_zoom=augment,
                           include_depth=use_depth,
                           include_rgb=include_rgb)

    indices = list(range(0,test_dataset.length, 10))
    split = int(np.floor(split * test_dataset.length))
    if ds_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # val_indices = indices[split:]
    val_indices = indices
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )
    logging.info('Done')
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # modelname = 'trained-models/jacquard-rgbd-grconvnet3-drop0-ch32/epoch_48_iou_0.93'
    modelname = 'trained-models/jacquard-d-grconvnet3-drop0-ch32/epoch_48_iou_0.93'
    modelname = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch/epoch_30_iou_0.97'
    modelname = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    net = torch.load(modelname)
    print('params = ', count_parameters(net))
    times = [] 
    with torch.no_grad():
        counts = 0
        for i in range(1):
            for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
                counts += 1
                print(idx)
                xc = x.to(device)
                yc = [yi.to(device) for yi in y]
                start_time = time.time()
                lossd = net.compute_loss(xc, yc)
                times.append(time.time() - start_time)

                q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                                lossd['pred']['sin'], lossd['pred']['width'])

                
                # if jacquard_output:
                #     grasps = grasp.detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
                #     with open(jo_fn, 'a') as f:
                #         for g in grasps:
                #             f.write(test_data.dataset.get_jname(didx) + '\n')
                #             f.write(g.to_jacquard(scale=1024 / 300) + '\n')

                avg_time = np.mean(times)
                logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))
                save_results(
                    rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False).astype(np.uint8),
                    depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=5,
                    grasp_width_img=width_img,
                    idx = idx
                )
            # break



    del net
    torch.cuda.empty_cache()
