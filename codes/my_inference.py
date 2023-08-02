from easydict import EasyDict as edict
import argparse
import os
import numpy as np
import torch
from importlib import import_module
from utils.loss_utils import get_loss
from utils.ply import read_ply, write_ply
import pointnet_utils.pc_util as pc_util
from PIL import Image
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='Training/Testing SeedFormer', help='description')
parser.add_argument('--net_model', type=str, default='model', help='Import module.')
parser.add_argument('--arch_model', type=str, default='seedformer_dim128', help='Model to use.')
parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
parser.add_argument('--inference', dest='inference', help='Inference for benchmark', action='store_true')
parser.add_argument('--output', type=int, default=False, help='Output testing results.')
parser.add_argument('--pretrained', type=str, default='', help='Pretrained path for testing.')
args = parser.parse_args()

def PCNConfig():
    __C                                              = edict()
    cfg                                              = __C

    #
    # Dataset Config
    #
    __C.DATASETS                                     = edict()
    __C.DATASETS.COMPLETION3D                        = edict()
    __C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
    __C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
    __C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
    __C.DATASETS.SHAPENET                            = edict()
    __C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
    __C.DATASETS.SHAPENET.N_RENDERINGS               = 8
    __C.DATASETS.SHAPENET.N_POINTS                   = 2048
    __C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = 'data/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
    __C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = 'data/ShapeNetCompletion/%s/complete/%s/%s.pcd'

    #
    # Dataset
    #
    __C.DATASET                                      = edict()
    # Dataset Options: Completion3D, ShapeNet, ShapeNetCars, Completion3DPCCT
    __C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
    __C.DATASET.TEST_DATASET                         = 'ShapeNet'

    #
    # Constants
    #
    __C.CONST                                        = edict()

    __C.CONST.NUM_WORKERS                            = 1    # 8
    __C.CONST.N_INPUT_POINTS                         = 2048

    #
    # Directories
    #

    __C.DIR                                          = edict()
    __C.DIR.OUT_PATH                                 = '../results'
    __C.DIR.TEST_PATH                                = '../test'
    __C.CONST.DEVICE                                 = '0'   # 双卡则为0,1
    __C.CONST.WEIGHTS                                = 'results/train_pcn_Log_2023_07_13_08_37_17/checkpoints/ckpt-best.pth' # 'ckpt-best.pth'  # specify a path to run test and inference

    #
    # Network
    #
    __C.NETWORK                                      = edict()
    __C.NETWORK.UPSAMPLE_FACTORS                     = [1, 4, 8] # 16384

    #
    # Train
    #
    __C.TRAIN                                        = edict()
    __C.TRAIN.BATCH_SIZE                             = 4   # 48
    __C.TRAIN.N_EPOCHS                               = 100   # 400
    __C.TRAIN.SAVE_FREQ                              = 5    # 25
    __C.TRAIN.LEARNING_RATE                          = 0.0001   # 0.001
    __C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
    __C.TRAIN.LR_DECAY_STEP                          = 50
    __C.TRAIN.WARMUP_STEPS                           = 200
    __C.TRAIN.WARMUP_EPOCHS                          = 5   # 20
    __C.TRAIN.GAMMA                                  = .5
    __C.TRAIN.BETAS                                  = (.9, .999)
    __C.TRAIN.WEIGHT_DECAY                           = 0
    __C.TRAIN.LR_DECAY                               = 150

    #
    # Test
    #
    __C.TEST                                         = edict()
    __C.TEST.METRIC_NAME                             = 'ChamferDistance'


    return cfg

def read_ply(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    ptcloud = np.array(pc.points)
    return torch.from_numpy(ptcloud).to(torch.float32)

def test_net(cfg, input_path):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    #######################
    # Prepare Network Model
    #######################
    
    ## 添加GPU推理时间测试代码：
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    Model = import_module(args.net_model)
    model = Model.__dict__[args.arch_model](up_factors=cfg.NETWORK.UPSAMPLE_FACTORS)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model'])

    data = read_ply(input_path).unsqueeze(0)
    torch.backends.cudnn.benchmark = True

    # Switch models to evaluation mode
    model.eval()
    pcds_pred = model(data.contiguous())
    ender.record()
    torch.cuda.synchronize()
    duration_time = starter.elapsed_time(ender)
    print("1 batch inference time: {} ms".format(duration_time))
    pred = pcds_pred[-1]
    write_ply('codes/predict.ply', pred[0,:].detach().cpu().numpy(), ['x', 'y', 'z'])

if __name__ == '__main__':
    cfg = PCNConfig()
    test_net(cfg, "codes/data/ShapeNetCompletion/val/partial/02691156/4bae467a3dad502b90b1d6deb98feec6/00.pcd")










