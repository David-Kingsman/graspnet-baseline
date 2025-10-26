""" Training routine for GraspNet baseline model. """

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from graspnet import GraspNet, get_loss
from pytorch_utils import BNMomentumScheduler
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from label_generation import process_grasp_labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers [default: 4]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

# 训练时间统计
TRAINING_START_TIME = None
EPOCH_TIMES = []
BATCH_TIMES = []

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=True, augment=True)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def train_one_epoch():
    global BATCH_TIMES
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    net.train()
    
    # 创建进度条
    pbar = tqdm(enumerate(TRAIN_DATALOADER), 
                total=len(TRAIN_DATALOADER), 
                desc=f"Epoch {EPOCH_CNT+1}/{cfgs.max_epoch}",
                ncols=120,
                leave=True)
    
    epoch_start_time = time.time()
    
    for batch_idx, batch_data_label in pbar:
        batch_start_time = time.time()
        
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        end_points = net(batch_data_label)

        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss(end_points)
        loss.backward()
        if (batch_idx+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        # 计算时间统计
        batch_time = time.time() - batch_start_time
        BATCH_TIMES.append(batch_time)
        if len(BATCH_TIMES) > 100:  # 只保留最近100个batch的时间
            BATCH_TIMES.pop(0)
        
        # 计算剩余时间
        avg_batch_time = np.mean(BATCH_TIMES) if BATCH_TIMES else 0
        remaining_batches = len(TRAIN_DATALOADER) - batch_idx - 1
        remaining_time = remaining_batches * avg_batch_time
        
        # 计算当前epoch进度
        epoch_progress = (batch_idx + 1) / len(TRAIN_DATALOADER) * 100
        
        # 更新进度条显示
        current_lr = get_current_lr(EPOCH_CNT)
        overall_loss = stat_dict.get('loss/overall_loss', 0) / max(1, (batch_idx + 1))
        
        pbar.set_postfix({
            'Loss': f'{overall_loss:.4f}',
            'LR': f'{current_lr:.2e}',
            'ETA': format_time(remaining_time),
            'Batch/s': f'{1/avg_batch_time:.2f}' if avg_batch_time > 0 else '0.00'
        })

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
    
    pbar.close()
    
    # 记录epoch时间
    epoch_time = time.time() - epoch_start_time
    EPOCH_TIMES.append(epoch_time)
    if len(EPOCH_TIMES) > 5:  # 只保留最近5个epoch的时间
        EPOCH_TIMES.pop(0)

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    
    # 创建评估进度条
    pbar = tqdm(enumerate(TEST_DATALOADER), 
                total=len(TEST_DATALOADER), 
                desc="Evaluating",
                ncols=120,
                leave=False)
    
    for batch_idx, batch_data_label in pbar:
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        
        # 更新进度条
        if batch_idx > 0:
            current_loss = stat_dict.get('loss/overall_loss', 0) / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

    pbar.close()

    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT, TRAINING_START_TIME
    min_loss = 1e10
    loss = 0
    
    # 记录训练开始时间
    TRAINING_START_TIME = time.time()
    
    print(f"\n🚀 开始训练 GraspNet 模型")
    print(f"📊 训练配置:")
    print(f"   - 总epoch数: {cfgs.max_epoch}")
    print(f"   - 批次大小: {cfgs.batch_size}")
    print(f"   - 学习率: {cfgs.learning_rate}")
    print(f"   - 数据集: {cfgs.camera}")
    print(f"   - 开始epoch: {start_epoch}")
    print("=" * 80)
    
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        
        # 计算总体进度
        total_epochs = cfgs.max_epoch - start_epoch
        current_epoch = epoch - start_epoch + 1
        overall_progress = (current_epoch - 1) / total_epochs * 100
        
        # 计算预计完成时间
        if EPOCH_TIMES:
            avg_epoch_time = np.mean(EPOCH_TIMES)
            remaining_epochs = cfgs.max_epoch - epoch
            estimated_remaining_time = remaining_epochs * avg_epoch_time
        else:
            estimated_remaining_time = 0
        
        # 计算已用时间
        elapsed_time = time.time() - TRAINING_START_TIME
        
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        
        print(f"\n📈 Epoch {epoch+1}/{cfgs.max_epoch} - 总体进度: {overall_progress:.1f}%")
        print(f"⏱️  已用时间: {format_time(elapsed_time)}")
        if estimated_remaining_time > 0:
            print(f"⏳ 预计剩余: {format_time(estimated_remaining_time)}")
        
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        
        # 训练一个epoch
        train_one_epoch()
        
        # 评估
        print(f"\n🔍 开始评估...")
        loss = evaluate_one_epoch()
        
        # 保存checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))
        
        # 显示epoch总结
        epoch_time = EPOCH_TIMES[-1] if EPOCH_TIMES else 0
        print(f"\n✅ Epoch {epoch+1} 完成!")
        print(f"   - 训练时间: {format_time(epoch_time)}")
        print(f"   - 验证损失: {loss:.4f}")
        print(f"   - 最佳损失: {min_loss:.4f}")
        
        if loss < min_loss:
            min_loss = loss
            print(f"   🎉 新的最佳损失! 已保存模型")
        
        print("=" * 80)
    
    # 训练完成
    total_time = time.time() - TRAINING_START_TIME
    print(f"\n🎉 训练完成!")
    print(f"⏱️  总训练时间: {format_time(total_time)}")
    print(f"📊 最终损失: {loss:.4f}")
    print(f"💾 模型已保存到: {cfgs.log_dir}/checkpoint.tar")

if __name__=='__main__':
    train(start_epoch)
