import argparse
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from models.difftr_flow3d import DiffTR_Flow3D
import torch
import numpy as np
from tqdm.auto import tqdm
import math
import copy
from pathlib import Path
from torch.optim import AdamW
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torchvision import transforms as utils
from datasetloader.datasets import build_train_dataset, build_test_dataset

from utils import compute_epe2

# import wandb
import torch.distributed as dist
import random
import os
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion_model,
        train_batch_size = 24,
        test_batch_size = 24,
        gradient_accumulate_every = 1,
        train_lr = 4e-4,
        train_num_epochs = 50,
        save_and_sample_every = 1000,
        results_folder = './results',
        checkpoint_dir = './checkpoints',
        max_grad_norm = 1.,
        dataset = 'f3d_occ',
        resume = None,
    ):
        super().__init__()

        self.args = args
        # model
        self.model = diffusion_model # diffusion model
        self.model_without_ddp = copy.deepcopy(self.model)
        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            self.model.model = torch.nn.DataParallel(self.model.model)
            self.model_without_ddp.model = self.model.model.module
        else:
            self.model_without_ddp.model = self.model.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every # 1000
        self.train_batch_size = train_batch_size # 24
        self.test_batch_size = test_batch_size # 24
        self.gradient_accumulate_every = gradient_accumulate_every # 1
        self.train_num_epochs = train_num_epochs # 600000
        self.max_grad_norm = max_grad_norm
        self.dataset = dataset

        # load training datasets
        self.train_dataset = build_train_dataset(self.dataset)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                                                        shuffle=True, num_workers=2,
                                                        pin_memory=True, drop_last=True,
                                                        sampler=None)
        # load datasets
        self.val_dataset = build_test_dataset(self.dataset)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.test_batch_size,
                                                        shuffle=False, num_workers=2,
                                                        pin_memory=True, drop_last=False,
                                                        sampler=None)

        # result and checkpoint folders
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok = True)

        # optimizer
        self.opt = AdamW(self.model_without_ddp.parameters(), lr=train_lr, weight_decay=1e-4)

        # epoch counter state
        self.epoch = 0

        if resume is not None:
            self.load(resume)
            print('start_epoch: %d' % (self.epoch))
        last_epoch = self.epoch if resume is not None and self.epoch > 0 else -1

        # lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, train_lr,
            train_num_epochs * len(self.train_loader),
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=last_epoch,
        )

        self.best_epe = 10


    @property
    def save(self, milestone):

        data = {
            'epoch': self.epoch,
            'model': self.get_state_dict(self.model_without_ddp),
            'opt': self.opt.state_dict(),
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, resume):
        device = self.device
        data = torch.load(str(self.results_folder / resume), map_location=device)
        self.model_without_ddp.load_state_dict(data['model'], strict = True)
        self.epoch = data['epoch']
        self.opt.load_state_dict(data['opt'])

        if 'version' in data:
            print(f"loading from version {data['version']}")


    def train(self):
        device = self.device

        while self.epoch < self.train_num_epochs:
            model.train()
            metrics_3d = {}
            loss_train = []
            train_progress = tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9) # ncols=150
            for i, sample in enumerate(train_progress):
                pcs = sample['pcs'].to(device)
                pcs = torch.permute(pcs, (0, 2, 1))
                flow_set = sample['flow_3d'].to(device)
                flow_set = torch.permute(flow_set, (0, 2, 1))

                total_loss = 0.
                loss, metrics_3d = self.model(flow_set, pcs)
                if isinstance(loss, float):
                    continue
                if torch.isnan(loss):
                    continue
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()
                metrics_3d.update({'total_loss': total_loss})
                # more efficient zero_grad
                for param in self.model_without_ddp.parameters():
                    param.grad = None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.lr_scheduler.step()

                loss_train.append(loss.detach().cpu())
                train_progress.set_description(
                    'Train Epoch {}: Loss: {:.5f}'.format(
                        self.epoch,
                        np.array(loss_train).mean(),
                    )
                )

            with (self.results_folder  / f"log.txt").open("a") as f:
                f.write(json.dumps({'Train Epoch': self.epoch, 'Loss': float(np.array(loss_train).mean()), 
                                    'Loss': float(np.array(loss_train).mean())}) + "\n")

            if self.epoch % 10 == 0:
                print('Save last checkpoint at epoch: %d' % self.epoch)
                checkpoint_path = str(self.results_folder / 'model_last.pt')
                torch.save({
                    'model': self.model_without_ddp.state_dict(),
                    'opt': self.opt.state_dict(),
                    'epoch': self.epoch,
                }, checkpoint_path)

            if (self.epoch+1) % 10 == 0:
                result = self.val()
                if self.best_epe > result['EPE']:
                    self.best_epe = result['EPE']
                    print('Save best checkpoint at epoch: %d' % self.epoch)
                    checkpoint_path = str(self.results_folder / 'model_best.pt')
                    torch.save({
                        'model': self.model_without_ddp.state_dict(),
                        'opt': self.opt.state_dict(),
                        'epoch': self.epoch,
                    }, checkpoint_path)

                # if self.args.wandb and get_rank() == 0: 
                #     wandb.log({
                #         'Loss': np.array(loss_train).mean(),
                #         'EPE': result['EPE'],
                #         'Outlier': result['Outlier'],
                #         'Acc3dRelax': result['Acc3dRelax'],
                #         'Acc3dStrict': result['Acc3dStrict'],
                #     })

            self.epoch += 1

    def val(self):
        device = self.device
        self.model.eval()
        
        with torch.inference_mode():
            results = {}
            epe_run = []
            outlier_run = []
            acc3dRelax_run = []
            acc3dStrict_run = []

            val_progress = tqdm(self.val_loader, total=len(self.val_loader), smoothing=0.9) # ncols=150
            for i, data in enumerate(val_progress):
                pcs = data['pcs'].to(device) # 6*8192
                pcs = torch.permute(pcs, (0, 1, 3, 2))
                flow_3d = data['flow_3d'].to(device) # B 3 N
                flow_3d = torch.permute(flow_3d, (0, 1, 3, 2))
                flow_3d_mask = torch.ones(flow_3d[:,:,0,:].shape, dtype=torch.int64).cuda()

                flow_pred = []
                flow_3d_ = []
                interval = 1
                start_points = pcs[:, 0]
                points_list = [start_points]
                for i in range(interval, pcs.shape[1], interval):
                    if i == interval:
                        pos_center = torch.mean(start_points, dim=-1, keepdim=True)
                        pred = self.model_without_ddp.sample(torch.cat([start_points-pos_center, pcs[:, i]-pos_center], dim=-2), return_all_timesteps = False)
                    else:
                        if i == 2*interval:
                            pflow = flow_pred[-1]
                        else:
                            next_points = self.polyfit(torch.stack(points_list, dim=1).permute(0, 1, 3, 2).cpu().numpy())
                            next_points = torch.tensor(next_points, dtype=start_points.dtype, device=start_points.device).permute(0, 2, 1)
                            pflow = next_points - start_points
                        pos_center = torch.mean(start_points+pflow, dim=-1, keepdim=True)
                        pred = self.model_without_ddp.sample(torch.cat([start_points+pflow-pos_center, pcs[:, i]-pos_center], dim=-2), return_all_timesteps = False) + pflow

                    start_points = start_points + pred
                    flow_pred.append(pred)
                    points_list.append(start_points)
                    flow_3d_.append(flow_3d[:, i-interval:i].sum(dim=1)) # 之前的flow相加
                flow_pred = torch.stack(flow_pred, dim=1)
                flow_3d = torch.stack(flow_3d_, dim=1)
                points_list = torch.stack(points_list, dim=1)
                flow_3d_mask = torch.ones(flow_3d[:,:,0,:].shape, dtype=torch.int64).cuda()

                epe, acc3d_strict, acc3d_relax, outlier = compute_epe2(torch.permute(flow_3d, (0, 1, 3, 2)).cpu().numpy(), 
                                                                       torch.permute(flow_pred, (0, 1, 3, 2)).cpu().numpy(), 
                                                                       flow_3d_mask.cpu().numpy())
                epe_run.append(epe)
                outlier_run.append(outlier)
                acc3dRelax_run.append(acc3d_relax)
                acc3dStrict_run.append(acc3d_strict)

                val_progress.set_description(
                    ' Epoch {}: EPE: {:.5f} Acc3dStrict: {:.5f} Acc3dRelax: {:.5f} Outlier: {:.5f}'.format(
                        self.epoch,
                        np.array(epe_run).mean(),
                        np.array(acc3dStrict_run).mean(),
                        np.array(acc3dRelax_run).mean(),
                        np.array(outlier_run).mean(),
                    )
                )

            results['EPE'] = float(np.array(epe_run).mean())
            results['Outlier'] = float(np.array(outlier_run).mean() * 100.0)
            results['Acc3dRelax'] = float(np.array(acc3dRelax_run).mean() * 100.0)
            results['Acc3dStrict'] = float(np.array(acc3dStrict_run).mean() * 100.0)
            print("Validation Things EPE: %.6f, Acc3dStrict: %.6f, Acc3dRelax: %.4f, Outlier: %.4f" % (results['EPE'], results['Acc3dStrict'], results['Acc3dRelax'], results['Outlier']))
            with (self.results_folder  / f"log.txt").open("a") as f:
                f.write(json.dumps(results) + "\n")

        return results

    def polyfit(self, data, degree=2, alpha=10.0):
        predicted_next_frame = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
        num_frames = min(data.shape[1], 3)

        poly_features = PolynomialFeatures(degree=degree)
        ridge_reg = Ridge(alpha=alpha)

        for b in range(data.shape[0]):
            for i in range(data.shape[3]):
                t = np.arange(num_frames)
                X = poly_features.fit_transform(t.reshape(-1, 1))
                y = data[b, -num_frames:, :, i]

                ridge_reg.fit(X, y)

                next_t = np.array([num_frames]).reshape(1, -1)
                predicted_next_frame[b, :, i] = ridge_reg.predict(poly_features.transform(next_t))[0]

        return predicted_next_frame

class Tester(object):
    def __init__(
        self,
        diffusion_model,
        test_batch_size = 12,
        resume = None,
        results_folder = './results',
        dataset = 'kitti_occ'
    ):
        super().__init__()

        # model
        self.model = diffusion_model # diffusion model
        self.model_without_ddp = copy.deepcopy(self.model)
        
        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            self.model.model = torch.nn.DataParallel(self.model.model)
            self.model_without_ddp.model = self.model.model.module
        else:
            self.model_without_ddp.model = self.model.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sampling and testing hyperparameters
        self.test_batch_size = test_batch_size
        self.resume = resume
        self.dataset = dataset

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)

    @property
    def load(self, milestone):
        device = self.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        self.model_without_ddp.load_state_dict(data['model'], strict = True)
        self.epoch = data['epoch']

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def polyfit(self, data, degree=2, alpha=10.0):
        predicted_next_frame = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
        num_frames = min(data.shape[1], 3)

        poly_features = PolynomialFeatures(degree=degree)
        ridge_reg = Ridge(alpha=alpha)

        for b in range(data.shape[0]):
            for i in range(data.shape[3]):
                t = np.arange(num_frames)
                X = poly_features.fit_transform(t.reshape(-1, 1))
                y = data[b, -num_frames:, :, i]

                ridge_reg.fit(X, y)

                next_t = np.array([num_frames]).reshape(1, -1)
                predicted_next_frame[b, :, i] = ridge_reg.predict(poly_features.transform(next_t))[0]

        return predicted_next_frame

    def test(self):
        device = self.device
        self.model_without_ddp.eval()
        
        with torch.inference_mode():
            data = torch.load(str(self.results_folder / self.resume), map_location=device)
            self.model_without_ddp.load_state_dict(data['model'], strict = True)
            self.epoch = data['epoch']

            # load datasets
            self.val_dataset = build_test_dataset(self.dataset)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.test_batch_size,
                                                            shuffle=False, num_workers=2,
                                                            pin_memory=True, drop_last=False,
                                                            sampler=None)
            results = {}
            metrics_3d = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}
            epe_run = []
            outlier_run = []
            acc3dRelax_run = []
            acc3dStrict_run = []
            multi_metircs_run = [{'EPE': [], 'ACC_Relax': [], 'ACC_Strict': [], 'outlier': []} for i in range(9)]

            val_progress = tqdm(self.val_loader, total=len(self.val_loader), smoothing=0.9) # ncols=150
            for i, data in enumerate(val_progress):
                flow_preds = []
                pcs = data['pcs'].to(device) # 6*8192
                pcs = torch.permute(pcs, (0, 1, 3, 2))
                flow_3d = data['flow_3d'].to(device) # B 3 N
                flow_3d = torch.permute(flow_3d, (0, 1, 3, 2))
                flow_3d_mask = torch.ones(flow_3d[:,:,0,:].shape, dtype=torch.int64).cuda()

                flow_pred = []
                flow_3d_ = []
                interval = 1
                start_points = pcs[:, 0]
                points_list = [start_points]
                for i in range(interval, pcs.shape[1], interval):
                    if i == interval:
                        pos_center = torch.mean(start_points, dim=-1, keepdim=True)
                        pred = self.model_without_ddp.sample(torch.cat([start_points-pos_center, pcs[:, i]-pos_center], dim=-2), return_all_timesteps = False)
                    else:
                        if i == 2*interval:
                            pflow = flow_pred[-1]
                        else:
                            next_points = self.polyfit(torch.stack(points_list, dim=1).permute(0, 1, 3, 2).cpu().numpy())
                            next_points = torch.tensor(next_points, dtype=start_points.dtype, device=start_points.device).permute(0, 2, 1)
                            pflow = next_points - start_points
                        pos_center = torch.mean(start_points+pflow, dim=-1, keepdim=True)
                        pred = self.model_without_ddp.sample(torch.cat([start_points+pflow-pos_center, pcs[:, i]-pos_center], dim=-2), return_all_timesteps = False) + pflow

                    start_points = start_points + pred
                    flow_pred.append(pred)
                    points_list.append(start_points)
                    flow_3d_.append(flow_3d[:, i-interval:i].sum(dim=1)) # 之前的flow相加
                flow_pred = torch.stack(flow_pred, dim=1)
                flow_3d = torch.stack(flow_3d_, dim=1)
                points_list = torch.stack(points_list, dim=1)
                flow_3d_mask = torch.ones(flow_3d[:,:,0,:].shape, dtype=torch.int64).cuda()

                flow_3d_zeros = torch.zeros_like(flow_3d_mask, dtype=torch.int64).to(device)
                epe, acc3d_strict, acc3d_relax, outlier = compute_epe2(torch.permute(flow_3d, (0, 1, 3, 2)).cpu().numpy(), 
                                                                       torch.permute(flow_pred, (0, 1, 3, 2)).cpu().numpy(), 
                                                                       flow_3d_mask.cpu().numpy())
                epe_run.append(epe)
                outlier_run.append(outlier)
                acc3dRelax_run.append(acc3d_relax)
                acc3dStrict_run.append(acc3d_strict)

                val_progress.set_description(
                    ' Epoch {}: EPE: {:.5f} Acc3dStrict: {:.5f} Acc3dRelax: {:.5f} Outlier: {:.5f}'.format(
                        self.epoch,
                        np.array(epe_run).mean(),
                        np.array(acc3dStrict_run).mean(),
                        np.array(acc3dRelax_run).mean(),
                        np.array(outlier_run).mean(),
                    )
                )

            results['EPE'] = np.array(epe_run).mean()
            results['Outlier'] = np.array(outlier_run).mean() * 100.0
            results['Acc3dRelax'] = np.array(acc3dRelax_run).mean() * 100.0
            results['Acc3dStrict'] = np.array(acc3dStrict_run).mean() * 100.0
            print("Validation Things EPE: %.6f, Acc3dStrict: %.6f, Acc3dRelax: %.4f, Outlier: %.4f" % (results['EPE'], results['Acc3dStrict'], results['Acc3dRelax'], results['Outlier']))


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='where to save the training log and models')
    parser.add_argument('--result_dir', default='./results', type=str, help='where to save the training log and models')
    parser.add_argument('--train_dataset', default='f3d_occ', type=str, help='training dataset on different datasets')
    parser.add_argument('--val_dataset', default='f3d_occ', type=str, help='validation dataset on different datasets ')
    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--train_batch_size', default=24, type=int)
    parser.add_argument('--test_batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=326, type=int)
    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    # model: learnable parameters
    parser.add_argument('--timesteps', default=20, type=int)
    parser.add_argument('--samplingtimesteps', default=2, type=int)
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--backbone', default='DGCNN', type=str, help='feature extraction backbone (DGCNN / PointNet)')
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_pt_layers', default=1, type=int)
    parser.add_argument('--num_transformer_layers', default=14, type=int)
    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')
    # log
    parser.add_argument('--summary_freq', default=500, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--test_epoch', default=600, type=int)

    # # * WanDB
    # parser.add_argument('--wandb', action='store_true')
    # parser.add_argument('--project_name', default='climate')
    # parser.add_argument('--group_name', default='economics')
    # parser.add_argument('--run_name', default='test')
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    model = DiffTR_Flow3D(backbone=args.backbone,
                channels=args.feature_channels,
                ffn_dim_expansion=args.ffn_dim_expansion,
                num_transformer_pt_layers=args.num_transformer_pt_layers,
                num_transformer_layers=args.num_transformer_layers).to(device)
    # diffusion
    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_x0', # pred_x0
        beta_schedule = 'cosine',  # sigmoid, cosine, linear
        timesteps = args.timesteps,           # number of steps
        sampling_timesteps = args.samplingtimesteps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ).to(device)

    # # add argparse
    # if args.wandb and get_rank() == 0:
    #     wandb.init(
    #         project=args.project_name,
    #         group=args.group_name,
    #         name=args.run_name,
    #         config=args
    #     )
    #     wandb.watch(diffusion)

    # test
    if args.eval:
        tester = Tester(
            diffusion,
            test_batch_size = args.test_batch_size,
            resume = args.resume,
            results_folder = args.result_dir,
            dataset = args.val_dataset
        )
        tester.test()

    # train
    else:
        trainer = Trainer(
            args,
            diffusion,
            train_batch_size = args.train_batch_size,
            test_batch_size = args.test_batch_size,
            train_lr = args.lr,
            train_num_epochs = args.num_epochs,
            results_folder = args.result_dir,
            checkpoint_dir = args.checkpoint_dir,
            dataset = args.train_dataset,
            resume = args.resume,
        )
        trainer.train()




