import os
import shutil
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import pandas as pd
from toolbox.loss import MscCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from toolbox import get_dataset, get_logger, get_model
from toolbox import averageMeter, runningScore
from toolbox.losses import LovaszSoftmax, MSE
from thop import profile
import torch.backends.cudnn as cudnn
from fvcore.nn import FlopCountAnalysis

torch.manual_seed(123)
cudnn.benchmark = True

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, smooth=1e-5, dice_weight=0.5, ce_weight=0.5):
        """
        dice_weight: weight for dice loss component
        ce_weight: weight for cross entropy loss component
        """
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # -------- CrossEntropy Loss --------
        ce = self.ce_loss(inputs, targets)

        # -------- Dice Loss --------
        # One-hot encode targets
        num_classes = inputs.shape[1]
        targets_onehot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)

        # Get probabilities via softmax
        probs = F.softmax(inputs, dim=1)

        # Calculate Dice score
        dims = (0, 2, 3)  # Sum over batch and spatial dimensions
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = 1 - dice_score.mean()

        # -------- Total Loss --------
        return self.dice_weight * dice_loss + self.ce_weight * ce

def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Get row index and iterate
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text  # Return dictionary (key: values)

def count_trainable_flops(model, img_size=640):
    # 1. Prepare dummy inputs
    device = next(model.parameters()).device
    rgb = torch.randn(1, 3, img_size, img_size).to(device)
    dep = torch.randn(1, 3, img_size, img_size).to(device)
    text_token = torch.randint(0, 49408, (1, 77)).to(device)
    text_mask = torch.ones((1, 77)).to(device)
    inputs = (rgb, dep, text_token, text_mask)

    # 2. Initialize FlopCountAnalysis with error ignoring
    flops = FlopCountAnalysis(model, inputs)
    
    # Ignore warnings for unsupported ops, uncalled modules, custom modules
    flops.unsupported_ops_warnings(False) 
    flops.uncalled_modules_warnings(False)
    
    # 3. Filter module names to skip (including Embeddings, CrossAttn related)
    skip_modules = ["Embeddings", "_emb_rgb", "_emb_t", "cross_attn"]

    # 4. Get FLOPs dictionary for all modules
    flop_counts = flops.by_module()

    trainable_flops = 0
    total_flops = 0

    # 5. Iterate through all submodules, filter skipped modules and count
    for name, module in model.named_modules():
        # Skip specified modules + non-leaf modules + uncounted modules
        if any(skip in name for skip in skip_modules):
            continue
        if name not in flop_counts or len(list(module.children())) > 0:
            continue
        
        current_module_flops = flop_counts[name]
        total_flops += current_module_flops

        # Check for trainable parameters
        params = list(module.parameters(recurse=False))
        if params and any(p.requires_grad for p in params):
            trainable_flops += current_module_flops
    
    # 6. Print results
    print("-" * 40)
    print(f"Total FLOPs (Frozen + Trainable, skipped incompatible modules): {total_flops / 1e9:.2f} G")
    print(f"Trainable FLOPs (Only Unfrozen, skipped incompatible modules):  {trainable_flops / 1e9:.2f} G")
    print(f"Trainable Ratio: {trainable_flops / total_flops * 100:.2f}%" if total_flops > 0 else "Trainable Ratio: N/A")
    print("-" * 40)
    
    return trainable_flops

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{cfg["exp_name"] + "_" + time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'
    os.makedirs(logdir, exist_ok=True)
    # File to record validation metrics per epoch
    epoch_metrics_path = os.path.join(logdir, "epochval.txt")
    # Write simple header if file doesn't exist (optional)
    if not os.path.exists(epoch_metrics_path):
        with open(epoch_metrics_path, "w") as f:
            f.write("Validation metrics per epoch\n\n")
    
    try:
        shutil.copy('/data1/Code/zhaoxiaowei/CAILA-main/toolbox/models/cainet.py', os.path.join(logdir, 'cainet.py'))
        shutil.copy('/data1/Code/zhaoxiaowei/CAILA-main/sam2/modeling/backbones/hieradet.py', os.path.join(logdir, 'hieradet.py'))
        shutil.copy('/data1/Code/zhaoxiaowei/CAILA-main/train_pst900.py', os.path.join(logdir, 'train_pst900.py'))
        shutil.copy('/data1/Code/zhaoxiaowei/CAILA-main/configs/pst900.json', os.path.join(logdir, 'pst900.json'))
    except Exception as e:
        print(f"Failed to copy config files: {e}")

    shutil.copy(args.config, logdir)
    logger = get_logger(logdir)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(cfg)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # -------------------------------------------------------------------------
    # [Insert Start] Calculate parameters and FLOPs
    # -------------------------------------------------------------------------
    try:
        # Get image size, priority from cfg, default 640 if not exists
        # Assume cfg has image_h / image_w or img_size
        img_s = cfg.get('image_h', 640) 
        
        # Call the defined function directly
        count_trainable_flops(model, img_size=img_s)
        
    except Exception as e:
        print(f"Failed to calculate FLOPs: {e}")
    
    try:
        print("Calculating model parameters and FLOPs ...")
        
        # 1. Calculate trainable parameters
        # Filter out parameters with requires_grad=False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")

        # Prepare dummy input for FLOPs calculation
        h, w = cfg.get('image_h', 640), cfg.get('image_w', 640)  
        dummy_image = torch.randn(1, 3, h, w).to(device)
        
        # Prepare input arguments based on input type
        if cfg['inputs'] != 'rgb':
            dummy_depth = torch.randn(1, 3, h, w).to(device)
            dummy_text_token = torch.randint(0, 1000, (1, 77)).to(device) 
            dummy_text_mask = torch.randint(0, 2, (1, 77)).to(device)
            # thop.profile accepts inputs as a tuple, which will be unpacked to model.forward
            # Note: model.forward signature is (image, depth, text_token, text_mask)
            input_args = (dummy_image, dummy_depth, dummy_text_token, dummy_text_mask)
        else:
            input_args = (dummy_image, )

        # Calculate FLOPs using thop
        # custom_ops can be used to ignore unsupported custom layers or define custom calculation rules
        macs, _ = profile(model, inputs=input_args, verbose=False)
        
        # thop returns MACs (Multiply-Accumulate Operations), usually FLOPs ≈ 2 * MACs
        print(f"FLOPs (G): {macs * 2 / 1e9:.2f} G")
        print(f"MACs (G): {macs / 1e9:.2f} G")
        
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
    
    print("-" * 30)
 
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model_dict = model.state_dict()
        backbone_dict = {k: v for k, v in checkpoint.items() if k in model_dict and not k.startswith('classifier')}
        model_dict.update(backbone_dict)
        model.load_state_dict(model_dict)
    
    if args.dataset == 'pst900':
        train_text = read_text("/root fo /text/" + 'Train_newtext.xlsx')
        val_text = read_text("/root fo/text/" + 'Test_newtext.xlsx')
 
    trainset, *testset = get_dataset(cfg, train_text, val_text)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(testset[-1], batch_size=cfg['ims_per_gpu'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    criterion_0 = DiceCELoss(dice_weight=0.5, ce_weight=0.5).to(device)

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'])

    maxmiou = 0.0
    best_epoch = 0

    for ep in range(cfg['epochs']):
        model.train()
        train_loss_meter.reset()

        train_pbar = tqdm(train_loader, ncols=120, leave=False, desc=f"Train | Epoch [{ep + 1}/{cfg['epochs']}]")
        for sample in train_pbar:
            optimizer.zero_grad()

            image = sample['image'].to(device)
            label = sample['label'].to(device)
            text_token = sample['text_token'].to(device)
            text_mask = sample['txt_mask'].to(device)
            depth = sample.get('depth', None)
            depth = depth.to(device) if depth is not None else None

            if cfg['inputs'] == 'rgb':
                out = model(image)
            else:
                out, loss_dic = model(image, depth, text_token, text_mask)
 
            lad_loss = loss_dic['lad_loss']
            loss = criterion_0(out, label) + 0.001 * lad_loss    
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())
            train_pbar.set_postfix(loss=f"{train_loss_meter.avg:.4f}", lad_loss=f"{lad_loss:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3e}")

        model.eval()
        running_metrics_val.reset()
        val_loss_meter.reset()

        val_pbar = tqdm(val_loader, ncols=120, leave=False, desc=f"Valid | Epoch [{ep + 1}/{cfg['epochs']}]")
        with torch.no_grad():
            for sample in val_pbar:
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                text_token = sample['text_token'].to(device)
                text_mask = sample['txt_mask'].to(device)
                depth = sample.get('depth', None)
                depth = depth.to(device) if depth is not None else None

                if cfg['inputs'] == 'rgb':
                    out = model(image)
                else:
                    out, loss_dic = model(image, depth, text_token, text_mask)
                lad_loss = loss_dic['lad_loss']
                loss = criterion_0(out, label) + 0.001 * lad_loss      
                val_loss_meter.update(loss.item())

                pred = out.max(1)[1].cpu().numpy()
                gt = label.cpu().numpy()
                running_metrics_val.update(gt, pred)

                scores = running_metrics_val.get_scores()[0]
                miou_now = scores["mIou: "]
                acc_now = scores["class_acc: "]
                val_pbar.set_postfix(val_loss=f"{val_loss_meter.avg:.4f}", lad_loss=f"{lad_loss:.4f}", mIoU=f"{miou_now:.3f}", Acc=f"{acc_now:.3f}")
 
        scores_overall, cls_acc, cls_iu = running_metrics_val.get_scores()
        mAcc = scores_overall["class_acc: "]
        miou = scores_overall["mIou: "]

        logger.info(
            f'Epoch {ep + 1}: '
            f'TrainLoss={train_loss_meter.avg:.4f}, '
            f'ValLoss={val_loss_meter.avg:.4f}, '
            f'mIoU={miou:.3f}, Acc={mAcc:.3f}'
        )

        # Write epoch metrics to file
        with open(epoch_metrics_path, "a") as f:
            f.write(f"Epoch {ep + 1}\n")
            f.write(f"  TrainLoss: {train_loss_meter.avg:.6f}\n")
            f.write(f"  ValLoss  : {val_loss_meter.avg:.6f}\n")
            f.write(f"  pixel_acc: {scores_overall['pixel_acc: ']:.6f}\n")
            f.write(f"  class_acc: {scores_overall['class_acc: ']:.6f}\n")
            f.write(f"  mIoU     : {scores_overall['mIou: ']:.6f}\n")
            f.write(f"  fwIou    : {scores_overall['fwIou: ']:.6f}\n")

            f.write("  class_acc_per_class:\n")
            for cid, v in cls_acc.items():
                f.write(f"    class {cid}: {float(v):.6f}\n")

            f.write("  class_iou_per_class:\n")
            for cid, v in cls_iu.items():
                f.write(f"    class {cid}: {float(v):.6f}\n")

            f.write("\n")

        # Checkpoint saving function
        def save_ckpt(logdir, name, model):
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, os.path.join(logdir, name + 'model.pth'))

        # Save best model (only after 5 epochs)
        if miou > maxmiou and (ep + 1) > 5:
            logger.info(f"Saving best model, mIoU improved from {maxmiou:.4f} to {miou:.4f}")
            maxmiou = miou
            best_epoch = ep + 1
            save_ckpt(logdir, 'best_', model)

        scheduler.step()

    print(f"Best mIoU: {round(maxmiou, 4)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--config", type=str, default="/root for /configs/pst900.json")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbt'])
    parser.add_argument("--resume", type=str, default='/root for /sam2_hiera_large.pt')
    parser.add_argument("--dataset", type=str, default='pst900')
    parser.add_argument("--root", type=str, default='/root for /train/')
    parser.add_argument("--device", type=str, default='0', help="GPU ids to use, e.g. '0' or '0,1,2'")
    args = parser.parse_args()
    run(args)