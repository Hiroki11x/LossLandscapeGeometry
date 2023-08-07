import argparse
import torch
import numpy as np

import wandb
import os
import random

from utils import misc, calc
from dataset import build_dataset, DatasetSetting, sample_n_batches
from model import build_model, ModelSetting
from optimizer import build_optimizer, OptimizerSetting
from scheduler import linear_warmup_cosine_decay

def get_xstar(exp_dict):
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    # Initialize model
    model = build_model(ModelSetting(name=exp_dict['model'], 
                                     num_classes=exp_dict['num_classes']))

    model.to(device)
    misc.print_model_summary(model, exp_dict['dataset'])
    
    if exp_dict['num_gpu'] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)

    # Initialize Data Set and Data Loader
    dataset = build_dataset(DatasetSetting(name=exp_dict['dataset'], 
                                           root=exp_dict['data_root']))

    train_fixed_generator = torch.Generator()
    train_fixed_generator.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(dataset.train_dataset,
                                              batch_size=exp_dict['batch_size'],
                                              shuffle=True,
                                              num_workers=exp_dict['num_workers'],
                                              worker_init_fn=misc.seed_worker,
                                              generator=train_fixed_generator,
                                              drop_last=True)


    val_fixed_generator = torch.Generator()
    val_fixed_generator.manual_seed(0)

    val_loader = torch.utils.data.DataLoader(dataset.val_dataset,
                                              batch_size=exp_dict['batch_size'],
                                              shuffle=False,
                                              num_workers=exp_dict['num_workers'],
                                              worker_init_fn=misc.seed_worker,
                                              generator=val_fixed_generator,
                                              drop_last=True)
 
    # Initialize optimizer
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"] 
                                           ))
    
    # Initialize scheduler
    if exp_dict["use_scheduler"]:
        warmup_steps = len(train_loader) * exp_dict["warmup_epochs"]
        total_steps = len(train_loader) * exp_dict["epochs_budget"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                        opt,
                        linear_warmup_cosine_decay(warmup_steps, total_steps),
                    )
        
    # Misc
    CEloss = torch.nn.CrossEntropyLoss()


    for epoch in range(0, exp_dict['epochs_budget']):

        train_total = 0
        train_correct = 0
        train_losses = []

        # Training
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()

            preds = model(imgs)
            _, pred_label = torch.max(preds.data, 1)
            train_total += labels.size(0)
            train_correct += (pred_label == labels).sum().item()

            train_loss = CEloss(preds, labels)
            train_losses.append(train_loss.item())

            train_loss.backward()
            opt.step()
            
            if exp_dict["use_scheduler"]:
                scheduler.step()

        # Validation
        model.eval()
        val_total = 0
        val_correct = 0
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = model(imgs)
                _, pred_label = torch.max(preds.data, 1)
                val_total += labels.size(0)
                val_correct += (pred_label == labels).sum().item()

                val_loss = CEloss(preds, labels)
                val_losses.append(val_loss.item())


        avg_train_loss = np.average(train_losses)
        avg_train_acc = 100 * train_correct / train_total
        avg_val_loss = np.average(val_losses)
        avg_val_acc = 100 * val_correct / val_total


        wandb.log({'epoch': epoch,
                   '_train_loss': avg_train_loss,
                   '_train_acc': avg_train_acc,
                   '_val_loss': avg_val_loss,
                   '_val_acc': avg_val_acc})

    X_STAR = calc.get_weights(model)
    
    return X_STAR, avg_train_loss
    

def get_fullbatch_gradient(exp_dict, model, loader):
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    CEloss = torch.nn.CrossEntropyLoss()
    num_accumulate = len(loader)
    accumulated_loss = 0

    model.train(True) 

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        preds = model(imgs)
        loss = CEloss(preds, labels)
        accumulated_loss += loss.item()
        loss = loss / num_accumulate
        loss.backward()

    fb_grad = calc.get_grads(model)
    fb_loss = accumulated_loss/num_accumulate
    return fb_grad, fb_loss


def get_batch_list_grads(batch_list, model, opt, X_star):
        model.train()
        CEloss = torch.nn.CrossEntropyLoss()
        batch_dict = {}
        for i, (imgs, labels) in enumerate(batch_list):
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()

            preds = model(imgs)
            loss = CEloss(preds, labels)
            loss.backward()
            weights, grads = calc.get_weights_and_grads(model)
            opt.zero_grad()
            norm_grad_wd, dot_prod_wd = calc.get_norm_dot_prod_with_wd(weights, grads, X_star, exp_dict["weight_decay"])
            norm_dist, norm_grad, dot_prod = calc.get_norm_dot_prod(weights, grads, X_star)
            batch_dict[f"norm_grad_wd_{i}"] = norm_grad_wd
            batch_dict[f"dot_prod_wd_{i}"] = dot_prod_wd
            batch_dict[f"norm_grad_{i}"] = norm_grad
            batch_dict[f"dot_prod_{i}"] = dot_prod
            batch_dict['norm_dist'] = norm_dist
        return batch_dict
            

def trainval(exp_dict, X_star):
    
    if not exp_dict['fast_compute']:
        batch_list = sample_n_batches(exp_dict)
    
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    # Initialize Model
    model = build_model(ModelSetting(name=exp_dict['model'], 
                                     num_classes=exp_dict['num_classes']))
    model.to(device)
    if exp_dict['num_gpu'] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)

    # Initialize Data Set and Data Loader
    dataset = build_dataset(DatasetSetting(name=exp_dict['dataset'], 
                                           root=exp_dict['data_root']))

    fixed_generator = torch.Generator()
    fixed_generator.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(dataset.train_dataset,
                                              batch_size=exp_dict['batch_size'],
                                              shuffle=True,
                                              num_workers=exp_dict['num_workers'],
                                              worker_init_fn=misc.seed_worker,
                                              generator=fixed_generator,
                                              drop_last=True)


    if not exp_dict['fast_compute']:
    # Initialize Data Set and Data Loader
        fb_dataset = build_dataset(DatasetSetting(name=exp_dict['dataset'], 
                                                root=exp_dict['data_root']))

        fb_fixed_generator = torch.Generator()
        fb_fixed_generator.manual_seed(0)

        fb_train_loader = torch.utils.data.DataLoader(fb_dataset.train_dataset,
                                                batch_size=exp_dict['batch_size'],
                                                shuffle=True,
                                                num_workers=exp_dict['num_workers'],
                                                worker_init_fn=misc.seed_worker,
                                                generator=fb_fixed_generator,
                                                drop_last=True)

    # Initialize Optimizer
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"] 
                                           ))
    
    # Initialize Scheduler
    if exp_dict["use_scheduler"]:
        warmup_steps = len(train_loader) * exp_dict["warmup_epochs"]
        total_steps = len(train_loader) * exp_dict["epochs_budget"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                        opt,
                        linear_warmup_cosine_decay(warmup_steps, total_steps),
                    )
        
    CEloss = torch.nn.CrossEntropyLoss()
    iter = 0

    for epoch in range(0, exp_dict['required_epochs']):

        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()

            preds = model(imgs)
            loss = CEloss(preds, labels)
            loss.backward()
            weights, grads = calc.get_weights_and_grads(model)
            norm_dist, norm_grad, dot_prod = calc.get_norm_dot_prod(weights, grads, X_star)
            if exp_dict["weight_decay"]:
                norm_grad_wd, dot_prod_wd = calc.get_norm_dot_prod_with_wd(weights, grads, X_star, exp_dict["weight_decay"])
            opt.step()
            
            new_weights, _ = calc.get_weights_and_grads(model)
            update = new_weights - weights
            _, norm_update, dot_prod_update = calc.get_norm_dot_prod(weights, update, X_star)
            
            if exp_dict["use_scheduler"]:
                scheduler.step()

            score_dict = {
                "iteration": iter,
                "epoch": epoch,
                "train_loss": loss.item(),
                "norm_dist" : norm_dist,
                "norm_grad" : norm_grad,
                "dot_prod" : dot_prod,
                "norm_update" : norm_update,
                "dot_prod_update" : dot_prod_update
            }
            if exp_dict["weight_decay"]:
                score_dict["norm_grad_wd"] = norm_grad_wd
                score_dict["dot_prod_wd"] = dot_prod_wd

            wandb.log(score_dict)
            iter += 1

        opt.zero_grad()
        
        if not exp_dict['fast_compute']:
            # fullbatch evaluation
            fb_grads, fb_loss = get_fullbatch_gradient(exp_dict, model, fb_train_loader)
            fb_norm_dist, fb_norm_grad, fb_dot_prod = calc.get_norm_dot_prod(weights, fb_grads, X_star)
            if exp_dict["weight_decay"]:
                fb_norm_grad_wd, fb_dot_prod_wd = calc.get_norm_dot_prod_with_wd(
                                                        weights, fb_grads, X_star, exp_dict["weight_decay"]
                                                    )

            fb_score_dict = {
                    "iteration": iter,
                    "epoch": epoch,
                    "fb_loss": fb_loss,
                    "fb_norm_dist" : fb_norm_dist,
                    "fb_norm_grad" : fb_norm_grad,
                    "fb_dot_prod" : fb_dot_prod,
                }
            if exp_dict["weight_decay"]:
                fb_score_dict["fb_norm_grad_wd"] = fb_norm_grad_wd
                fb_score_dict["fb_dot_prod_wd"] = fb_dot_prod_wd

            wandb.log(fb_score_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LossLandscape Geometry Project (Image Classification)')

    # Environmental Setting
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpu', type=int, default=2, help="num of gpus")

    # Experimental Setting
    parser.add_argument('--epochs_budget', type=int, default=200)
    parser.add_argument('--model', type=str, default='resnet18_1')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--num_classes', type=int, default=10)

    # Hyperparameter
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--opt', type=str, default='vanilla_sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--use_scheduler', type=bool, default=False)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    
    parser.add_argument('--tuning_only', action = 'store_true')
    parser.add_argument('--fast_compute', action = 'store_true')

    parser.add_argument('--weight_decay', type=float, default=0) # follow pytorch default
    parser.add_argument('--eps', type=float, default=1e-08) # follow pytorch default
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)

    # Wandb Configuration
    parser.add_argument("--wandb_exp_id", type=int, default=99999999)
    parser.add_argument("--wandb_entity", type=str, default='your_wandb_entity', help="entitiy of wandb team")
    parser.add_argument("--wandb_project_name", type=str, default='default_project', help="should include dataset and model")
    parser.add_argument('--wandb_offline', action = 'store_true')
    args = parser.parse_args()


    # Environmental Setting
    device = torch.device("cuda" if (torch.cuda.is_available() and args.num_gpu > 0) else "cpu")
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    misc.set_seeds(args.seed)
    misc.print_libs_version()
    args.data_root =  misc.update_dataroot(args.dataset, args.data_root)

    # Wandb Setting
    wandb_exp_name = f'exp_id-{args.wandb_exp_id}_opt-{args.opt}_bs-{args.batch_size}_lr-{args.lr}_ep-{args.epochs_budget}'
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(config=args,
               project=args.wandb_project_name,
               name=wandb_exp_name,
               entity=args.wandb_entity)

    print('\nWandb Setting:')
    print(f'\twandb_project_name: f{args.wandb_project_name}')
    print(f'\twandb_exp_name: f{wandb_exp_name}')


    # Set Training Config
    exp_dict = wandb.config
    exp_dict['device'] = device

    print('\nExperimental Configuration:')
    for k, v in sorted(exp_dict.items()):
        print('\t{}: {}'.format(k, v))

    X_star, avg_train_loss = get_xstar(exp_dict)
    print(f'avg_train_loss: {avg_train_loss}')
    exp_dict['avg_train_loss'] = avg_train_loss

    # Execute Training
    if not args.tuning_only:
        trainval(exp_dict, X_star)
