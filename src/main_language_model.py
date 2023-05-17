

import argparse
import math
import os
import torch
import numpy as np
import wandb
import random

from model import build_nlp_model
from utils import calc, misc
from dataset.corpus import Corpus
from dataset.datahandler import DataHandler
from dataset.preprocess import preprocess
from optimizer import build_optimizer, OptimizerSetting

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer


def get_xstar(exp_dict):
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    # Build Dataset
    if exp_dict['dataset'] == 'wikitext-2':
        corpus = Corpus(exp_dict['data_root'])
    else:
        raise NotImplementedError

    # Data Loader
    datasets = WikiText2(exp_dict['data_root'], split=("train", "valid", "test"))
    tokenizer = get_tokenizer("basic_english", language="en")
    
    handler = DataHandler(datasets)
    train_loader, val_loader, vocab, _counter = handler.load_data(preprocess=preprocess, 
                                                                 tokenizer=tokenizer, 
                                                                 batch_size=exp_dict['batch_size'],
                                                                 window_size=exp_dict['window_size'], 
                                                                 min_freq=5,
                                                                 num_workers=exp_dict['num_workers'],
                                                                 shuffle=True)

    # Build Model
    ntokens = len(corpus.dictionary)
    exp_dict['vocab_size'] = len(vocab)
    exp_dict['pad_idx'] = vocab["<pad>"]
    model = build_nlp_model(exp_dict, ntokens)
    model.to(device)
    misc.print_model_summary(model, exp_dict['dataset'])
    if exp_dict['num_gpu'] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)

    # Loss Function
    if 'Transformer' in exp_dict['model']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=exp_dict['pad_idx'])


    # Optimizer
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"] 
                                           ))


    for epoch in range(0, exp_dict['epochs_budget']):

        # Training
        model.train()
        train_losses = []

        for batch, data in enumerate(train_loader):
            data = data.to(device)

            model.zero_grad()
            if 'Transformer' in exp_dict['model']:
                output = model(data[:, :-1])
                output = output.view(-1, ntokens)
                _, vocab_size = output.shape
            else:
                output = model(data[:, :-1])
                _, _, vocab_size = output.shape

            opt.zero_grad()
            loss = criterion(output.reshape(-1, vocab_size), data[:, 1:].flatten())
 
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_dict['clip'])
            opt.step()


        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                data = data.to(device)
                if 'Transformer' in exp_dict['model']:
                    output = model(data[:, :-1])
                    output = output.view(-1, ntokens)
                    _, vocab_size = output.shape
                else:
                    output = model(data[:, :-1])
                    _, _, vocab_size = output.shape

                loss = criterion(output.reshape(-1, vocab_size), data[:, 1:].flatten())
                val_losses.append(loss.item())


        avg_train_loss = np.average(train_losses)
        avg_train_ppl = np.exp(avg_train_loss)
        avg_val_loss = np.average(val_losses)
        avg_val_ppl = np.exp(avg_val_loss)

        wandb.log({'epoch': epoch,
                   '_train_loss': avg_train_loss,
                   '_train_ppl': avg_train_ppl,
                   '_val_loss': avg_val_loss,
                   '_val_ppl': avg_val_ppl})

        X_STAR = calc.get_weights(model)

    return X_STAR, avg_train_loss



def trainval(exp_dict, X_star):
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    # Build Dataset
    if exp_dict['dataset'] == 'wikitext-2':
        corpus = Corpus(exp_dict['data_root'])
    else:
        raise NotImplementedError

    # Build Model
    ntokens = len(corpus.dictionary)
    model = build_nlp_model(exp_dict, ntokens)
    model.to(device)
    misc.print_model_summary(model, exp_dict['dataset'])
    if exp_dict['num_gpu'] > 1:
        print("DataParallel")
        model = torch.nn.DataParallel(model)

    # Data Loader
    datasets = WikiText2(exp_dict['data_root'], split=("train", "valid", "test"))
    tokenizer = get_tokenizer("basic_english", language="en")
    
    handler = DataHandler(datasets)
    train_loader, _, _vocab, _counter = handler.load_data(preprocess=preprocess, 
                                                        tokenizer=tokenizer, 
                                                        batch_size=exp_dict['batch_size'],
                                                        window_size=exp_dict['window_size'], 
                                                        min_freq=5,
                                                        num_workers=exp_dict['num_workers'],
                                                        shuffle=True)


    fb_train_loader, _, _vocab, _counter = handler.load_data(preprocess=preprocess, 
                                                           tokenizer=tokenizer, 
                                                           batch_size=exp_dict['batch_size'],
                                                           window_size=exp_dict['window_size'], 
                                                           min_freq=5,
                                                           num_workers=exp_dict['num_workers'],
                                                           shuffle=True)

    # Loss and Early Stopping
    if 'Transformer' in exp_dict['model']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=exp_dict['pad_idx'])

    # Optimizer
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"] 
                                           ))

    # Training
    model.train()
    iter = 0

    for epoch in range(0, exp_dict['required_epochs']):

        for batch, data in enumerate(train_loader):
            data = data.to(device)

            model.zero_grad()
            if 'Transformer' in exp_dict['model']:
                output = model(data[:, :-1])
                output = output.view(-1, ntokens)
                _, vocab_size = output.shape
            else:
                output = model(data[:, :-1])
                _, _, vocab_size = output.shape

            opt.zero_grad()
            loss = criterion(output.reshape(-1, vocab_size), data[:, 1:].flatten())
            loss.backward()

            # # Calc Metrics
            weights, grads = calc.get_weights_and_grads(model)
            norm_dist, norm_grad, dot_prod = calc.get_norm_dot_prod(weights, grads, X_star)
            if exp_dict["weight_decay"]:
                norm_grad_wd, dot_prod_wd = calc.get_norm_dot_prod_with_wd(weights, grads, X_star, exp_dict["weight_decay"])
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp_dict['clip'])
            opt.step()

            if iter % 51 == 0:
                # Initialize Cosine Log Dictionary
                cosine_dict = {} 

                # compute g_i and x_i-x* 
                x_i = weights
                g_i = grads
                diff_x_i_and_x_star = x_i - X_star
                
                # compute orth_g_i = g_i - (g_i^T . (x_i - x*)) * (x_i - x*) / ||x_i-x*||^2
                _norm_i = norm_dist
                orth_g_i = g_i - dot_prod * diff_x_i_and_x_star /((_norm_i )** 2)

                base_grad_orth = orth_g_i
                base_x = x_i

            k = iter % 51
            if k in [1, 2, 5, 10, 20, 50]:
                
                # compute orth_g_i+k 
                x_k = weights
                g_k = grads
                diff_x_k_and_x_star = x_k - X_star
                _norm_k = norm_dist
                orth_g_k = g_k - dot_prod * diff_x_k_and_x_star / ((_norm_k )** 2)

                # compute sim_k = cosine(base_grad_orth, orth_g_i+k)
                sim_k = torch.cosine_similarity(base_grad_orth, orth_g_k, dim=0)

                # compute dist_k = ||base_x - x_i+k||_2
                dist_k = torch.norm(base_x - x_k, p=2)

                # cosine_dict[str(k)].append((sim_k, dist_k))
                cosine_dict[f'sim_{k}'] = sim_k
                cosine_dict[f'dist_{k}'] = dist_k

                if k in [50]:
                    cosine_dict['iteration'] = iter
                    cosine_dict['epoch'] = epoch
                    wandb.log(cosine_dict)

            score_dict = {
                "iteration": iter,
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_ppl": math.exp(loss.item()),
                "norm_dist" : norm_dist,
                "norm_grad" : norm_grad,
                "dot_prod" : dot_prod,
            }
            if exp_dict["weight_decay"]:
                score_dict["norm_grad_wd"] = norm_grad_wd
                score_dict["dot_prod_wd"] = dot_prod_wd

            wandb.log(score_dict)
            iter += 1


        # fullbatch evaluation
        fb_grads, fb_loss = get_fullbatch_gradient(exp_dict, model, fb_train_loader, ntokens)
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


def get_fullbatch_gradient(exp_dict, model, data_loader, ntokens):
    device = exp_dict['device']
    misc.set_seeds(exp_dict['seed'])

    num_accumulate = len(data_loader)
    accumulated_loss = 0

    # Loss and Early Stopping
    if 'Transformer' in exp_dict['model']:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=exp_dict['pad_idx'])

    # Optimizer
    opt = build_optimizer(OptimizerSetting(name=exp_dict['opt'],
                                           lr=exp_dict['lr'],
                                           weight_decay=exp_dict['weight_decay'],
                                           model=model,
                                           momentum=exp_dict["momentum"],
                                           eps=exp_dict["eps"],
                                           beta_1=exp_dict["beta_1"],
                                           beta_2=exp_dict["beta_2"] 
                                           ))

    # Enable Training for Getting Full Batch Gradient
    opt.zero_grad()
    model.train()

    for batch, data in enumerate(data_loader):
        data = data.to(device)

        model.zero_grad()
        if 'Transformer' in exp_dict['model']:
            output = model(data[:, :-1])
            output = output.view(-1, ntokens)
            _, vocab_size = output.shape
        else:
            output = model(data[:, :-1])
            _, _, vocab_size = output.shape

        opt.zero_grad()
        loss = criterion(output.reshape(-1, vocab_size), data[:, 1:].flatten())
        accumulated_loss += loss.item()
        loss.backward()

    fb_grad = calc.get_grads(model)
    fb_loss = accumulated_loss/num_accumulate
    return fb_grad, fb_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='LossLandscape Geometry Project (Word Language Model)')

    # Environmental Setting
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_gpu', type=int, default=2, help="num of gpus")

    # Experimental Setting
    parser.add_argument('--epochs_budget', type=int, default=200)
    parser.add_argument('--model', type=str, default='Transformer', help='type of network')
    parser.add_argument('--dataset', type=str, default='wikitext-2')
    parser.add_argument('--data_root', type=str, default='./data/wikitext-2', help='location of the data corpus')
    parser.add_argument('--patience', type=int, default=200)

    # Hyperparameter
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=20)

    parser.add_argument('--weight_decay', type=float, default=0) # follow pytorch default
    parser.add_argument('--eps', type=float, default=1e-08) # follow pytorch default
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)

    # NLP Hyperparameter
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--window_size', type=int, default=32)

    # Wandb Configuration
    parser.add_argument("--wandb_exp_id", type=int, default=99999999)
    parser.add_argument("--wandb_entity", type=str, default='your_wandb_entity', help="entitiy of wandb team")
    parser.add_argument("--wandb_project_name", type=str, default='default_project', help="should include dataset and model")
    parser.add_argument('--wandb_offline', action = 'store_true')

    args = parser.parse_args()


    # ============= environmental setting =============
    device = torch.device("cuda" if (torch.cuda.is_available() and args.num_gpu > 0) else "cpu")
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    misc.set_seeds(args.seed)
    misc.print_libs_version()
    args.data_root =  misc.update_dataroot(args.dataset, args.data_root)


    # ================ wandb ================ 
    wandb_exp_name = f'exp_id-{args.wandb_exp_id}_opt-{args.opt}_bs-{args.batch_size}_lr-{args.lr}'
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(config=args,
               project=args.wandb_project_name,
               name=wandb_exp_name,
               entity=args.wandb_entity)


    # ============= train config =============
    exp_dict = wandb.config
    exp_dict['device'] = device
    print('\nExperimental Configuration:')
    for k, v in sorted(exp_dict.items()):
        print('\t{}: {}'.format(k, v))

    X_star, avg_train_loss = get_xstar(exp_dict)
    print(f'avg_train_loss: {avg_train_loss}')
    exp_dict['avg_train_loss'] = avg_train_loss

    trainval(exp_dict, X_star)