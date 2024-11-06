from models import SICDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 3
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')

# parser
parser = argparse.ArgumentParser(description="SICDTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'random1', 'random2', 'random3', 'random4','cold'])
args = parser.parse_args() 


def main():
    torch.cuda.empty_cache()    #clean CUDA
    warnings.filterwarnings("ignore", message="invalid value encountered in divide") 
    cfg = get_cfg_defaults()    # get_cfg_defaults
    set_seed(cfg.SOLVER.SEED)   # random seed
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}')  #madir

    print("start...")
    print(f"dataset:{args.data}")   #dataname
    print(f"Hyperparameters: {dict(cfg)}")  #Hyperparameter
    print(f"Running on: {device}", end="\n\n") # device

    dataFolder = f'./datasets/{args.data}'   
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    # DataLoader 
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                               'drop_last':True, 'collate_fn': graph_collate_func}
    
    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    
    # create model
    model = SICDTI(device=device, **cfg).to(device=device)
    
    # optim
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True
    
    #trainer
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, args.data, args.split, **cfg)
    result = trainer.train()
   
    #result
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')

    return result


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    result = main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s, ")




