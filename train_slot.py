import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import torch.utils.data as Data
from model import SeqTagger
'''
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
'''
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    # TODO: crecate DataLoader for train / dev datasets
    batch_size = args.batch_size
    train_loader = Data.DataLoader(
        dataset =  datasets['train'],
        batch_size = batch_size,
        shuffle = True,
        collate_fn = datasets['train'].collate_fn
    )
    eval_loader = Data.DataLoader(
        dataset =  datasets['eval'],
        batch_size = batch_size,
        shuffle = True,
        collate_fn = datasets['eval'].collate_fn
    )
    # END TODO
    
     
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device(args.device)  
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets["train"].num_classes)
    model.to(device = device)

    # TODO: init optimizer
    #optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr , weight_decay = 1.0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 5)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    pre_eval_acc_rate = 0.0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss = 0.0
        m = 0
        for i , batch in enumerate(tqdm(train_loader)):
            datas = batch["tokens"].to(device)
            labels = batch["tags"].to(device)
            m = m + labels.size(0)
            
            # forward
            optimizer.zero_grad()
            outputs = model(datas)
            
            loss = loss_function(outputs , labels)
            
            # backward
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item() * labels.size(0)
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            eval_acc = 0
            n = 0
            for i , batch in enumerate(tqdm(eval_loader)):
                datas = batch["tokens"].to(device)
                labels = batch["tags"].to(device)
                n = n + labels.size(0)

                outputs = model(datas)
                preds = torch.argmax(outputs, dim=1)
                
                eval_acc += torch.all(preds.eq(labels), dim = 1).sum().item()

                # Q4
                #print(classification_report(labels.cpu().numpy() , preds.cpu().numpy() , mode='strict' , scheme=IOB2))

            if pre_eval_acc_rate < float(eval_acc / n):
                print()
                print("++++++++++++++++++++++++++++++++++++++++++++")
                print("Current best epoch : " , epoch)
                print("Current best eval_acc : %.6f" %float(eval_acc / n))
                print("++++++++++++++++++++++++++++++++++++++++++++")
                model_path = args.ckpt_dir / "best.pt"
                checkpoint = {
                        "model" : model.state_dict(),
                        "train_loss" : float(train_loss / m),
                        "eval_acc" : float(eval_acc / n),
                }
                torch.save(checkpoint , model_path)
                pre_eval_acc_rate = float(eval_acc / n)

        print()
        print("--------------------------------------------")
        print("Epoch : " , epoch)
        print("train_loss_rate : %.6f" %float(train_loss / m))
        print("eval_acc_rate : %.6f" %float(eval_acc / n))
        print("--------------------------------------------")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)
    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=300)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
