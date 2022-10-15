import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

import torch.utils.data as Data
from tqdm import tqdm
import csv

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
	
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
	
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    
    # TODO: crecate DataLoader for test dataset
    batch_size = args.batch_size
    test_loader = Data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = dataset.collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    device = torch.device(args.device)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, dataset.num_classes)
    model.to(device = device)
    
    # load weights into model
    ckpt = args.ckpt_path
    checkpoint = torch.load(ckpt) 
    model.load_state_dict(checkpoint["model"])
    
    # TODO: predict dataset
    with open(args.pred_file , "w") as file:
        writer = csv.writer(file)
        writer.writerow(["id","intent"])
        
        with torch.no_grad():
            model.eval()
            for i , batch in enumerate(tqdm(test_loader)): 
                datas = batch["text"].to(device)
                outputs = model(datas)
            
                preds = torch.argmax(outputs, 1)
        
                # TODO: write prediction to file (args.pred_file)
                for j in range(len(preds)):
                    tmp = preds.view(-1).cpu().numpy()
                    writer.writerow([batch["id"][j],dataset.idx2label(tmp[j])])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="./result/intent/pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
