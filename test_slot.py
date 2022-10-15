import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import torch.utils.data as Data
from tqdm import tqdm
import csv

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)

    # TODO: crecate DataLoader for test dataset
    batch_size = args.batch_size
    test_loader = Data.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = dataset.collate_fn,
    )
    # END

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    device = torch.device(args.device)
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, dataset.num_classes)
    model.to(device = device)

    # load weights into model
    ckpt = args.ckpt_path
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model"])

    # TODO: predict dataset

    with open(args.pred_file , "w") as file:
        writer = csv.writer(file)
        writer.writerow(["id","tags"])

        model.eval()
        with torch.no_grad():
            x_labels = []
            x_preds = []
            for i , batch in enumerate(tqdm(test_loader)):
                datas = batch["tokens"].to(device)
                outputs = model(datas)
                preds = torch.argmax(outputs, 1)

                # TODO: write prediction to file (args.pred_file)
                for j in range(len(preds)):
                    data_csv = ""
                    tmp = preds.view(-1).cpu().numpy()
                    for k in range(batch["len"][j]):
                        data_csv = data_csv + dataset.idx2label(tmp[args.max_len * j + k]) + " "
                    
                    data_csv = data_csv[:-1]
                    writer.writerow([batch["id"][j] , data_csv])
                
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        required=True
        #default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        required=True
        #default="./ckpt/slot/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="./result/slot/pred.slot.csv")

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
