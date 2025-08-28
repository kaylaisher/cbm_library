import os, torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from . import data_utils
from ..models.vlg_cbm import NormalizationLayer

def get_concept_dataloader(dataset, split, concepts, preprocess,
                           val_split, batch_size, num_workers,
                           shuffle, confidence_threshold,
                           crop_to_concept_prob, label_dir,
                           use_allones, seed):
    root = os.environ.get("TORCH_DATA","./data")
    train = (split in ["train","val"])
    base = CIFAR10(root=root, train=train, download=True)
    base.transform = preprocess
    if split=="val" and val_split:
        n=len(base); n_val=int(n*val_split)
        torch.manual_seed(seed)
        perm=torch.randperm(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        subset = Subset(base, val_idx)
    elif split=="train" and val_split:
        n=len(base); n_val=int(n*val_split)
        torch.manual_seed(seed)
        perm=torch.randperm(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        subset = Subset(base, tr_idx)
    else:
        subset=base

    class Wrap(Dataset):
        def __init__(self, base, concepts, use_allones):
            self.base=base; self.C=len(concepts); self.use_allones=use_allones
        def __len__(self): return len(self.base)
        def __getitem__(self,idx):
            img,y=self.base[idx]
            if self.use_allones: c=torch.ones(self.C)
            else: c=torch.zeros(self.C)
            return img,c,y
    return DataLoader(Wrap(subset,concepts,use_allones),
                      batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)

def get_filtered_concepts_and_counts(dataset, raw_concepts, preprocess,
                                     val_split, batch_size, num_workers,
                                     confidence_threshold, label_dir,
                                     use_allones, seed):
    cons=[data_utils.format_concept(c) for c in raw_concepts]
    counts={c:1 for c in cons}
    return cons, counts, cons

class ConceptTensorDataset(Dataset):
    def __init__(self,X,y): self.X=X; self.y=y
    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y[idx], idx

def get_final_layer_dataset(backbone,cbl,train_loader,val_loader,
                            save_dir,load_dir,batch_size,device):
    def encode(loader):
        Xs,Ys=[],[]
        with torch.no_grad():
            for imgs,_,y in tqdm(loader):
                imgs=imgs.to(device)
                emb=backbone(imgs); logits=cbl(emb)
                Xs.append(logits.cpu()); Ys.append(y)
        return torch.cat(Xs), torch.cat(Ys)
    Xtr,Ytr=encode(train_loader); Xva,Yva=encode(val_loader)
    mean, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    norm=NormalizationLayer(mean,std,device=device)
    Xtr=(Xtr-mean)/std; Xva=(Xva-mean)/std
    tr_ds,va_ds=ConceptTensorDataset(Xtr,Ytr),ConceptTensorDataset(Xva,Yva)
    tr_dl=DataLoader(tr_ds,batch_size=batch_size,shuffle=True)
    va_dl=DataLoader(va_ds,batch_size=batch_size,shuffle=False)
    return tr_dl,va_dl,norm
