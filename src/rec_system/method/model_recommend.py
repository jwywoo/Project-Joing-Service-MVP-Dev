
'''
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
'''


import argparse
import os
import pickle

import dgl
# import evaluation
from src.rec_system.method import layers
import numpy as np
from src.rec_system.method import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textsets, hidden_dims
        )
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks, item_emb):
        h_item = self.get_repr(blocks, item_emb)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks, item_emb):
        # project features
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)

        # add to the item embedding itself
        h_item = h_item + item_emb(blocks[0].srcdata[dgl.NID].cpu()).to(h_item)
        h_item_dst = h_item_dst + item_emb(
            blocks[-1].dstdata[dgl.NID].cpu()
        ).to(h_item_dst)

        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, args):
    g = dataset["train-graph"]
    val_matrix = dataset["val-matrix"].tocsr()
    test_matrix = dataset["test-matrix"].tocsr()
    item_texts = dataset["item-texts"]
    user_ntype = dataset["user-type"]
    item_ntype = dataset["item-type"]

    device = torch.device(args.device)

    # Prepare torchtext dataset and vocabulary
    textset = {}
    tokenizer = get_tokenizer(None)

    textlist = []
    batch_first = True

    # 수정된 코드(item text 이슈)
    for i in range(g.num_nodes(item_ntype)):
        l = tokenizer(item_texts[i].lower())
        textlist.append(l)

    vocab2 = build_vocab_from_iterator(
        textlist, specials=["<unk>", "<pad>"]
    )
    textset["item-texts"] = (
        textlist,
        vocab2,
        vocab2.get_stoi()["<pad>"],
        batch_first,
    )

    # Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size
    )
    neighbor_sampler = sampler_module.NeighborSampler(
        g,
        user_ntype,
        item_ntype,
        args.random_walk_length,
        args.random_walk_restart_prob,
        args.num_random_walks,
        args.num_neighbors,
        args.num_layers,
        user_to_item_etype= "creator_to_item",  # 새로운 엣지 타입 반영
        item_to_user_etype= "item_to_creator"   # 새로운 엣지 타입 반영
    )
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset
    )
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers,
    )

    # `torch.arange(g.num_nodes(item_ntype))`을 Dataset으로 감싸기
    item_tensor = torch.arange(g.num_nodes(item_ntype))
    item_dataset = TensorDataset(item_tensor)

    dataloader_test = DataLoader(
        item_dataset,
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )

    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)
    item_emb = nn.Embedding(
        g.num_nodes(item_ntype), args.hidden_dims, sparse=True
    )
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_emb = torch.optim.SparseAdam(item_emb.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks, item_emb).mean()
            opt.zero_grad()
            opt_emb.zero_grad()
            loss.backward()
            opt.step()
            opt_emb.step()

        # Evaluate
        '''
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                args.batch_size
            )
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader_test):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                h_item_batches.append(model.get_repr(blocks, item_emb))
            h_item = torch.cat(h_item_batches, 0)

            print(
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
            )
        '''

    '''
    dataloader_test = DataLoader(
        torch.arange(g.num_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers,
    )
    '''
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(
        g, item_ntype, textset, args.hidden_dims, args.num_layers
    ).to(device)
    item_emb = nn.Embedding(
        g.num_nodes(item_ntype), args.hidden_dims, sparse=True
    )
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt_emb = torch.optim.SparseAdam(item_emb.parameters(), lr=args.lr)

    # For each batch of head-tail-negative triplets...
    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in tqdm.trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks, item_emb).mean()
            opt.zero_grad()
            opt_emb.zero_grad()
            loss.backward()
            opt.step()
            opt_emb.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.num_nodes(item_ntype)).split(
                args.batch_size
            )
            h_item_batches = []
            for blocks in tqdm.tqdm(dataloader_test):
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks, item_emb))
            h_item = torch.cat(h_item_batches, 0)
            '''
            print(
                evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size)
            )
            '''
    return model, item_emb  # 학습이 완료된 모델과 임베딩을 반환

