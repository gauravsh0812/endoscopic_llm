# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
    data_path, 
    train_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    ddp=False,
    rank=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (imgs, qtn_ids, qtn_attns, ans) in enumerate(tset):
        qtn_attns = qtn_attns.to(device)
        qtn_ids = qtn_ids.to(device)
        
        # setting gradients to zero
        optimizer.zero_grad()

        output = model(
            imgs,
            qtn_ids,
            qtn_attns,
            ans,
            device,
        )

        print("output shape: ", output.shape)

        exit()

        labels = torch.argmax(labels, dim=1)  # (B,)

        loss = criterion(output.contiguous().view(-1, output.shape[-1]), 
                         labels.contiguous().view(-1))
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss