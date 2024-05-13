# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
    train_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    use_ddp=False,
    rank=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (imgs, qtn_ids, qtn_attns, ans) in enumerate(tset):
        qtn_attns = qtn_attns.to(device)
        qtn_ids = qtn_ids.to(device)
        
        ans = torch.stack(ans).long()
        ans = ans.to(device)

        # setting gradients to zero
        optimizer.zero_grad()

        output = model(
            imgs,
            qtn_ids,
            qtn_attns,
            device,
        )

        pred = torch.argmax(output, dim=-1)

        print("q, a, i, o, p shapes: ", qtn_ids.shape, ans.shape, imgs.shape, output.shape, pred.shape)    
                
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), 
                         ans.contiguous().view(-1))
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not use_ddp) or (use_ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss