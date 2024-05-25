# -*- coding: utf-8 -*-

import torch

def evaluate(
    model,
    test_dataloader,
    criterion,
    device,
    qtn_tokenizer,
    ans_vocab,
    is_test=False,
):
    model.eval()
    epoch_loss = 0

    if is_test:
        labels_file = open("logs/predictions.txt","w")
        labels_file.write("Images \t Qtns \t Target \t Prediction \n")

    with torch.no_grad():
        for i, (imgs, qtn_ids, qtn_attns, ans) in enumerate(test_dataloader):
            qtn_attns = qtn_attns.to(device)
            qtn_ids = qtn_ids.to(device)
    
            ans = torch.tensor(ans).long().to(device)

            output = model(
                imgs,
                qtn_ids,
                qtn_attns,
                device,
            )
            
            pred = torch.argmax(output, dim=-1)

            """
            q shape:  torch.Size([16, 32])
            a shape:  torch.Size([16, 3])
            ouptut shape:  torch.Size([16, 3, 52])
            pred shape:  torch.Size([16, 3])
            """

            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]), 
                ans.contiguous().view(-1))

            epoch_loss += loss.item()
            
            if is_test:
                for b in range(ans.shape[0]):
                    im = imgs[b]
                    q = qtn_tokenizer.decode(qtn_ids[b,:])
                    a = "".join([ans_vocab.itos[i] for i in ans[b,:]])
                    p = "".join([ans_vocab.itos[i] for i in pred[b,:]])
                    
                    labels_file.write(
                        f"{im} \t {q} \t {a} \t {p} \n"
                    )
            

        net_loss = epoch_loss / len(test_dataloader)
        return net_loss   