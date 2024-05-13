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
    accuracy = 0

    if is_test:
        labels_file = open("logs/predictions.txt","w")
        labels_file.write("Images \t Qtns \t Target \t Prediction \n")

    with torch.no_grad():
        for i, (imgs, qtn_ids, qtn_attns, ans) in enumerate(test_dataloader):
            qtn_attns = qtn_attns.to(device)
            qtn_ids = qtn_ids.to(device)
        
            ans = torch.stack(ans).long()
            ans = ans.to(device)

            output = model(
                imgs,
                qtn_ids,
                qtn_attns,
                device,
            )
            
            pred = torch.argmax(output, dim=-1)

            loss = criterion(
                output.contiguous().view(-1, output.shape[-1]), 
                ans.contiguous().view(-1))

            epoch_loss += loss.item()
            
            if is_test:
                # for b in imgs.shape[0]:
                    
                for i,q,a,p in zip(imgs,qtn_ids,ans,pred):
                    qtn = qtn_tokenizer.decode(q)
                    

                    labels_file.write(
                        f"{i} \t {qtn} \t {a} \t {p}"
                    )

        net_loss = epoch_loss / len(test_dataloader)
        accuracy = accuracy / len(test_dataloader)
        return net_loss, accuracy   