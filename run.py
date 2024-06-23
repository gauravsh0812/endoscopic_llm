import os
import yaml
import random
import time
import math
import wandb
import numpy as np
import multiprocessing as mp
from box import Box
import torch 
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from preprocessing.create_dataloaders import data_loaders

# from models.clip import ClipVisionEncoder
# from models.roberta import RobertaEncoder
# from models.model import Endoscopic_model
# from models.adaptor import ClipAdaptor, Projector, RobertaAdaptor

from src.training import train
from src.testing import evaluate
from utils import *

with open("config/config_cliprob.yaml") as f:
    cfg = Box(yaml.safe_load(f))

def set_random_seed(SEED):
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def count_parameters(model):
    """
    counting total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    """
    epoch timing
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def define_model(device, max_len, ans_vocab):
    
    if cfg.model_name == "clip_roberta_adaptor_clf":
        from models.clip_rob_clf import Endoscopic_model
        model = Endoscopic_model(max_len, ans_vocab)
    
    elif cfg.model_name == "surgical_gpt":
        from models.surgpt import Endoscopic_model
        model = Endoscopic_model(device, max_len, len(ans_vocab))
    
    return model

    # CLIPENC = ClipVisionEncoder()
    # ROBENC = RobertaEncoder()    

    # if cfg.training.clip.finetune:
    #     in_dim = cfg.training.clip.configuration.hidden_size
    # else:
    #     in_dim = 768
    # CLIPADA = ClipAdaptor(in_dim, 
    #               cfg.training.adaptor.features,
    #               max_len,
    #               )
    
    # ROBADA = RobertaAdaptor(
    #     cfg.training.roberta.in_dim,
    #     cfg.training.adaptor.features,
    # )
    
    # PROJ = Projector(
    #     cfg.training.adaptor.fusion,
    #     cfg.training.adaptor.features,
    #     max_len, 
    #     len(ans_vocab),
    # )

    # # freezing the pre-trained models
    # # only training the adaptor layer
    # for param in CLIPENC.parameters():
    #     param.requires_grad = cfg.training.clip.finetune

    # for param in ROBENC.parameters():
    #     param.requires_grad = cfg.training.roberta.finetune 

    # model = Endoscopic_model(CLIPENC, 
    #                         ROBENC,
    #                         CLIPADA,
    #                         ROBADA,
    #                         PROJ,)
    # return model

def train_model(rank=None):

    if (cfg.general.wandb):
        if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0): 
            # initiate the wandb    
            wandb.init()
            wandb.config.update(cfg)

    # set_random_seed
    set_random_seed(cfg.general.seed)
    
    # to save trained model and logs
    FOLDER = ["trained_models", "logs"]
    for f in FOLDER:
        if not os.path.exists(f):
            os.mkdir(f)

    # to log losses
    loss_file = open("logs/loss_file.txt", "w")
    
    # defining model using DataParallel
    if torch.cuda.is_available() and cfg.general.device == "cuda":
        if not cfg.general.ddp:
            print(f"using single gpu:{cfg.general.gpus}...")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.general.gpus)
            device = torch.device(f"cuda:{cfg.general.gpus}")
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                qtn_tokenizer,
                ans_vocab,
                max_len,
            ) = data_loaders(cfg.training.general.batch_size)
            model = define_model(device, max_len, ans_vocab).to(device)

        elif cfg.general.ddp:
            # create default process group
            dist.init_process_group("nccl", rank=rank, world_size=cfg.general.world_size)
            device = f"cuda:{rank}"
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                qtn_tokenizer, ans_vocab,
                max_len,
            ) = data_loaders(cfg.training.general.batch_size)
            model = define_model(device, max_len,ans_vocab)
            model = DDP(
                model.to(f"cuda:{rank}"),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )

    else:
        import warnings

        warnings.warn("No GPU input has provided. Falling back to CPU. ")
        device = torch.device("cpu")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            qtn_tokenizer, ans_vocab,
            max_len,
        ) = data_loaders()
        model = define_model(device, max_len,ans_vocab).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.training.general.learning_rate,
        weight_decay=cfg.training.general.weight_decay,
        betas=cfg.training.general.betas,
    )

    if cfg.training.scheduler.isScheduler:
        # scheduler
        print("scheduler ON...")
        scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size = cfg.training.scheduler.scheduler_step_size,
                    gamma=cfg.training.scheduler.scheduler_gamma,
        )

    else:
        scheduler = None

    best_valid_loss = float("inf")
    
    if (cfg.general.wandb):
        if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0): 
            wandb.watch(model)

    if not cfg.general.load_trained_model_for_testing:
        count_es = 0
        for epoch in range(cfg.training.general.epochs):
            if count_es <= cfg.training.general.early_stopping:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    train_dataloader,
                    optimizer,
                    criterion,
                    cfg.training.general.clip,
                    device,
                    use_ddp=cfg.general.ddp,
                    rank=rank,
                )

                val_loss = evaluate(
                    model,
                    val_dataloader,
                    criterion,
                    device,
                    qtn_tokenizer,
                    ans_vocab,
                )

                if cfg.general.ddp: pause_ddp_for_collection()

                if cfg.training.scheduler.isScheduler:
                    scheduler.step()

                if (cfg.general.wandb):
                    if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0): 
                        wandb.log({"train_loss": train_loss})
                        wandb.log({"val_loss": val_loss})

                end_time = time.time()
                # total time spent on training an epoch
                
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                
                # saving the current model for transfer learning
                if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/latest_model.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):                    
                        torch.save(
                            model.state_dict(),
                            f"trained_models/clip_roberta_adaptor_best_model.pt",
                        )

                        if (cfg.general.wandb):
                            wandb.save(f"trained_models/clip_roberta_adaptor_best_model.pt")
                else:
                    count_es += 1

                # logging
                if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
                    print(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
                    )
                    print(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
                    )
                    print(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}"
                    )

                    loss_file.write(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    )
                    loss_file.write(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n"
                    )
                    loss_file.write(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n"
                    )
                    
            else:
                print(
                    f"Terminating the training process as the validation loss hasn't been reduced from last {cfg.training.general.early_stopping} epochs."
                )
                break

        print(
            "best model saved as:  ",
            f"trained_models/{cfg.model_name}.pt",
        )

    if cfg.general.ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/clip_roberta_adaptor_best_model.pt",
    )
    # loading pre_tained_model
    model.load_state_dict(
        torch.load(
            f"trained_models/clip_roberta_adaptor_best_model.pt"
        )
    )

    test_loss = evaluate(
        model,
        test_dataloader,
        criterion,
        device,
        qtn_tokenizer,
        ans_vocab,
        is_test=True,
    )

    if (not cfg.general.ddp) or (cfg.general.ddp and rank == 0):
        print(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}"
        )
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}"
        )

    # stopping time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

def pause_ddp_for_collection():
    # Get the rank of the current process
    rank = dist.get_rank()

    # Print the rank for debugging
    print(f"Process rank: {rank}")

    # Synchronize all processes
    dist.barrier()

    if rank == 0:  # Only core 0 should print the message
        print("Pausing for 3 seconds to collect everything on core 0...")
    time.sleep(3)

    # Synchronize all processes again
    dist.barrier()


def ddp_main(world_size,):    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    mp.spawn(train_model, args=(), nprocs=world_size, join=True)

if __name__ == "__main__":
    if cfg.general.ddp:
        gpus = cfg.general.gpus
        world_size = cfg.general.world_size
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29800"
        ddp_main(world_size)

    else:
        train_model()
        