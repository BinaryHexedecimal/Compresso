import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os, datetime
import time
import json
import multiprocessing

#import Process, Queue, Event
#from multiprocessing import Process, Queue, Event
#import random
#from torchvision import datasets, transforms

import globals
from src import TR   
from util_data import load_dataset_classes



# -----------------------------
# an simple model: ConvNet 
# -----------------------------
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNet, self).__init__()
        # Input: [in_channels, H, W]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Compute feature size dynamically
        # assuming input images are 28x28 (MNIST) or 32x32 (CIFAR/SVHN)
        # store it as attribute
        self.input_size = 28 if in_channels == 1 else 32
        self.fc1 = nn.Linear(64 * (self.input_size//2) * (self.input_size//2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




# --- Worker ---
# avoids the PyTorch multiprocessing + generator deadlocks
def train_worker(
                    req_obj ,
                    train_dataset,
                    progress_queue: multiprocessing.Queue,
                    cancel_event: multiprocessing.Event,
                    eps_linf=0.3,
                    eps_l2=1.5,
                ):

    test_dataset_file = globals.TEST_DATA_DIR / f"{req_obj.data_info.get('dataset_name')}_test.pt"
    test_dataset = torch.load(test_dataset_file, weights_only=False)

    all_epochs = []
    final_result_for_save = {"status": "started", 
                             "train_job_id": req_obj.train_job_id,
                             "timestamp": "",
                             "epochs": [], 
                             "req_obj": vars(req_obj)}

    try:
        # Notify start
        progress_queue.put({"type": "start"})
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False,  num_workers=0)


        sample, _ = train_dataset[0]
        _in_channels = sample.shape[0] 
        _num_classes = len(load_dataset_classes()[req_obj.data_info["dataset_name"]])
        print(f"inside training, channael is {_in_channels} , num of class is {_num_classes}")
        model = ConvNet(in_channels=_in_channels, num_classes=_num_classes)
        trainer = TR(model, train_loader, test_loader, eps_linf=eps_linf, eps_l2=eps_l2)
        
        # Select optimizer
        if req_obj.optimizer.lower() == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=req_obj.learning_rate, momentum=0.9, weight_decay=5e-4)
        elif req_obj.optimizer.lower() == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=req_obj.learning_rate, weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer: {req_obj.optimizer}")

        for epoch in range(req_obj.num_iterations):
            epoch_path = globals.TMP_TRAIN_CHECKPOINT_DIR / f"{req_obj.train_job_id}_epoch_{epoch}.pt"
            if req_obj.kind == "standard":                
                train_acc, train_loss = trainer.epoch(train_loader, weight=False, opt=opt)
                test_acc, test_loss = trainer.epoch(test_loader)

                if (req_obj.num_iterations -1 == epoch or cancel_event.is_set()) and req_obj.require_adv_attack_test:   
                    linf_adv_acc, linf_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_linf, weight=False, epsilon=trainer.eps_linf)
                    l2_adv_acc, l2_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_l2, weight=False, epsilon=trainer.eps_l2)
                else:
                    linf_adv_acc, linf_adv_loss = -1, -1
                    l2_adv_acc, l2_adv_loss = -1, -1

                torch.save(model.state_dict(), epoch_path)
                print(f"Saved temporary checkpoint: {epoch_path}")    

            elif req_obj.kind == "adversarial":
                attack = trainer.pgd_linf if req_obj.attack == "PGD-linf" else trainer.pgd_l2
                train_acc, train_loss = trainer.epoch_adv(train_loader, attack=attack, weight=False, opt=opt, epsilon=req_obj.epsilon, alpha = req_obj.alpha  )
                test_acc, test_loss = trainer.epoch(test_loader)

                if (req_obj.num_iterations -1 == epoch or cancel_event.is_set()) and req_obj.require_adv_attack_test:  
                    linf_adv_acc, linf_adv_loss = trainer.epoch_adv(test_loader, trainer.pgd_linf, weight=False, epsilon=trainer.eps_linf)

                else:
                    linf_adv_acc, linf_adv_loss = -1, -1
                    l2_adv_acc, l2_adv_loss = -1, -1

                torch.save(model.state_dict(), epoch_path)
                print(f"Saved temporary checkpoint: {epoch_path}")    

            else:
                print("ulalalalla, no such a kind of train")
            epoch_data = {
                "type": "epoch",
                "epoch": epoch + 1,
                "train_acc": train_acc,
                #"train_loss": train_loss,
                "test_acc": test_acc,
                #"test_loss": test_loss,
                "linf_adv_acc": linf_adv_acc,
                #"linf_adv_loss": linf_adv_loss,
                #"l2_adv_acc": l2_adv_acc,
                #"l2_adv_loss": l2_adv_loss,
            }
            progress_queue.put(epoch_data)
            all_epochs.append(epoch_data)
            time.sleep(0.5) 

            if cancel_event.is_set():
                progress_queue.put({"type": "cancelled"})
                final_result_for_save["status"] = "cancelled"
                break

        progress_queue.put({"type": "done"})

        if final_result_for_save["status"] != "cancelled":
            final_result_for_save["status"] = "done"
        final_result_for_save["epochs"] = all_epochs
        final_result_for_save["timestamp"] = datetime.datetime.now().isoformat()


        if os.path.exists(globals.TRAINING_HISTORY_PATH):
            print("yes, sve a history")
            with open(globals.TRAINING_HISTORY_PATH, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        if not isinstance(data, list):
            data = [data]

        # Append new run
        data.append(final_result_for_save)

        # Save back
        with open(globals.TRAINING_HISTORY_PATH, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved training result for job {final_result_for_save['train_job_id']}")

        try:
            progress_queue.close()
        except:
            pass

    except Exception as e:
        err = {"type": "error", "error": str(e)}
        progress_queue.put(err)
        try:
            progress_queue.close()
        except:
            pass

