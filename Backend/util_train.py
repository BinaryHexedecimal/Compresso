import torch.nn as nn
import torch.nn.functional as F
import os, json, torch, asyncio
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

import globals
from util_dataset import load_dataset, load_dataset_classes
from src import TR


# Simple convolutional neural network for image classification.
# Adapts input size based on dataset (e.g., MNIST vs. CIFAR).
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.input_size = 28 if in_channels == 1 else 32  # MNIST vs CIFAR/SVHN
        self.fc1 = nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Main async generator to train a model and stream progress.
# Supports standard and adversarial modes, and early cancel.
async def run_training_job(req_obj):
    
    # Load compressed dataset for training
    training_data_path = globals.COMPRESSION_CONTAINER_DIR / f"{req_obj.data_info.get('data_id')}_compressed_data.pt"
    res = torch.load(training_data_path, weights_only=False)
    train_dataset = TensorDataset(res["train_x"], res["train_y"])
    test_dataset = load_dataset(req_obj.data_info.get('dataset_name'), train_=False)

    # Data loaders for batching and multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=min(8, globals.NUM_CPU // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=min(8, globals.NUM_CPU // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # Determine image channels and class count from data
    sample, _ = train_dataset[0]
    _in_channels = sample.shape[0]
    _num_classes = len(load_dataset_classes()[req_obj.data_info["dataset_name"]])
    print(f"Training data: {_in_channels} channels, {_num_classes} classes")
    model = ConvNet(in_channels=_in_channels, num_classes=_num_classes)

    # Auto-select device (CPU or GPU)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Trainer handles training and evaluation logic
    trainer = TR(model, train_loader, test_loader, eps_linf=globals.EPS_LINF, eps_l2=globals.EPS_L2)

    # Choose optimizer
    if req_obj.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=req_obj.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif req_obj.optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=req_obj.learning_rate, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {req_obj.optimizer}")

    all_epochs = []
    yield {"type": "start"}

    for epoch in range(1, req_obj.num_iterations + 1):
        await asyncio.sleep(0.2)
        if globals.ACTIVE_JOBS["training"][req_obj.train_job_id]["cancel"]:
            yield {"type": "cancelled"}
            break

        # Path to save model state after each epoch (temporary checkpoint)
        model_save_path = globals.TMP_TRAIN_CHECKPOINT_DIR / f"{req_obj.train_job_id}_epoch_{epoch}.pt"

        # Training mode selection: standard or adversarial
        if req_obj.kind == "standard":
            train_acc, train_loss = trainer.epoch(train_loader, weight=False, opt=opt)
            test_acc, test_loss = trainer.epoch(test_loader)

            # Detect actual epsilon being used (in case of lists/tensor inputs)
            _epsilon = trainer.eps_linf[0] if isinstance(trainer.eps_linf, (list, tuple)) else float(trainer.eps_linf)

            if epoch == req_obj.num_iterations and req_obj.require_adv_attack_test:
                linf_adv_acc, linf_adv_loss = trainer.epoch_adv(
                    test_loader, trainer.pgd_linf, weight=False, epsilon=_epsilon
                )
            else:
                linf_adv_acc, linf_adv_loss = -1, -1

            torch.save(model.state_dict(), model_save_path)
            print(f"Saved temporary checkpoint: {model_save_path}")

        elif req_obj.kind == "adversarial":
            attack = trainer.pgd_linf if req_obj.attack == "PGD-linf" else trainer.pgd_l2
            train_acc, train_loss = trainer.epoch_adv(
                train_loader,
                attack=attack,
                weight=False,
                opt=opt,
                epsilon=req_obj.epsilon,
                alpha=req_obj.alpha,
            )
            test_acc, test_loss = trainer.epoch(test_loader)

            _epsilon = trainer.eps_linf[0] if isinstance(trainer.eps_linf, (list, tuple)) else float(trainer.eps_linf)

            if epoch == req_obj.num_iterations and req_obj.require_adv_attack_test:
                linf_adv_acc, linf_adv_loss = trainer.epoch_adv(
                    test_loader, trainer.pgd_linf, weight=False, epsilon=_epsilon
                )
            else:
                linf_adv_acc, linf_adv_loss = -1, -1

            torch.save(model.state_dict(), model_save_path)
            print(f"Saved temporary checkpoint: {model_save_path}")
        else:
            print("Invalid train kind")
            continue

        # Stream progress to frontend
        epoch_data = {
            "type": "epoch",
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "linf_adv_acc": linf_adv_acc,
        }
        all_epochs.append(epoch_data)
        yield epoch_data

    # Store full run information to history JSON
    run_info = {
        "status": "cancelled" if globals.ACTIVE_JOBS["training"][req_obj.train_job_id]["cancel"] else "done",
        "train_job_id": req_obj.train_job_id,
        "timestamp": datetime.now().isoformat(),
        "epochs": all_epochs,
        "req_obj": vars(req_obj),
    }

    if os.path.exists(globals.TRAINING_HISTORY_PATH):
        with open(globals.TRAINING_HISTORY_PATH, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    if not isinstance(history, list):
        history = [history]
    history.append(run_info)
    with open(globals.TRAINING_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved training result for job {run_info['train_job_id']}")

    if run_info["status"] == "cancelled":
        yield {"type": "cancelled"}
    else:
        yield {"type": "done"}
