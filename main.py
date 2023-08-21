from data_loaders import surreal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import tqdm
import common.arg_parser

from models import VideoPose3D_model


def main(args):
    # file = os.path.join(datasetDir,'sequenceFiles',seq_name+'.pkl')
    # dataset = pickle.load(f)

    ds_train = surreal.SurrealDataset("../../dataset/SURREAL_v1/cmu/", split="train")
    ds_test = surreal.SurrealDataset("../../dataset/SURREAL_v1/cmu/", split="test")
    num_joints_in = ds_train.num_joints

    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=64, shuffle=True, num_workers=0
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=64, shuffle=False, num_workers=0
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_hpe = VideoPose3D_model.TemporalModel().to(device)

    criterion = nn.CrossEntropyLoss()

    # TODO
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    train_losses_per_epoch = []
    test_losses_per_epoch = []
    accuracies_per_epoch = []

    EPOCHS = 10
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}/{EPOCHS - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model_ft.train()  # Set model_ft to training mode
            else:
                model_ft.eval()  # Set model_ft to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, target in tqdm.tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                target = target.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    # target.shape = (batch_size)
                    # outputs.shape = (batch_size, num_classes)
                    # The performance of this criterion is generally better when target contains
                    # class indices, as this allows for optimized computation.
                    loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == target).sum().item()
            if phase == "train":
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects * 100.0 / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                test_losses_per_epoch.append(epoch_loss)
                accuracies_per_epoch.append(epoch_acc)
            else:
                train_losses_per_epoch.append(epoch_loss)

            # deep copy the model_ft
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    torch.save(
        {
            "model_state_dict": best_model_wts,
            "optimizer_state_dict": optimizer_ft.state_dict(),
        },
        "checkpoints/" + MODEL_NAME + ".pch",
    )

    np.savez(
        "logs/" + MODEL_NAME,
        train_losses_per_epoch=train_losses_per_epoch,
        test_losses_per_epoch=test_losses_per_epoch,
        accuracies_per_epoch=accuracies_per_epoch,
    )


if __name__ == "__main__":
    args = common.arg_parser.parse_args()
    main(args)
