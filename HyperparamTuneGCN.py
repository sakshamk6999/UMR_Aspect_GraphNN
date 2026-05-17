import optuna
from optuna.trial import TrialState
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from GraphDataset import MyOwnDataset
from GraphModel import GCNModel
from TrainGCN import train_epoch, evaluate, print_scores, NUM_CLASSES

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

TUNE_EPOCHS = 5
dataset     = MyOwnDataset(root='UMRDataset/', split='train')
val_dataset = MyOwnDataset(root='UMRDataset/', split='val')


def objective(trial):
    lr               = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    hidden_channels  = trial.suggest_categorical("hidden_channels", 512)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)

    dataloader     = DataLoader(dataset,     shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)

    model     = GCNModel(hidden_channels=hidden_channels, num_hidden_layers=num_hidden_layers).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=TUNE_EPOCHS)

    val_f1 = 0.0
    for epoch in range(TUNE_EPOCHS):
        print(f"  epoch {epoch}")
        train_epoch(model, dataloader, optimizer, criterion)
        scheduler.step()

        val_labels, val_preds = evaluate(model, val_dataloader)
        val_f1 = print_scores("val", val_labels, val_preds)

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    pruned_trials   = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics:")
    print(f"  Finished trials:  {len(study.trials)}")
    print(f"  Pruned trials:    {len(pruned_trials)}")
    print(f"  Complete trials:  {len(complete_trials)}")

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value: {best.value:.4f}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
