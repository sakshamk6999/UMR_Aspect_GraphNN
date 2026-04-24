import os
import optuna
from optuna.trial import TrialState
from GraphDataset import MyOwnDataset
from GraphModel import GCNModel
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score
from Train import one_hot_ce_loss, ImplicationRule, label_mapping, CustomLossRules, train
from torch.optim.lr_scheduler import CosineAnnealingLR

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def objective(trial):
    dataset = MyOwnDataset(root='UMRDataset/', split='train')
    val_dataset = MyOwnDataset(root='UMRDataset/', split='val')


    dataloader = DataLoader(dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=True)

    lr = trial.suggest_float("lr", 1e-9, 1e-1, log=True)
    hidden_channel = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)
    loss_hyperparam = trial.suggest_float("lambda", 1e-4, 1, log=True)

    model = GCNModel(hidden_channels=hidden_channel, num_hidden_layers=num_hidden_layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    customLoss = CustomLossRules(hyperparam=loss_hyperparam).to(device)

    for epoch in range(5):
        print("Epoch", epoch)
        eval_scores = train(model, dataloader, val_dataloader, optimizer, customLoss)
        print("eval scores", eval_scores)
        trial.report(eval_scores['val_macro_f1'], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return eval_scores['val_macro_f1']

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))