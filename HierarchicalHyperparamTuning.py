import optuna
from optuna.trial import TrialState
import torch
from torch_geometric.loader import DataLoader
from TrainHierarchical import train
from HeirarchicalModel import CustomLoss, TeacherStudentModule
from HeirarchicalDataset import HeirarchicalDataset

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def objective(trial):
    dataset = HeirarchicalDataset(root='UMRDataset/', split='val')
    val_dataset = HeirarchicalDataset(root='UMRDataset/', split='val')

    dataloader = DataLoader(dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=True)

    lambda_each_rule = trial.suggest_float("lambda_each", 0.1, 100.0, log=True)
    regularization_term = trial.suggest_float("lambda_reg", 0.1, 1.0, log=True)

    lr = trial.suggest_float("lr", 1e-7, 1e-3, log=True)
    # hidden_channel = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])
    # num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 6)
    hidden_channel = 512
    num_hidden_layers = 1

    model = TeacherStudentModule(hidden_channels=hidden_channel, num_hidden_layers=num_hidden_layers, lambda_each_rule=lambda_each_rule, lambda_regularization=regularization_term).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    customLoss = CustomLoss(device).to(device)

    for epoch in range(5):
        print("Epoch", epoch)
        eval_scores = train(model, dataloader, val_dataloader, optimizer, customLoss)
        print("eval scores", eval_scores)
        trial.report(eval_scores['val_multiclass_macro_f1'], epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return eval_scores['val_multiclass_macro_f1']

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