import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from GraphDataset import MyOwnDataset
from GraphModel import GCNModel

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

CLASS_NAMES = ["process", "performance", "endeavor", "habitual", "state", "activity", "none"]
NUM_CLASSES = len(CLASS_NAMES)
NUM_EPOCHS = 20
PATIENCE = 3


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.target_node)   # (1, 9)
        logits = out[:, :NUM_CLASSES]                             # (1, 7)
        label = torch.argmax(data.y[:NUM_CLASSES].float()).unsqueeze(0)  # (1,)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.target_node)
            pred = torch.argmax(out[:, :NUM_CLASSES], dim=1)
            label = torch.argmax(data.y[:NUM_CLASSES].float()).unsqueeze(0)
            all_preds.append(pred)
            all_labels.append(label)
    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return labels, preds


def print_scores(split, labels, preds):
    macro = f1_score(labels, preds, average='macro', zero_division=0)
    micro = f1_score(labels, preds, average='micro', zero_division=0)
    weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    print(f"[{split}] macro={macro:.4f}  micro={micro:.4f}  weighted={weighted:.4f}")
    return macro


def save_confusion_matrix(labels, preds, prefix="test"):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(f"{prefix.capitalize()} Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {prefix}_confusion_matrix.png")

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(f"{prefix}_confusion_matrix.csv")
    print(f"Confusion matrix CSV saved to {prefix}_confusion_matrix.csv")


def main():
    dataset      = MyOwnDataset(root='UMRDataset/', split='train')
    val_dataset  = MyOwnDataset(root='UMRDataset/', split='val')
    test_dataset = MyOwnDataset(root='UMRDataset/', split='test')

    dataloader      = DataLoader(dataset,      shuffle=True)
    val_dataloader  = DataLoader(val_dataset,  shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    model     = GCNModel(hidden_channels=512, num_hidden_layers=1).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.3e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")

    os.makedirs('model_checkpoints', exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}")
        train_epoch(model, dataloader, optimizer, criterion)
        scheduler.step()
        print(f"lr: {scheduler.get_last_lr()[0]:.2e}")

        train_labels, train_preds = evaluate(model, dataloader)
        print_scores("train", train_labels, train_preds)

        val_labels, val_preds = evaluate(model, val_dataloader)
        val_f1 = print_scores("val", val_labels, val_preds)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, 'model_checkpoints/best_gcn.pth.tar')
            print(f"New best val macro F1: {best_val_f1:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping after epoch {epoch}.")
                break

    print("\n===== Evaluating on Test Set =====")
    checkpoint = torch.load('model_checkpoints/best_gcn.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (val F1: {checkpoint['val_f1']:.4f})")

    test_labels, test_preds = evaluate(model, test_dataloader)
    print_scores("test", test_labels, test_preds)
    save_confusion_matrix(test_labels, test_preds, prefix="test")


if __name__ == "__main__":
    main()
