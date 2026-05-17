import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from HeirarchicalModel import CustomLoss, TeacherStudentModule
from torch_geometric.loader import DataLoader
from HeirarchicalDataset import HeirarchicalDataset
import numpy as np
from sklearn.metrics import f1_score
from conf import label_to_node

THRESHOLD = 0.5
SMALL_NONZERO = 1e-11

# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = "cpu"

def macro_f1_helper(y_true, y_pred):
    # Both inputs should be integer tensors
    print(f"shapes: {y_true.shape}, {y_pred.shape}")
    y_true = y_true.float()
    y_pred = y_pred.float()

    true_positives      = (y_true * y_pred).count_nonzero(dim=1).double()
    all_positives       = y_true.count_nonzero(dim=1).double()
    predicted_positives = y_pred.count_nonzero(dim=1).double()

    precision = true_positives / (predicted_positives + SMALL_NONZERO)
    recall    = true_positives / (all_positives + SMALL_NONZERO)
    f1        = 2 * precision * recall / (precision + recall + SMALL_NONZERO)

    return f1.mean()


def h_recall_score(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()

    true_positives = (y_true * y_pred).count_nonzero().double()
    all_positives  = y_true.count_nonzero().double()

    return true_positives / all_positives


def h_precision_score(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()

    true_positives = (y_true * y_pred).count_nonzero().double()
    all_results    = y_pred.count_nonzero().double()

    return true_positives / all_results


def hierarchical_f1_helper(y_true, y_pred, beta=1.0):
    hP = h_precision_score(y_true, y_pred)
    hR = h_recall_score(y_true, y_pred)
    hF = (1.0 + beta ** 2) * hP * hR / (beta ** 2 * hP + hR)
    return hP, hR, hF


# --- Metric Functions ---

def macro_f1_student(y_true, y_pred):
    output_divide   = y_pred.shape[1] // 2
    p_student       = y_pred[:, :output_divide]
    p_student_rounded = (p_student > THRESHOLD).int()
    return macro_f1_helper(y_true.int(), p_student_rounded)


def macro_f1_teacher(y_true, y_pred):
    output_divide   = y_pred.shape[1] // 2
    p_teacher       = y_pred[:, output_divide:]
    p_teacher_rounded = (p_teacher > THRESHOLD).int()
    return macro_f1_helper(y_true.int(), p_teacher_rounded)


def hierarchical_f1_student(y_true, y_pred):
    output_divide   = y_pred.shape[1] // 2
    p_student       = y_pred[:, :output_divide]
    p_student_rounded = (p_student > THRESHOLD).int()
    return hierarchical_f1_helper(y_true.int(), p_student_rounded)


def hierarchical_f1_teacher(y_true, y_pred):
    output_divide   = y_pred.shape[1] // 2
    p_teacher       = y_pred[:, output_divide:]
    p_teacher_rounded = (p_teacher > THRESHOLD).int()
    return hierarchical_f1_helper(y_true.int(), p_teacher_rounded)

# train model

def print_metrics(expected_vals, total_preds):
    print(f"macro f1 teacher: {macro_f1_teacher(expected_vals, total_preds)}")
    print(f"macro f1 student: {macro_f1_student(expected_vals, total_preds)}")
    student_hprec, student_hrec, student_hf1 =  hierarchical_f1_student(expected_vals, total_preds)
    teacher_hprec, teacher_hrec, teacher_hf1 =  hierarchical_f1_teacher(expected_vals, total_preds)
    print(f"Student Hierarchial; precision: {student_hprec}, recall: {student_hrec}, f1: {student_hf1}")
    print(f"Teacher Hierarchial; precision: {teacher_hprec}, recall: {teacher_hrec}, f1: {teacher_hf1}")

# Priority from deepest to shallowest among the 7 target classes:
# {performance, endeavor, activity, state, habitual, process, none}.
# When multiple labels are predicted, the deepest (most specific) one wins.
_MULTICLASS_PRIORITY = [
    label_to_node['performance'],
    label_to_node['endeavor'],
    label_to_node['activity'],
    label_to_node['state'],
    label_to_node['habitual'],
    label_to_node['process'],
    label_to_node['none'],
]

# Ancestor sets for each of the 7 target classes (inclusive of self).
# Used for per-sample hierarchical overlap in multi-class hierarchical F1.
_ANCESTORS = {
    label_to_node['performance']: frozenset([label_to_node['performance'], label_to_node['perfective'], label_to_node['process'], label_to_node['aspect']]),
    label_to_node['endeavor']:    frozenset([label_to_node['endeavor'],    label_to_node['perfective'], label_to_node['atelic'], label_to_node['process'], label_to_node['imperfective'], label_to_node['aspect']]),
    label_to_node['activity']:    frozenset([label_to_node['activity'],    label_to_node['atelic'],     label_to_node['process'], label_to_node['imperfective'], label_to_node['aspect']]),
    label_to_node['state']:       frozenset([label_to_node['state'],       label_to_node['imperfective'], label_to_node['aspect']]),
    label_to_node['habitual']:    frozenset([label_to_node['habitual'],    label_to_node['aspect']]),
    label_to_node['process']:     frozenset([label_to_node['process'],     label_to_node['aspect']]),
    label_to_node['none']:        frozenset([label_to_node['none']]),
}

def _multilabel_to_multiclass(pred):
    """
    pred: (N, 11) binary tensor
    Returns (N,) class indices by selecting the deepest predicted label
    among the 7 target classes. Defaults to 'none' if none are predicted.
    """
    N = pred.shape[0]
    result = torch.full((N,), label_to_node['none'], dtype=torch.long)
    assigned = torch.zeros(N, dtype=torch.bool)
    for idx in _MULTICLASS_PRIORITY:
        active = (pred[:, idx] == 1) & ~assigned
        result[active] = idx
        assigned |= active
    return result

def _multilabel_to_multiclass_custom(pred):
    N = pred.shape[0]
    result = torch.full((N,), label_to_node['none'], dtype=torch.long)

def eval_multilabel_f1(y_true, student_pred, teacher_pred):
    """
    y_true:        (N, 11) binary multilabel ground truth
    student_pred:  (N, 11) binary predictions from the student network
    teacher_pred:  (N, 11) binary predictions from the teacher network
    Returns macro F1 across all 11 labels for student and teacher.
    """
    # macro
    student_f1_macro = f1_score(y_true, student_pred, average='macro', zero_division=0)
    teacher_f1_macro = f1_score(y_true, teacher_pred, average='macro', zero_division=0)

    # micro
    student_f1_micro = f1_score(y_true, student_pred, average='micro', zero_division=0)
    teacher_f1_micro = f1_score(y_true, teacher_pred, average='micro', zero_division=0)

    # weighted
    student_f1_weighted = f1_score(y_true, student_pred, average='weighted', zero_division=0)
    teacher_f1_weighted = f1_score(y_true, teacher_pred, average='weighted', zero_division=0)
    return {'student_macro': student_f1_macro, 'teacher_macro': teacher_f1_macro, 'student_micro': student_f1_micro, 'teacher_micro': teacher_f1_micro, 'student_weighted': student_f1_weighted, 'teacher_weighted': teacher_f1_weighted}

def eval_multiclass_f1(y_true_multiclass, student_pred, teacher_pred):
    """
    y_true_multiclass: (N,) integer class indices
    student_pred:      (N, 11) binary predictions from the student network
    teacher_pred:      (N, 11) binary predictions from the teacher network
    Converts multi-label predictions to a single class by picking the deepest
    predicted label among {none, process, performance, endeavor, habitual, state, activity}.
    Returns macro F1 across those 7 classes for student and teacher.
    """
    student_class = _multilabel_to_multiclass(student_pred)
    teacher_class = _multilabel_to_multiclass(teacher_pred)

    # macro
    student_f1_macro = f1_score(y_true_multiclass, student_class, average='macro', zero_division=0)
    teacher_f1_macro = f1_score(y_true_multiclass, teacher_class, average='macro', zero_division=0)

    # micro
    student_f1_micro = f1_score(y_true_multiclass, student_class, average='micro', zero_division=0)
    teacher_f1_micro = f1_score(y_true_multiclass, teacher_class, average='micro', zero_division=0)

    # weighted
    student_f1_weighted = f1_score(y_true_multiclass, student_class, average='weighted', zero_division=0)
    teacher_f1_weighted = f1_score(y_true_multiclass, teacher_class, average='weighted', zero_division=0)

    return {'student_macro': student_f1_macro, 'teacher_macro': teacher_f1_macro, 'student_micro': student_f1_micro, 'teacher_micro': teacher_f1_micro, 'student_weighted': student_f1_weighted, 'teacher_weighted': teacher_f1_weighted}

def eval_multiclass_f1_custom(y_true_multiclass, student_pred, teacher_pred):
    """
    y_true_multiclass: (N,) integer class indices
    student_pred:      (N, 11) binary predictions from the student network
    teacher_pred:      (N, 11) binary predictions from the teacher network
    Converts multi-label predictions to a single class by picking the deepest
    predicted label among {none, process, performance, endeavor, habitual, state, activity}.
    Returns macro F1 across those 7 classes for student and teacher.
    """
    student_class = _multilabel_to_multiclass(student_pred)
    teacher_class = _multilabel_to_multiclass(teacher_pred)

    # macro
    student_f1_macro = f1_score(y_true_multiclass, student_class, average='macro', zero_division=0)
    teacher_f1_macro = f1_score(y_true_multiclass, teacher_class, average='macro', zero_division=0)

    # micro
    student_f1_micro = f1_score(y_true_multiclass, student_class, average='micro', zero_division=0)
    teacher_f1_micro = f1_score(y_true_multiclass, teacher_class, average='micro', zero_division=0)

    # weighted
    student_f1_weighted = f1_score(y_true_multiclass, student_class, average='weighted', zero_division=0)
    teacher_f1_weighted = f1_score(y_true_multiclass, teacher_class, average='weighted', zero_division=0)

    return {'student_macro': student_f1_macro, 'teacher_macro': teacher_f1_macro, 'student_micro': student_f1_micro, 'teacher_micro': teacher_f1_micro, 'student_weighted': student_f1_weighted, 'teacher_weighted': teacher_f1_weighted}

def eval_hierarchical_multiclass_f1(y_true_multiclass, student_pred, teacher_pred):
    """
    Multi-class counterpart to the multi-label hierarchical F1.
    For each sample, computes the overlap between the ancestor sets of the true
    label and the predicted label (inclusive of self), then averages across samples.

        hP_i = |A_true ∩ A_pred| / |A_pred|
        hR_i = |A_true ∩ A_pred| / |A_true|
        hF   = harmonic mean of mean(hP_i) and mean(hR_i)

    Partial credit is given when the predicted label shares ancestors with the
    true label, even if the leaf prediction is wrong.

    y_true_multiclass: (N,) integer class indices
    student_pred:      (N, 11) binary predictions from the student network
    teacher_pred:      (N, 11) binary predictions from the teacher network
    """
    student_class = _multilabel_to_multiclass(student_pred)
    teacher_class = _multilabel_to_multiclass(teacher_pred)

    def _compute(y_true, y_pred_class):
        hP_scores, hR_scores = [], []
        for true_idx, pred_idx in zip(y_true.flatten().tolist(), y_pred_class.tolist()):
            true_idx = int(true_idx)
            pred_idx = int(pred_idx)
            A_true = _ANCESTORS.get(true_idx, frozenset([true_idx]))
            A_pred = _ANCESTORS.get(pred_idx, frozenset([pred_idx]))
            tp = len(A_true & A_pred)
            hP_scores.append(tp / len(A_pred))
            hR_scores.append(tp / len(A_true))
        hP = sum(hP_scores) / len(hP_scores)
        hR = sum(hR_scores) / len(hR_scores)
        hF = 2 * hP * hR / (hP + hR + SMALL_NONZERO)
        return hP, hR, hF

    s_hP, s_hR, s_hF = _compute(y_true_multiclass, student_class)
    t_hP, t_hR, t_hF = _compute(y_true_multiclass, teacher_class)
    return {
        'student': (s_hP, s_hR, s_hF),
        'teacher': (t_hP, t_hR, t_hF),
    }

# def convert_to_multiclass(predictions):
def train(model, dataloader, val_dataloader, optimizer, customLoss):
    steps = 0
    model.train()
    for data in dataloader:
        data = data.to(device)

        optimizer.zero_grad()

        outputs = model(data.x.to(device), data.edge_index.to(device), data.target_node.to(device)).to(device)

        loss = customLoss(outputs, data.y.float(), steps % 100 == 0)
        
        loss.backward()
        optimizer.step()
        steps += 1

    with torch.no_grad():
        model.eval()
        test_predictions = []
        test_values = []
        test_single_values = []
        for test_data in val_dataloader:
            test_data = test_data.to(device)
            out = model(test_data.x.to(device), test_data.edge_index.to(device), test_data.target_node.to(device)).to(device)
            
            test_predictions.append(torch.tensor(out).squeeze())
            test_values.append(torch.tensor(test_data.y).squeeze())
            test_single_values.append(test_data.single_y)

        test_predictions = torch.sigmoid(torch.stack(test_predictions))
        multi_class_test_value = torch.stack(test_single_values)
        test_values = torch.stack(test_values) > 0.5

        student_pred = test_predictions[:, :11] > THRESHOLD
        teacher_pred = test_predictions[:, 11:] > THRESHOLD

        multilabel_scores = eval_multilabel_f1(test_values, student_pred, teacher_pred)
        print(f"multilabel macro f1  — student: {multilabel_scores['student_macro']:.4f}, teacher: {multilabel_scores['teacher_macro']:.4f}")
        print(f"multilabel micro f1  — student: {multilabel_scores['student_micro']:.4f}, teacher: {multilabel_scores['teacher_micro']:.4f}")
        print(f"multilabel weighted f1  — student: {multilabel_scores['student_weighted']:.4f}, teacher: {multilabel_scores['teacher_weighted']:.4f}")

        multiclass_scores = eval_multiclass_f1(multi_class_test_value, student_pred, teacher_pred)
        print(f"multiclass macro f1  — student: {multiclass_scores['student_macro']:.4f}, teacher: {multiclass_scores['teacher_macro']:.4f}")
        print(f"multiclass micro f1  — student: {multiclass_scores['student_micro']:.4f}, teacher: {multiclass_scores['teacher_micro']:.4f}")
        print(f"multiclass weighted f1  — student: {multiclass_scores['student_weighted']:.4f}, teacher: {multiclass_scores['teacher_weighted']:.4f}")

        ## hierarchical f1 (multi-label)
        print("hierarchical f1 (multi-label)")
        print_metrics(test_values, test_predictions)

        ## hierarchical f1 (multi-class)
        hier_mc_scores = eval_hierarchical_multiclass_f1(multi_class_test_value, student_pred, teacher_pred)
        s_hP, s_hR, s_hF = hier_mc_scores['student']
        t_hP, t_hR, t_hF = hier_mc_scores['teacher']
        print(f"hierarchical multiclass f1 — student: precision={s_hP:.4f}, recall={s_hR:.4f}, f1={s_hF:.4f}")
        print(f"hierarchical multiclass f1 — teacher: precision={t_hP:.4f}, recall={t_hR:.4f}, f1={t_hF:.4f}")

        return {
            "val_macro_f1": multilabel_scores['student_macro'],
            "val_micro_f1": multilabel_scores['student_micro'],
            "val_weighted_f1": multilabel_scores['student_weighted'],
            "val_multiclass_macro_f1": multiclass_scores['student_macro'],
            "val_multiclass_micro_f1": multiclass_scores['student_micro'],
            "val_multiclass_weighted_f1": multiclass_scores['student_weighted'],
            
        }

def training():
    # log_file = open("logs/logs.txt", "a")
    dataset = HeirarchicalDataset(root='UMRDataset/', split='train')
    val_dataset = HeirarchicalDataset(root='UMRDataset/', split='val')
    test_dataset = HeirarchicalDataset(root='UMRDataset/', split='test')
    # print("len train dataset", len(dataset))
    # print("val dataset", len(val_dataset))
    dataloader = DataLoader(dataset, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True)

    customLoss = CustomLoss(device).to(device)
    model = TeacherStudentModule(hidden_channels=512, num_hidden_layers=1, lambda_each_rule=7.49, lambda_regularization=0.38).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
    NUM_EPOCHS = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    os.makedirs('model_checkpoints', exist_ok=True)
    best_student_f1 = 0.0
    patience_counter = 0
    PATIENCE = 3

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        eval_scores = train(model, dataloader, val_dataloader, optimizer, customLoss)
        scheduler.step()
        print(f"lr: {scheduler.get_last_lr()[0]:.2e}")
        student_f1 = eval_scores['val_multiclass_macro_f1']

        if student_f1 > best_student_f1:
            best_student_f1 = student_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'student_f1': student_f1,
            }, 'model_checkpoints/best_hierarchical.pth.tar')
            print(f"New best student multilabel F1: {best_student_f1:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping after epoch {epoch}.")
                break

    checkpoint = torch.load('model_checkpoints/best_hierarchical.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (student F1: {checkpoint['student_f1']:.4f})")

    print("In the test set")
    with torch.no_grad():
        model.eval()
        test_predictions = []
        test_values = []
        test_single_values = []
        for test_data in test_dataloader:
            test_data = test_data.to(device)
            out = model(test_data.x.to(device), test_data.edge_index.to(device), test_data.target_node.to(device)).to(device)
            
            test_predictions.append(torch.tensor(out).squeeze())
            test_values.append(torch.tensor(test_data.y).squeeze())
            test_single_values.append(test_data.single_y)

        test_predictions = torch.sigmoid(torch.stack(test_predictions))
        multi_class_test_value = torch.stack(test_single_values)
        test_values = torch.stack(test_values) > 0.5

        student_pred = test_predictions[:, :11] > THRESHOLD
        teacher_pred = test_predictions[:, 11:] > THRESHOLD

        # multilabel_scores = eval_multilabel_f1(test_values, student_pred, teacher_pred)
        # print(f"multilabel f1  — student: {multilabel_scores['student']:.4f}, teacher: {multilabel_scores['teacher']:.4f}")

        # multiclass_scores = eval_multiclass_f1(multi_class_test_value, student_pred, teacher_pred)
        # print(f"multiclass f1  — student: {multiclass_scores['student']:.4f}, teacher: {multiclass_scores['teacher']:.4f}")

        multilabel_scores = eval_multilabel_f1(test_values, student_pred, teacher_pred)
        print(f"multilabel macro f1  — student: {multilabel_scores['student_macro']:.4f}, teacher: {multilabel_scores['teacher_macro']:.4f}")
        print(f"multilabel micro f1  — student: {multilabel_scores['student_micro']:.4f}, teacher: {multilabel_scores['teacher_micro']:.4f}")
        print(f"multilabel weighted f1  — student: {multilabel_scores['student_weighted']:.4f}, teacher: {multilabel_scores['teacher_weighted']:.4f}")

        multiclass_scores = eval_multiclass_f1(multi_class_test_value, student_pred, teacher_pred)
        print(f"multiclass macro f1  — student: {multiclass_scores['student_macro']:.4f}, teacher: {multiclass_scores['teacher_macro']:.4f}")
        print(f"multiclass micro f1  — student: {multiclass_scores['student_micro']:.4f}, teacher: {multiclass_scores['teacher_micro']:.4f}")
        print(f"multiclass weighted f1  — student: {multiclass_scores['student_weighted']:.4f}, teacher: {multiclass_scores['teacher_weighted']:.4f}")

        ## hierarchical f1 (multi-label)
        print("hierarchical f1 (multi-label)")
        print_metrics(test_values, test_predictions)

        ## hierarchical f1 (multi-class)
        hier_mc_scores = eval_hierarchical_multiclass_f1(multi_class_test_value, student_pred, teacher_pred)
        s_hP, s_hR, s_hF = hier_mc_scores['student']
        t_hP, t_hR, t_hF = hier_mc_scores['teacher']
        print(f"hierarchical multiclass f1 — student: precision={s_hP:.4f}, recall={s_hR:.4f}, f1={s_hF:.4f}")
        print(f"hierarchical multiclass f1 — teacher: precision={t_hP:.4f}, recall={t_hR:.4f}, f1={t_hF:.4f}")

if __name__ == "__main__":
    training()