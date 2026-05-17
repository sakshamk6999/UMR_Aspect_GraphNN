from GraphDataset import MyOwnDataset
from GraphModel import GCNModel
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from torcheval.metrics.functional import multiclass_f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def one_hot_ce_loss(outputs, targets, criterion, should_print):
    # criterion = nn.CrossEntropyLoss()
    # print("shape", outputs.shape)
    _, outputLabels = torch.max(outputs, dim=0)
    _, labels = torch.max(targets, dim=0)
    if should_print:
      print(outputLabels, labels)
    return criterion(outputLabels.float(), labels.float())

label_mapping = {
  "process": 0,
  "performance": 1,
  "endeavor": 2,
  "habitual": 3,
  "state": 4,
  "activity": 5,
  "none": 6,
  "complete": 7,
  "ended": 8
}

class ImplicationRule(torch.nn.Module):
    def __init__(self, label1, label2, condition):
      super(ImplicationRule, self).__init__()
      self.predicate = label_mapping[label1]
      self.conclusion = label_mapping[label2]
      self.condition = condition

    def forward(self, output):
      # print("shape", output.shape)
      # print("probs", torch.sigmoid(output.squeeze(0))[self.predicate], torch.sigmoid(output.squeeze(0))[self.conclusion])
      if self.condition:
        implication = torch.clamp((1 - torch.sigmoid(output.squeeze(0))[self.predicate]) + torch.sigmoid(output.squeeze(0))[self.conclusion], max=1.0)

        return implication
      else:
        implication = torch.clamp((1 - torch.sigmoid(output.squeeze(0))[self.predicate]) + (1 - torch.sigmoid(output.squeeze(0))[self.conclusion]), max=1.0)
        return implication
      
class CustomLossRules(torch.nn.Module):
    def __init__(self, hyperparam = 0.01):
      super(CustomLossRules, self).__init__()
      self.crossEntropy = torch.nn.CrossEntropyLoss()
      self.completionLoss = torch.nn.BCEWithLogitsLoss()
      self.resultLoss = torch.nn.BCEWithLogitsLoss()
      self.implicationRules = [
          ImplicationRule("performance", "complete", True),
          ImplicationRule("performance", "ended", True),
          ImplicationRule("endeavor", "complete", True),
          ImplicationRule("endeavor", "ended", False),
          ImplicationRule("activity", "complete", False),
      ]

      self.hyperparam = hyperparam

    def forward(self, output, target):
      # print("output shape", output.shape)
      # print("target shape", target.shape)
      loss = self.crossEntropy(output[:, :7].squeeze(0), target[:7].float())
      completion_loss = self.completionLoss(output[:, 7].squeeze(0), target[7].float())
      result_loss = self.resultLoss(output[:, 8].squeeze(0), target[8].float())

      implication_loss = None

      for rule in self.implicationRules:
        # print("implication loss", rule(output))
        if implication_loss == None:
          implication_loss = rule(output)
        else:
          implication_loss += rule(output)
        # loss += rule(output)
      # print("implication loss", loss - ce_loss)
      return loss +  completion_loss + result_loss + (self.hyperparam * implication_loss), {
          "cross_entropy": loss,
          "completion_loss": completion_loss,
          "result_loss": result_loss,
          "implication_loss": implication_loss
      }

def get_dataset_summary(dataset):
  label_list = [
    "process",
    "performance",
    "endeavor",
    "habitual",
    "state",
    "activity",
    "none"
  ]

  rec = {}
  total = 0
  for d in dataset:
    # print("label shape", d.y.shape)
    label = label_list[torch.argmax(d.y[:7])]
    # print('label',label )
    # break
    rec[label] = rec.get(label, 0) + 1
    total += 1
  
  for k in rec:
    print("label", k)
    print("percentage", rec[k])

# def get_count_raw(true_values, predictions):

def train(model, dataloader, val_dataloader, optimizer, customLoss):
  model.train()
  training_predictions = []
  training_values = []

  training_completion_predictions = []
  training_completion_values = []

  training_result_predictions = []
  training_result_values = []

  n = 0
  for data in dataloader:
    data = data.to(device)

    optimizer.zero_grad()  # Clear gradients.
    tensor_index = torch.where(data.target_node > 0)
    # print(data.input_ids)
    out = model(data.x, data.edge_index, data.target_node)  # Perform a single forward pass.
    # customLoss(out.squeeze(0), data.y.float())
    # print("data shape", data.x.shape,"out shape", out.shape, "data y shape", data.y.shape)
    # print("output shape", out, data.y.float())
    # loss = criterion1(out[:, 6].squeeze(0), data.y[6].float()) + criterion2(out[:, 7].squeeze(0), data.y[7].float())
    # cross_entropy(out[:, :6].squeeze(0), data.y[:6].float()) +  + criterion2(out[:, 7].squeeze(0), data.y[7].float())
    # loss = cross_entropy(out[:, :6].squeeze(0), data.y[:6].float())
    # bce1 = criterion1(out[:, 6].squeeze(0), data.y[6].float())
    # bce2 = criterion2(out[:, 7].squeeze(0), data.y[7].float())

    # totalLoss = loss + bce1 + bce2
    totalLoss, lossVals = customLoss(out, data.y)

    training_predictions.append(out[:, :7].squeeze(0))
    training_values.append(data.y[:7].float())

    training_completion_predictions.append(out[:, 7].squeeze(0))
    training_completion_values.append(data.y[7].float())

    training_result_predictions.append(out[:, 8].squeeze(0))
    training_result_values.append(data.y[8].float())
    # print("loss", loss)
    totalLoss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    # if n % 300 == 0:
    #   # print("\tlosses", loss, bce1, bce2)
    #   print("\tlosses", lossVals)
    #   print("\ttotal_loss", totalLoss.item())
    n += 1
    # return loss
  model.eval()
  with torch.no_grad():
    # print("training_values shape", torch.stack(training_values).shape, torch.stack(training_predictions).shape)

    #multiclass_f1_score(input, target, num_classes=2, average="macro")
    training_predictions = torch.softmax(torch.stack(training_predictions), dim=1)
    training_predictions = torch.argmax(training_predictions, dim=1)
    training_values = torch.stack(training_values)

    completion_predictions = torch.sigmoid(torch.stack(training_completion_predictions)).cpu().numpy()
    completion_values = torch.stack(training_completion_values).cpu().numpy()

    result_predictions = torch.sigmoid(torch.stack(training_result_predictions)).cpu().numpy()
    result_values = torch.stack(training_result_values).cpu().numpy()

    # f1_macro = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='macro')
    # f1_micro = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='micro')
    # f1_weighted = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='weighted')
    
    f1_macro = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='macro')
    f1_micro = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='micro')
    f1_weighted = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='weighted')

    print("macro_f1", precision_recall_fscore_support(torch.argmax(training_values, dim=1), training_predictions, average='macro'))
    print("micro_f1", precision_recall_fscore_support(torch.argmax(training_values, dim=1), training_predictions, average='micro'))
    print("weighted_f1", precision_recall_fscore_support(torch.argmax(training_values, dim=1), training_predictions, average='micro'))

    completion_f1_macro = f1_score(completion_values, np.round(completion_predictions), average='macro')
    completion_f1_micro = f1_score(completion_values, np.round(completion_predictions), average='micro')
    completion_f1_weighted = f1_score(completion_values, np.round(completion_predictions), average='weighted')
    print("completion_macro_f1", completion_f1_macro)
    print("completion_micro_f1", completion_f1_micro)
    print("completion_weighted_f1", completion_f1_weighted)

    result_f1_macro = f1_score(result_values, np.round(result_predictions), average='macro')
    result_f1_micro = f1_score(result_values, np.round(result_predictions), average='micro')
    result_f1_weighted = f1_score(result_values, np.round(result_predictions), average='weighted')
    print("result_macro_f1", result_f1_macro)
    print("result_micro_f1", result_f1_micro)
    print("result_f1_weighted", result_f1_weighted)

    print("validation")

    val_predictions = []
    val_values = []

    val_completion_predictions = []
    val_completion_values = []

    val_result_predictions = []
    val_result_values = []

    for val_data in val_dataloader:
      val_data = val_data.to(device)
      tensor_index = torch.where(val_data.target_node > 0)
      # print(data.input_ids)
      out = model(val_data.x, val_data.edge_index, val_data.target_node)

      val_predictions.append(out[:, :7].squeeze(0))
      val_values.append(val_data.y[:7].float())

      val_completion_predictions.append(out[:, 7].squeeze(0))
      val_completion_values.append(val_data.y[7].float())

      val_result_predictions.append(out[:, 8].squeeze(0))
      val_result_values.append(val_data.y[8].float())

    val_predictions = torch.softmax(torch.stack(val_predictions), dim=1)
    val_predictions = torch.argmax(val_predictions, dim=1)
    val_values = torch.stack(val_values)

    val_completion_predictions = torch.sigmoid(torch.stack(val_completion_predictions)).cpu().numpy()
    val_completion_values = torch.stack(val_completion_values).cpu().numpy()

    val_result_predictions = torch.sigmoid(torch.stack(val_result_predictions)).cpu().numpy()
    val_result_values = torch.stack(val_result_values).cpu().numpy()

    # val_f1_macro = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='macro')

    # val_f1_micro = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='micro')
    # val_f1_weighted = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='weighted')

    val_f1_macro = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='macro')

    val_f1_micro = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='micro')
    val_f1_weighted = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='weighted')

    print("val_macro_f1", val_f1_macro)
    print("val_micro_f1", val_f1_micro)
    print("val_weighted_f1", val_f1_weighted)
    val_completion_f1_macro = f1_score(val_completion_values, np.round(val_completion_predictions), average='macro')
    val_completion_f1_micro = f1_score(val_completion_values, np.round(val_completion_predictions), average='micro')
    val_completion_f1_weighted = f1_score(val_completion_values, np.round(val_completion_predictions), average='weighted')
    print("val_completion_macro_f1", val_completion_f1_macro)
    print("val_completion_micro_f1", val_completion_f1_micro)
    print("val_completion_weighted_f1", val_completion_f1_weighted)

    val_result_f1_macro = f1_score(val_result_values, np.round(val_result_predictions), average='macro')
    val_result_f1_micro = f1_score(val_result_values, np.round(val_result_predictions), average='micro')
    val_result_f1_weighted = f1_score(val_result_values, np.round(val_result_predictions), average='weighted')
    print("val_result_macro_f1", val_result_f1_macro)
    print("val_result_micro_f1", val_result_f1_micro)
    print("val_result_weighted_f1", val_result_f1_weighted)

    return {
    'train_macro_f1': f1_macro,
    'val_macro_f1': val_f1_macro,
    'val_micro_f1': val_f1_micro,
    'val_weighted_f1': val_f1_weighted,
    }

def main():
  dataset = MyOwnDataset(root='UMRDataset/', split='train')
  val_dataset = MyOwnDataset(root='UMRDataset/', split='val')
  test_dataset = MyOwnDataset(root='UMRDataset/', split='test')
  # print("len train dataset", len(dataset))
  # print("val dataset", len(val_dataset))
  dataloader = DataLoader(dataset, shuffle=True)
  val_dataloader = DataLoader(val_dataset, shuffle=True)
  test_dataloader = DataLoader(test_dataset, shuffle=True)

  print("train summary")
  print(get_dataset_summary(dataloader))
  print("val summary")
  print(get_dataset_summary(val_dataloader))
  print("test summary")
  print(get_dataset_summary(test_dataloader))

  # model = GCNModel(hidden_channels=512, num_hidden_layers=1).to(device)
  model = GCNModel(hidden_channels=512, num_hidden_layers=1).to(device)
  total_params = sum(p.numel() for p in model.parameters())

  print("Total params are", total_params)

  optimizer = torch.optim.AdamW(model.parameters(), lr=5.9e-6)
  customLoss = CustomLossRules(hyperparam=0.003).to(device)

  # optimizer = torch.optim.AdamW(model.parameters(), lr=1.3e-5)
  # customLoss = CustomLossRules(hyperparam=0.71).to(device)

  scheduler = CosineAnnealingLR(optimizer, T_max=15)

  val_scores = []
  patience = 3
  best_score = None

  # def train():
  #   model.train()
  #   training_predictions = []
  #   training_values = []

  #   training_completion_predictions = []
  #   training_completion_values = []

  #   training_result_predictions = []
  #   training_result_values = []

  #   n = 0
  #   for data in dataloader:
  #     data = data.to(device)

  #     optimizer.zero_grad()  # Clear gradients.
  #     tensor_index = torch.where(data.target_node > 0)
  #     # print(data.input_ids)
  #     out = model(data.x, data.edge_index, data.target_node)  # Perform a single forward pass.
  #     # customLoss(out.squeeze(0), data.y.float())
  #     # print("data shape", data.x.shape,"out shape", out.shape, "data y shape", data.y.shape)
  #     # print("output shape", out, data.y.float())
  #     # loss = criterion1(out[:, 6].squeeze(0), data.y[6].float()) + criterion2(out[:, 7].squeeze(0), data.y[7].float())
  #     # cross_entropy(out[:, :6].squeeze(0), data.y[:6].float()) +  + criterion2(out[:, 7].squeeze(0), data.y[7].float())
  #     # loss = cross_entropy(out[:, :6].squeeze(0), data.y[:6].float())
  #     # bce1 = criterion1(out[:, 6].squeeze(0), data.y[6].float())
  #     # bce2 = criterion2(out[:, 7].squeeze(0), data.y[7].float())

  #     # totalLoss = loss + bce1 + bce2
  #     totalLoss, lossVals = customLoss(out, data.y)

  #     training_predictions.append(out[:, :7].squeeze(0))
  #     training_values.append(data.y[:7].float())

  #     training_completion_predictions.append(out[:, 7].squeeze(0))
  #     training_completion_values.append(data.y[7].float())

  #     training_result_predictions.append(out[:, 8].squeeze(0))
  #     training_result_values.append(data.y[8].float())
  #     # print("loss", loss)
  #     totalLoss.backward()  # Derive gradients.
  #     optimizer.step()  # Update parameters based on gradients.

  #     if n % 300 == 0:
  #       # print("\tlosses", loss, bce1, bce2)
  #       print("\tlosses", lossVals)
  #       print("\ttotal_loss", totalLoss.item())
  #     n += 1
  #     # return loss
  #   model.eval()
  #   with torch.no_grad():
  #     # print("training_values shape", torch.stack(training_values).shape, torch.stack(training_predictions).shape)

  #     #multiclass_f1_score(input, target, num_classes=2, average="macro")
  #     training_predictions = torch.softmax(torch.stack(training_predictions), dim=1)
  #     training_predictions = torch.argmax(training_predictions, dim=1)
  #     training_values = torch.stack(training_values)

  #     completion_predictions = torch.sigmoid(torch.stack(training_completion_predictions)).cpu().numpy()
  #     completion_values = torch.stack(training_completion_values).cpu().numpy()

  #     result_predictions = torch.sigmoid(torch.stack(training_result_predictions)).cpu().numpy()
  #     result_values = torch.stack(training_result_values).cpu().numpy()

  #     # f1_macro = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='macro')
  #     # f1_micro = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='micro')
  #     # f1_weighted = multiclass_f1_score(training_values, training_predictions, num_classes=7, average='weighted')
      
  #     f1_macro = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='macro')
  #     f1_micro = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='micro')
  #     f1_weighted = f1_score(torch.argmax(training_values, dim=1), training_predictions, average='weighted')

  #     print("macro_f1", f1_macro)
  #     print("micro_f1", f1_micro)
  #     print("weighted_f1", f1_weighted)

  #     completion_f1_macro = f1_score(completion_values, np.round(completion_predictions), average='macro')
  #     completion_f1_micro = f1_score(completion_values, np.round(completion_predictions), average='micro')
  #     # completion_f1_weighted = f1_score(completion_values, np.round(completion_predictions), average='weighted')
  #     print("completion_macro_f1", completion_f1_macro)
  #     print("completion_micro_f1", completion_f1_micro)

  #     result_f1_macro = f1_score(result_values, np.round(result_predictions), average='macro')
  #     result_f1_micro = f1_score(result_values, np.round(result_predictions), average='micro')
  #     print("result_macro_f1", result_f1_macro)
  #     print("result_micro_f1", result_f1_micro)

  #     print("validation")

  #     val_predictions = []
  #     val_values = []

  #     val_completion_predictions = []
  #     val_completion_values = []

  #     val_result_predictions = []
  #     val_result_values = []

  #     for val_data in val_dataloader:
  #       val_data = val_data.to(device)
  #       tensor_index = torch.where(val_data.target_node > 0)
  #       # print(data.input_ids)
  #       out = model(val_data.x, val_data.edge_index, val_data.target_node)

  #       val_predictions.append(out[:, :7].squeeze(0))
  #       val_values.append(val_data.y[:7].float())

  #       val_completion_predictions.append(out[:, 7].squeeze(0))
  #       val_completion_values.append(val_data.y[7].float())

  #       val_result_predictions.append(out[:, 8].squeeze(0))
  #       val_result_values.append(val_data.y[8].float())

  #     val_predictions = torch.softmax(torch.stack(val_predictions), dim=1)
  #     val_predictions = torch.argmax(val_predictions, dim=1)
  #     val_values = torch.stack(val_values)

  #     val_completion_predictions = torch.sigmoid(torch.stack(val_completion_predictions)).cpu().numpy()
  #     val_completion_values = torch.stack(val_completion_values).cpu().numpy()

  #     val_result_predictions = torch.sigmoid(torch.stack(val_result_predictions)).cpu().numpy()
  #     val_result_values = torch.stack(val_result_values).cpu().numpy()

  #     # val_f1_macro = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='macro')

  #     # val_f1_micro = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='micro')
  #     # val_f1_weighted = multiclass_f1_score(val_values, val_predictions, num_classes=7, average='weighted')

  #     val_f1_macro = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='macro')

  #     val_f1_micro = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='micro')
  #     val_f1_weighted = f1_score(torch.argmax(val_values, dim=1), val_predictions, average='weighted')

  #     print("val_macro_f1", val_f1_macro)
  #     print("val_micro_f1", val_f1_micro)
  #     print("val_weighted_f1", val_f1_weighted)
  #     val_completion_f1_macro = f1_score(val_completion_values, np.round(val_completion_predictions), average='macro')
  #     val_completion_f1_micro = f1_score(val_completion_values, np.round(val_completion_predictions), average='micro')
  #     print("val_completion_macro_f1", val_completion_f1_macro)
  #     print("val_completion_micro_f1", val_completion_f1_micro)

  #     val_result_f1_macro = f1_score(val_result_values, np.round(val_result_predictions), average='macro')
  #     val_result_f1_micro = f1_score(val_result_values, np.round(val_result_predictions), average='micro')
  #     print("val_result_macro_f1", val_result_f1_macro)
  #     print("val_result_micro_f1", val_result_f1_micro)

  #     return {
  #     'train_macro_f1': f1_macro,
  #     'val_macro_f1': val_f1_macro
  #   }
  
  def test():
    model.eval()

    test_predictions = []
    test_values = []

    test_completion_predictions = []
    test_completion_values = []

    test_result_predictions = []
    test_result_values = []

    for test_data in test_dataloader:
      test_data = test_data.to(device)
      tensor_index = torch.where(test_data.target_node > 0)
      # print(test_data)
      out = model(test_data.x, test_data.edge_index, test_data.target_node)

      test_predictions.append(out[:, :7].squeeze(0))
      test_values.append(test_data.y[:7].float())

      test_completion_predictions.append(out[:, 7].squeeze(0))
      test_completion_values.append(test_data.y[7].float())

      test_result_predictions.append(out[:, 8].squeeze(0))
      test_result_values.append(test_data.y[8].float())
    

    test_predictions = torch.softmax(torch.stack(test_predictions), dim=1)
    test_predictions = torch.argmax(test_predictions, dim=1)
    test_values = torch.stack(test_values)

    test_completion_predictions = torch.sigmoid(torch.stack(test_completion_predictions)).cpu().numpy()
    test_completion_values = torch.stack(test_completion_values).cpu().numpy()

    test_result_predictions = torch.sigmoid(torch.stack(test_result_predictions)).cpu().numpy()
    test_result_values = torch.stack(test_result_values).cpu().numpy()

    # test_f1_macro = multiclass_f1_score(test_values, test_predictions, num_classes=7, average='macro')

    # test_f1_micro = multiclass_f1_score(test_values, test_predictions, num_classes=7, average='micro')
    # test_f1_weighted = multiclass_f1_score(test_values, test_predictions, num_classes=7, average='weighted')

    test_f1_macro = f1_score(torch.argmax(test_values, dim=1), test_predictions, average='macro')

    test_f1_micro = f1_score(torch.argmax(test_values, dim=1), test_predictions, average='micro')
    test_f1_weighted = f1_score(torch.argmax(test_values, dim=1), test_predictions, average='weighted')

    print("test_macro_f1", precision_recall_fscore_support(torch.argmax(test_values, dim=1), test_predictions, average='macro'))
    print("test_micro_f1", precision_recall_fscore_support(torch.argmax(test_values, dim=1), test_predictions, average='micro'))
    print("test_weighted_f1", precision_recall_fscore_support(torch.argmax(test_values, dim=1), test_predictions, average='weighted'))
    test_completion_f1_macro = f1_score(test_completion_values, np.round(test_completion_predictions), average='macro')
    test_completion_f1_micro = f1_score(test_completion_values, np.round(test_completion_predictions), average='micro')
    print("test_completion_macro_f1", test_completion_f1_macro)
    print("test_completion_micro_f1", test_completion_f1_micro)

    test_result_f1_macro = f1_score(test_result_values, np.round(test_result_predictions), average='macro')
    test_result_f1_micro = f1_score(test_result_values, np.round(test_result_predictions), average='micro')
    print("test_result_macro_f1", test_result_f1_macro)
    print("test_result_micro_f1", test_result_f1_micro)

    class_names = ["process", "performance", "endeavor", "habitual", "state", "activity", "none"]
    y_true = torch.argmax(test_values, dim=1).cpu().numpy()
    y_pred = test_predictions.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Confusion matrix saved to confusion_matrix.png")

    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv("confusion_matrix.csv")
    print("Confusion matrix saved to confusion_matrix.csv")

  for epoch in range(1, 20):
    print("Epoch", epoch)
    eval_scores = train(model, dataloader, val_dataloader, optimizer, customLoss)

    if best_score == None or eval_scores['val_macro_f1'] > best_score:
      patience = 3
      best_score = eval_scores['val_macro_f1']
      val_scores.append(eval_scores)
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_score': eval_scores['train_macro_f1'],
        'val_score': eval_scores['val_macro_f1']
      }, f'./model_checkpoints/epoch_{epoch}.pth.tar')

    elif eval_scores['val_macro_f1'] <= best_score:
      patience -= 1

    if patience == 0:
      print(f"Early stopping point reached at epoch {epoch}")
      break
    scheduler.step()
    print("lr", scheduler.get_last_lr())
    print("patience", patience)

    
  print("===== Now Evaluating on Test =====")
  with torch.no_grad():
    test()

if __name__ == "__main__":
    main()