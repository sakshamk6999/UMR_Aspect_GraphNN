import torch.nn as nn
import torch
from transformers import AutoModel
from GraphModel import GCNModel
import torch.nn.functional as F
from conf import label_to_node

class Hierarchy_Rule(nn.Module):
    def __init__(self, num_classes, parent_index, child_index, nature='positive'):
        super(Hierarchy_Rule, self).__init__()
        self.num_classes = num_classes
        self.parent_index = parent_index
        self.child_index = child_index

        # One-hot encodings as buffers so they move with the model device
        parent_encoding = torch.zeros(num_classes)
        parent_encoding[parent_index] = 1.0
        self.register_buffer('parent_encoding', parent_encoding)

        child_encoding = torch.zeros(num_classes)
        child_encoding[child_index] = 1.0
        self.register_buffer('child_encoding', child_encoding)

        self.nature = nature

    def rule_evaluation(self, p_student):
        # p_student: (n, k)
        # Equivalent to tf.tensordot(..., 1) along last axis → shape (n,)
        child_proj  = p_student @ self.child_encoding   # (n,)
        parent_proj = p_student @ self.parent_encoding  # (n,)
        return torch.clamp((1 - child_proj) + parent_proj, max=1.0)

    def rule_evaluation_neg(self, p_student):
        # p_student: (n, k)
        # Equivalent to tf.tensordot(..., 1) along last axis → shape (n,)
        child_proj  = p_student @ self.child_encoding   # (n,)
        parent_proj = p_student @ self.parent_encoding  # (n,)
        return torch.clamp((1 - child_proj) + (1 - parent_proj), max=1.0)

    def log_distribution(self, p_student, regularization_term, confidence_val):
        # log_dist and rule_perf start as all-ones, same shape as p_student
        log_dist  = torch.ones_like(p_student)   # (n, k)
        rule_perf = torch.ones_like(p_student)   # (n, k)

        if self.nature == "positive":
            rule_eval = self.rule_evaluation(p_student)  # (n,)
        else:
           rule_eval = self.rule_evaluation_neg(p_student)
        # Scatter rule_eval into the child column of log_dist
        log_dist[:, self.child_index] = rule_eval

        test_val     = -float(regularization_term) * float(confidence_val)
        sub_test_val = rule_perf - log_dist          # (n, k)
        output       = test_val * sub_test_val       # (n, k)

        return output

class TeacherNetwork(nn.Module):
    def __init__(self, lamba_each_rule=100.0, lambda_regularization=1):
        super(TeacherNetwork, self).__init__()
        self.len_node = len(label_to_node)
        
        self.rules = nn.ModuleList([
            Hierarchy_Rule(self.len_node, label_to_node["aspect"], label_to_node['process']),
            Hierarchy_Rule(self.len_node, label_to_node["aspect"], label_to_node['habitual']),
            Hierarchy_Rule(self.len_node, label_to_node["aspect"], label_to_node['imperfective']),
            Hierarchy_Rule(self.len_node, label_to_node["process"], label_to_node['perfective']),
            Hierarchy_Rule(self.len_node, label_to_node["process"], label_to_node['atelic']),
            Hierarchy_Rule(self.len_node, label_to_node["imperfective"], label_to_node['atelic']),
            Hierarchy_Rule(self.len_node, label_to_node["imperfective"], label_to_node['state']),
            Hierarchy_Rule(self.len_node, label_to_node["perfective"], label_to_node['performance']),
            Hierarchy_Rule(self.len_node, label_to_node["perfective"], label_to_node['endeavor']),
            Hierarchy_Rule(self.len_node, label_to_node["atelic"], label_to_node['activity']),
            Hierarchy_Rule(self.len_node, label_to_node["atelic"], label_to_node['endeavor']),
            # Hierarchy_Rule(self.len_node, label_to_node["process"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["performance"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["endeavor"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["habitual"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["activity"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["aspect"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["imperfective"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["perfective"], label_to_node['none'], "negative"),
            # Hierarchy_Rule(self.len_node, label_to_node["atelic"], label_to_node['none'], "negative")
        ])

        self.register_buffer('rule_lambdas', torch.full((len(self.rules),), lamba_each_rule))
        self.regularization_term = lambda_regularization

    def forward(self, inputs):
        student_probs = inputs[0]
        rule_distr = self.calculate_rule_constraints(
            student_probs, self.rules, self.rule_lambdas, self.regularization_term
        )
        rule_adj_probs = student_probs * rule_distr
        return rule_adj_probs

    def calculate_rule_constraints(self, input, rules, rule_confidences, C):
        distr_total = torch.zeros_like(input)
        for i, rule in enumerate(rules):
            distr = rule.log_distribution(input, C, rule_confidences[i])
            distr_total = distr_total + distr
        distr_total = torch.clamp(distr_total, min=-60, max=60)
        return torch.exp(distr_total)    

class TeacherStudentModule(nn.Module):
    def __init__(self, hidden_channels=512, MODEL_DIM=4096, num_hidden_layers=2, lambda_each_rule=100.0, lambda_regularization=1):
        super(TeacherStudentModule, self).__init__()
        self.teacher = TeacherNetwork(lambda_each_rule, lambda_regularization)
        self.student = GCNModel(hidden_channels, MODEL_DIM, num_hidden_layers)
        self.final_linear = torch.nn.Linear(11, 7)

    def forward(self, x, edge_index, token_index):
        student_outputs = self.student(x, edge_index, token_index)
        teacher_outputs = self.teacher([student_outputs])
        final_answer = self.final_linear(student_outputs)
        return torch.cat((student_outputs, teacher_outputs, final_answer), dim=1)

class CustomLoss(nn.Module):
  def __init__(self, device):
    super(CustomLoss, self).__init__()
    self.BCE = nn.BCEWithLogitsLoss()
    self.KLDiv = F.kl_div
    self.teacher_loss = 0
    self.student_loss = 0
    self.device = device
    self.num_classes = len(label_to_node)

  def get_loss(self):
    return self.teacher_loss, self.student_loss

  def forward(self, logits, labels, should_print):
    # print("the shapes", logits.shape, labels.shape)
    # print("logits require grad", logits.requires_grad)
    student_logits = logits[:, :self.num_classes]
    student_logits_sigmoid = torch.sigmoid(student_logits)
    regularized_logits = torch.sigmoid(logits[:, self.num_classes:2*self.num_classes])
    # final_logits = torch.softmax(logits[:,2*self.num_classes:], dim=1)

    bceLoss = self.BCE(student_logits, labels.float())
    totalKldLoss = torch.tensor(0.0).to(self.device)
    self.student_loss = bceLoss
    # print("student_logits_sigmoid", student_logits_sigmoid.requires_grad)
    # print("student_logits", student_logits.shape, "regularized_logits", regularized_logits.shape)
    for i in range(student_logits.shape[0]):
      for j in range(self.num_classes):
        p = student_logits_sigmoid[i, j]
        q = regularized_logits[i, j]
        log_p = torch.stack([p, 1 - p]).log()
        log_q = torch.stack([q, 1 - q]).log()
        temp = self.KLDiv(log_p, log_q, reduction="batchmean", log_target=True)
        # print("temp", temp.requires_grad)
        totalKldLoss += temp

    totalKldLoss = totalKldLoss / (self.num_classes * student_logits.shape[0])

    self.teacher_loss = totalKldLoss

    if should_print:
      print("bceLoss", bceLoss, "kldLoss", totalKldLoss)

    return bceLoss + totalKldLoss

class CustomLossWithCE(nn.Module):
  def __init__(self, device, lambda_loss=1):
    super(CustomLoss, self).__init__()
    self.BCE = nn.BCEWithLogitsLoss()
    self.CE = nn.CrossEntropyLoss()
    self.KLDiv = F.kl_div
    self.reg_loss = lambda_loss
    self.teacher_loss = 0
    self.student_loss = 0
    self.device = device
    self.num_classes = len(label_to_node)

  def get_loss(self):
    return self.teacher_loss, self.student_loss

  def forward(self, logits, labels, true_label, should_print):
    # print("the shapes", logits.shape, labels.shape)
    # print("logits require grad", logits.requires_grad)
    student_logits = logits[:, :self.num_classes]
    student_logits_sigmoid = torch.sigmoid(student_logits)
    regularized_logits = torch.sigmoid(logits[:, self.num_classes:2*self.num_classes])
    final_logits = logits[:,2*self.num_classes:]

    bceLoss = self.BCE(student_logits, labels.float())
    totalKldLoss = torch.tensor(0.0).to(self.device)
    self.student_loss = bceLoss
    student_ce_loss = self.CE(final_logits, true_label)
    # print("student_logits_sigmoid", student_logits_sigmoid.requires_grad)
    # print("student_logits", student_logits.shape, "regularized_logits", regularized_logits.shape)
    for i in range(student_logits.shape[0]):
      for j in range(self.num_classes):
        p = student_logits_sigmoid[i, j]
        q = regularized_logits[i, j]
        log_p = torch.stack([p, 1 - p]).log()
        log_q = torch.stack([q, 1 - q]).log()
        temp = self.KLDiv(log_p, log_q, reduction="batchmean", log_target=True)
        # print("temp", temp.requires_grad)
        totalKldLoss += temp

    totalKldLoss = totalKldLoss / (self.num_classes * student_logits.shape[0])

    self.teacher_loss = totalKldLoss

    if should_print:
      print("bceLoss", bceLoss, "kldLoss", totalKldLoss, "CE loss", student_ce_loss)

    return bceLoss + totalKldLoss + student_ce_loss