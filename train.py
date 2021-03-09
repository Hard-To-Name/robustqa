import argparse
import json
import os
from collections import OrderedDict
import torch
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter

from model import DomainQA
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args
from prepare_data import get_dataset
from pytorch_pretrained_bert.optimization import BertAdam

from tqdm import tqdm


def get_opt(param_optimizer, num_train_optimization_steps, lr, warmup_proportion):
    """
    Hack to remove pooler, which is not used
    Thus it produce None grad that break apex
    """
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return BertAdam(optimizer_grouped_parameters,
                    lr=lr,
                    warmup=warmup_proportion,
                    t_total=num_train_optimization_steps)

# TODO: use a logger, use tensorboard
class Trainer():
  def __init__(self, args, log):
    self.lr = args.lr
    self.discriminator_lr = args.discriminator_lr
    self.num_epochs = args.num_epochs
    self.device = args.device
    self.eval_every = args.eval_every
    self.path = os.path.join(args.save_dir, 'checkpoint')
    self.num_visuals = args.num_visuals
    self.save_dir = args.save_dir
    self.log = log
    self.visualize_predictions = args.visualize_predictions
    self.enable_discriminator = args.adv
    self.model = DomainQA('bert-base-uncased', num_classes=6, hidden_size=768, num_layers=3, dropout=0.1, dis_lambda=args.discriminator_lambda, concat=False, anneal=False)
    qa_params = list(self.model.bert.named_parameters()) + list(self.model.qa_outputs.named_parameters())
    dis_params = list(self.model.discriminator.named_parameters())
    self.qa_optimizer = get_opt(qa_params, 45000, args.lr, args.warmup_proportion)
    self.dis_optimizer = get_opt(dis_params, 45000, args.discriminator_lr, args.warmup_proportion)
    if not os.path.exists(self.path):
      os.makedirs(self.path)

  def save(self, model):
    model.save_pretrained(self.path)

  def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
    device = self.device

    model.eval()
    pred_dict = {}
    all_start_logits = []
    all_end_logits = []
    with torch.no_grad(), \
      tqdm(total=len(data_loader.dataset)) as progress_bar:
      for batch in data_loader:
        # Setup for forward
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size = len(input_ids)
        start_logits, end_logits = model(input_ids, attention_mask)
        # outputs = model(input_ids, attention_mask=attention_mask)
        # # Forward
        # start_logits, end_logits = outputs.start_logits, outputs.end_logits
        # TODO: compute loss

        all_start_logits.append(start_logits)
        all_end_logits.append(end_logits)
        progress_bar.update(batch_size)

    # Get F1 and EM scores
    start_logits = torch.cat(all_start_logits).cpu().numpy()
    end_logits = torch.cat(all_end_logits).cpu().numpy()
    preds = util.postprocess_qa_predictions(data_dict,
                                            data_loader.dataset.encodings,
                                            (start_logits, end_logits))
    if split == 'validation':
      results = util.eval_dicts(data_dict, preds)
      results_list = [('F1', results['F1']),
                      ('EM', results['EM'])]
    else:
      results_list = [('F1', -1.0),
                      ('EM', -1.0)]
    results = OrderedDict(results_list)
    if return_preds:
      return preds, results
    return results

  # def compute_discriminator_loss(self, hidden_states):
  #   """
  #   Computes the loss for discriminator based on the hidden states of the DistillBERT model.
  #   Original paper implementation: https://github.com/seanie12/mrqa/blob/master/model.py
  #   https://huggingface.co/transformers/_modules/transformers/models/distilbert/modeling_distilbert.html#DistilBertForQuestionAnswering
  #   Input: last layer hidden states of the distillBERT model, with shape [batch_size, sequence_length, hidden_dim]
  #
  #   :return: loss from discriminator.
  #   """
  #   cls_embedding = hidden_states[:, 0]
  #   log_prob = self.discriminator(cls_embedding)
  #   targets = torch.ones_like(log_prob) * (1 / self.discriminator.num_classes)
  #   # print('discriminator loss : ', log_prob, targets)
  #   kl_criterion = nn.KLDivLoss(reduction="batchmean")
  #   return kl_criterion(log_prob, targets)
  #
  # def forward_discriminator(self, hidden_states, data_set_ids):
  #   cls_embedding = hidden_states[:, 0]
  #   # detach the embedding making sure it's not updated from discriminator
  #   log_prob = self.discriminator(cls_embedding.detach())
  #   # print('forward discriminator : ', log_prob, data_set_ids)
  #   criterion = nn.NLLLoss()
  #   loss = criterion(log_prob, data_set_ids)
  #
  #   return loss

  @staticmethod
  def cal_running_avg_loss(loss, running_avg_loss, decay=0.99):
      if running_avg_loss == 0:
          return loss
      else:
          running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
          return running_avg_loss

  def train(self, model, train_dataloader, eval_dataloader, val_dict):
    device = self.device

    self.model.to(device)

    global_idx = 0
    avg_qa_loss = 0
    avg_dis_loss = 0
    best_scores = {'F1': -1.0, 'EM': -1.0}
    tbx = SummaryWriter(self.save_dir)

    for epoch_num in range(self.num_epochs):
      self.log.info(f'Epoch: {epoch_num}')
      with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
        for batch in train_dataloader:
          self.model.train()

          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          start_positions = batch['start_positions'].to(device)
          end_positions = batch['end_positions'].to(device)
          data_set_ids = batch['data_set_id'].to(device)

          qa_loss = self.model(input_ids, attention_mask,
                               start_positions, end_positions, data_set_ids,
                               dtype="qa",
                               global_step=global_idx)
          qa_loss = qa_loss.mean()
          qa_loss.backward()

          # update qa model
          avg_qa_loss = self.cal_running_avg_loss(qa_loss.item(), avg_qa_loss)
          self.qa_optimizer.step()
          self.qa_optimizer.zero_grad()

          # update discriminator
          dis_loss = self.model(input_ids, attention_mask,
                                start_positions, end_positions, data_set_ids, dtype="dis",
                                global_step=global_idx)
          dis_loss = dis_loss.mean()
          dis_loss.backward()
          avg_dis_loss = self.cal_running_avg_loss(dis_loss.item(), avg_dis_loss)
          self.dis_optimizer.step()
          self.dis_optimizer.zero_grad()

          progress_bar.update(len(input_ids))
          progress_bar.set_postfix(epoch=epoch_num, NLL=qa_loss.item(), diss_loss=dis_loss.item())
          tbx.add_scalar('train/NLL', qa_loss.item(), global_idx)
          tbx.add_scalar('train/dis_loss', dis_loss.item(), global_idx)
          if (global_idx % self.eval_every) == 0 and global_idx > 0:
            self.log.info(f'Evaluating at step {global_idx}...')
            preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
            results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
            self.log.info('Visualizing in TensorBoard...')
            for k, v in curr_score.items():
              tbx.add_scalar(f'val/{k}', v, global_idx)
            self.log.info(f'Eval {results_str}')
            if self.visualize_predictions:
              util.visualize(tbx,
                             pred_dict=preds,
                             gold_dict=val_dict,
                             step=global_idx,
                             split='val',
                             num_visuals=self.num_visuals)
            if curr_score['F1'] >= best_scores['F1']:
              best_scores = curr_score
              self.save(model)
          global_idx += 1
    return best_scores

def main():
  # define parser and arguments
  args = get_train_test_args()

  util.set_seed(args.seed)
  model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

  if args.do_train:
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    else:
      args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
    log = util.get_logger(args.save_dir, 'log_train')
    log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
    log.info("Preparing Training Data...")
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(args, log)
    train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
    log.info("Preparing Validation Data...")
    val_dataset, val_dict = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=SequentialSampler(val_dataset))
    best_scores = trainer.train(model, train_loader, val_loader, val_dict)
  if args.do_eval:
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    split_name = 'test' if 'test' in args.eval_dir else 'validation'
    log = util.get_logger(args.save_dir, f'log_{split_name}')
    trainer = Trainer(args, log)
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
    model.to(args.device)
    eval_dataset, eval_dict = get_dataset(args, args.eval_datasets, args.eval_dir, tokenizer, split_name)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size,
                             sampler=SequentialSampler(eval_dataset))
    eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                               eval_dict, return_preds=True,
                                               split=split_name)
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
    log.info(f'Eval {results_str}')
    # Write submission file
    sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
      csv_writer = csv.writer(csv_fh, delimiter=',')
      csv_writer.writerow(['Id', 'Predicted'])
      for uuid in sorted(eval_preds):
        csv_writer.writerow([uuid, eval_preds[uuid]])

if __name__ == '__main__':
  main()
