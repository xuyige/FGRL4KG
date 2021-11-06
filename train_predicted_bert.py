import torch
import torch.nn as nn
import transformers as trf
import fastNLP as fnlp
import os


class BertPredictModel(nn.Module):
    
    def __init__(self, from_pretrained: str, vocab=None):
        super().__init__()
        self.vocab = vocab
        # self.bert = fnlp.embeddings.BertEmbedding(vocab, from_pretrained)
        self.tokenizer = trf.BertTokenizer.from_pretrained(from_pretrained)
        self.bert = trf.BertModel.from_pretrained(from_pretrained)
        self.predict = nn.Linear(768, 3)
        self.activate = nn.Sigmoid()
        self.check_device = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_str, target_str=None, score1=None, score2=None):
        inputs = self.tokenizer(f'{target_str} [SEP] {pred_str}', return_tensors="pt")
        inputs = {n: p.to(self.check_device.device) for n, p in inputs.items()}
        outputs = self.bert(**inputs)
        predicted_logits = self.predict(outputs[1]).squeeze(-1)
        # predicted_logits = self.predict(outputs.pooler_output).squeeze(-1)
        predicted_score = self.activate(predicted_logits)

        loss = None
        if score1 is not None:
            score_tensor = torch.tensor([[score1, score2, (score1 + score2) / 2]]).to(self.check_device.device)
            loss = self.mse_loss(predicted_score, score_tensor)

        return predicted_score, loss
        # if not isinstance(pred_str, str):
        #     outputs = self.bert(pred_str)[:, 0]
        #     predicted_score = self.activate(self.predict(outputs).squeeze(-1))
        #     return predicted_score, None
        #
        # input_str = f'{target_str} [SEP] {pred_str}'
        # input_tensor = torch.tensor([[self.vocab.to_index(w) for w in input_str.split(' ')]]).to(self.check_device.device)
        # outputs = self.bert(input_tensor)
        # predicted_logits = self.predict(outputs[:,0]).squeeze(-1)
        # bert_predicted_score = self.activate(predicted_logits)
        # model_loss = None
        # if score1 is not None:
        #     score_tensor = torch.tensor([[score1, score2, (score1 + score2) / 2]]).to(self.check_device.device)
        #     model_loss = self.mse_loss(bert_predicted_score, score_tensor)
        #
        # return bert_predicted_score, model_loss
    
    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        vocab = kwargs.pop('vocab', None)
        model = cls(model_dir_or_name, vocab)
        if not os.path.exists(os.path.join(model_dir_or_name, 'predict_tensor.pkl')):
            print(f'only load pretrain model from `{model_dir_or_name}`!')
            return model
        with open(os.path.join(model_dir_or_name, 'predict_tensor.pkl'), 'rb') as f:
            predict_state_dict = torch.load(f, map_location='cpu')
        model.predict.load_state_dict(predict_state_dict)
        print(f'load predict parameters from `{model_dir_or_name}`!')
        return model
        
    def save_pretrained(self, model_dir_or_name, *inputs, **kwargs):
        if not os.path.exists(model_dir_or_name):
            os.makedirs(model_dir_or_name)
        self.bert.save_pretrained(model_dir_or_name)
        with open(os.path.join(model_dir_or_name, 'predict_tensor.pkl'), 'wb') as f:
            torch.save(self.predict.state_dict(), f)

            
# if __name__ == '__main__':
#     with open('/remote-home/ygxu/workspace/KG/KGM/score_data.txt', 'r') as f:
#         lines = f.readlines()
#     # lines = lines[:1000]
#
#     model = BertPredictModel("bert-base-uncased")
#     model = model.cuda()
#
#     for n, p in model.named_parameters():
#         if 'bert' in n:
#             p.requires_grad = True
#
#     optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
#     update_every = 16
#
#
#     from datetime import datetime
#     import random
#     import numpy as np
#
#
#     seed = 42
#
#     # set random seed
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     n_gpu = torch.cuda.device_count()
#     if n_gpu > 0:
#         torch.cuda.manual_seed_all(seed)
#
#     save_root_dir = '/remote-home/ygxu/workspace/KG/KGM/BERT/new-bert-base-uncased'
#     total_steps = 0
#     total_epoch = 1
#     print_steps = 500
#     save_every = 5000
#     neg_example = []
#     total_neg = 0
#     total_pos = 0
#     now_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
#     print(f'start traing at time {now_time}')
#     print(f'{"="*20}')
#     try:
#         for epoch in range(total_epoch):
#             steps = 0
#             avg_loss = 0
#             optim.zero_grad()
#             for idx, line in enumerate(lines):
#                 target_str, pred_str, score1, score2 = line.strip().split('\t')
#                 if float(score1) == 0 and float(score2) == 0:
#                     if len(neg_example) == 0:
#                         neg_example.append((target_str, pred_str, score1, score2))
#                         continue
#                     elif neg_example[0][1] != pred_str:
#                         rand_idx = random.randint(0, len(neg_example) - 1)
#                         neg_example.append((target_str, pred_str, score1, score2))
#                         target_str, pred_str, score1, score2 = neg_example[rand_idx]
#                         neg_example = neg_example[-1:]
#                         total_neg += 1
#                     else:
#                         neg_example.append((target_str, pred_str, score1, score2))
#                         continue
#                 else:
#                     total_pos += 1
#                 predicted_score, loss = model(pred_str, target_str, float(score1), float(score2))
#                 if loss is not None:
#                     if float(score1) == 0 and float(score2) == 0:
#                         loss = loss / 2
#                     loss = loss / update_every
#                     avg_loss += loss.item()
#                     loss.backward()
#                     steps += 1
#                     if steps >= update_every or idx == len(lines) - 1:
#                         steps = 0
#                         optim.step()
#                         optim.zero_grad()
#                         total_steps += 1
#                         if total_steps % print_steps == 0 or total_steps == 1:
#                             avg_loss /= print_steps
#                             now_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
#                             print(f'[{now_time}][epoch {epoch + 1} of {total_epoch}; step {total_steps}: loss {round(avg_loss, 6)}]')
#                             print(f'[target_str: `{target_str}`; pred_str: `{pred_str}`; predicted_score: {predicted_score};]')
#                             print(f'[idx: {idx}][score1: {round(float(score1), 4)}; score2: {round(float(score2), 4)}]')
#                             print(f'[total_neg: {total_neg}; total_pos: {total_pos}]')
#                             print(f'{"="*20}')
#                             avg_loss = 0
#                         if total_steps % save_every == 0:
#                             model.save_pretrained(f'{save_root_dir}-{total_steps // 1000}k')
#                             print(f'save pretrained to dir `{save_root_dir}-{total_steps // 1000}k`.')
#     except BaseException as e:
#         print(f'exit with exception `{e.__class__.__name__}`.')
#         print(f'[{str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))}]: have update for {total_steps}')
#
#     model.save_pretrained(f'{save_root_dir}-{total_steps // 1000}k')
#     print(f'save pretrained to dir `{save_root_dir}-{total_steps // 1000}k`.')

if __name__ == '__main__':
    with open('/remote-home/ygxu/workspace/KG/KGM/score_data_new.txt', 'r') as f:
        lines = f.readlines()
    lines = lines[:10000]

    save_root_dir = '/remote-home/ygxu/workspace/KG/KGM/BERT/new-bert-base-uncased-50k'
    model = BertPredictModel.from_pretrained(save_root_dir)
    model = model.cuda()
    model.eval()

    for n, p in model.named_parameters():
        if 'bert' in n:
            p.requires_grad = False


    from datetime import datetime
    import random
    import numpy as np


    now_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
    print(f'start at time {now_time}')
    print(f'{"=" * 20}')
    stat_0 = [0] * 12
    stat_1 = [0] * 12
    stat_2 = [0] * 12
    stat_3 = [0] * 12
    with torch.no_grad():
        with open('/remote-home/ygxu/workspace/KG/KGM/human_evaluate.txt', 'w') as f:
            for idx, line in enumerate(lines):
                target_str, pred_str = line.strip().split('\t')
                predicted_score, loss = model(pred_str, target_str)  # [1,3]
                if len(predicted_score.size()) == 2:
                    predicted_score = predicted_score.squeeze(0)
                score_list = predicted_score.tolist()
                # if score_list[0] == 0.:
                #     stat_0[0] += 1
                # elif 0 < score_list[0] <= 0.1:
                #     stat_0[1] += 1
                # elif 0 < score_list[0] <= 0.2:
                #     stat_0[2] += 1
                # elif 0 < score_list[0] <= 0.3:
                #     stat_0[3] += 1
                # elif 0 < score_list[0] <= 0.4:
                #     stat_0[4] += 1
                # elif 0 < score_list[0] <= 0.5:
                #     stat_0[5] += 1
                # elif 0 < score_list[0] <= 0.6:
                #     stat_0[6] += 1
                # elif 0 < score_list[0] <= 0.7:
                #     stat_0[7] += 1
                # elif 0 < score_list[0] <= 0.8:
                #     stat_0[8] += 1
                # elif 0 < score_list[0] <= 0.9:
                #     stat_0[9] += 1
                # elif 0 < score_list[0] < 1.:
                #     stat_0[10] += 1
                # elif score_list[0] == 1.:
                #     stat_0[11] += 1
                #
                # if score_list[1] == 0.:
                #     stat_1[0] += 1
                # elif 0 < score_list[1] <= 0.1:
                #     stat_1[1] += 1
                # elif 0 < score_list[1] <= 0.2:
                #     stat_1[2] += 1
                # elif 0 < score_list[1] <= 0.3:
                #     stat_1[3] += 1
                # elif 0 < score_list[1] <= 0.4:
                #     stat_1[4] += 1
                # elif 0 < score_list[1] <= 0.5:
                #     stat_1[5] += 1
                # elif 0 < score_list[1] <= 0.6:
                #     stat_1[6] += 1
                # elif 0 < score_list[1] <= 0.7:
                #     stat_1[7] += 1
                # elif 0 < score_list[1] <= 0.8:
                #     stat_1[8] += 1
                # elif 0 < score_list[1] <= 0.9:
                #     stat_1[9] += 1
                # elif 0 < score_list[1] < 1.:
                #     stat_1[10] += 1
                # elif score_list[1] == 1.:
                #     stat_1[11] += 1
                #
                #
                # if score_list[2] == 0.:
                #     stat_2[0] += 1
                # elif 0 < score_list[2] <= 0.1:
                #     stat_2[1] += 1
                # elif 0 < score_list[2] <= 0.2:
                #     stat_2[2] += 1
                # elif 0 < score_list[2] <= 0.3:
                #     stat_2[3] += 1
                # elif 0 < score_list[2] <= 0.4:
                #     stat_2[4] += 1
                # elif 0 < score_list[2] <= 0.5:
                #     stat_2[5] += 1
                # elif 0 < score_list[2] <= 0.6:
                #     stat_2[6] += 1
                # elif 0 < score_list[2] <= 0.7:
                #     stat_2[7] += 1
                # elif 0 < score_list[2] <= 0.8:
                #     stat_2[8] += 1
                # elif 0 < score_list[2] <= 0.9:
                #     stat_2[9] += 1
                # elif 0 < score_list[2] < 1.:
                #     stat_2[10] += 1
                # elif score_list[2] == 1.:
                #     stat_2[11] += 1
                #
                # if (score_list[0] + score_list[1]) / 2 == 0.:
                #     stat_3[0] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.1:
                #     stat_3[1] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.2:
                #     stat_3[2] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.3:
                #     stat_3[3] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.4:
                #     stat_3[4] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.5:
                #     stat_3[5] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.6:
                #     stat_3[6] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.7:
                #     stat_3[7] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.8:
                #     stat_3[8] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 <= 0.9:
                #     stat_3[9] += 1
                # elif 0 < (score_list[0] + score_list[1]) / 2 < 1.:
                #     stat_3[10] += 1
                # elif (score_list[0] + score_list[1]) / 2 == 1.:
                #     stat_3[11] += 1


                if score_list[2]>0.05 and score_list[2] < 0.95:
                    f_str = f'{target_str}\t{pred_str}\t{score_list[2]}\n'
                    f.write(f_str)

    print(str(stat_0))
    print(str(stat_1))
    print(str(stat_2))
    print(str(stat_3))
    
    
    
    