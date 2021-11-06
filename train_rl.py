import torch
import numpy as np
import pykp.io
import torch.nn as nn
from utils.statistics import RewardStatistics
from utils.time_log import time_since
import time
from sequence_generator import SequenceGenerator
from utils.report import export_train_and_valid_loss, export_train_and_valid_reward
import sys
import logging
import os
from evaluate import evaluate_reward
from pykp.reward import *
from train_predicted_bert import  BertPredictModel
import math

import fastNLP as fnlp

EPS = 1e-8

def train_model(model, optimizer_ml, optimizer_rl, criterion, train_data_loader, valid_data_loader, opt):
    total_batch = -1
    early_stop_flag = False

    report_train_reward_statistics = RewardStatistics()
    total_train_reward_statistics = RewardStatistics()
    report_train_reward = []
    report_valid_reward = []
    best_valid_reward = float('-inf')
    num_stop_increasing = 0
    init_perturb_std = opt.init_perturb_std
    final_perturb_std = opt.final_perturb_std
    perturb_decay_factor = opt.perturb_decay_factor
    perturb_decay_mode = opt.perturb_decay_mode

    if opt.train_from:  # opt.train_from:
        #TODO: load the training state
        raise ValueError("Not implemented the function of load from trained model")
        pass

    generator = SequenceGenerator(model,
                                  bos_idx=opt.word2idx[pykp.io.BOS_WORD],
                                  eos_idx=opt.word2idx[pykp.io.EOS_WORD],
                                  pad_idx=opt.word2idx[pykp.io.PAD_WORD],
                                  peos_idx=opt.word2idx[pykp.io.PEOS_WORD],
                                  beam_size=1,
                                  max_sequence_length=opt.max_length,
                                  copy_attn=opt.copy_attention,
                                  coverage_attn=opt.coverage_attn,
                                  review_attn=opt.review_attn,
                                  cuda=opt.gpuid > -1
                                  )

    model.train()

    ###
    #bert score
    ###
    fnlp_vocab = fnlp.Vocabulary(padding=None, unknown=None)
    fnlp_vocab.add_word_lst(opt.word2idx.keys())
    fnlp_vocab.padding = '<pad>'
    fnlp_vocab.unknown = '<unk>'

    bert_dir = '/remote-home/ygxu/workspace/KG/KGM/BERT/new-bert-base-uncased-50k'
    bert = BertPredictModel.from_pretrained(bert_dir, vocab=fnlp_vocab)
    bert.eval()

    ###

    for epoch in range(opt.start_epoch, opt.epochs+1):
        if early_stop_flag:
            break

        # TODO: progress bar
        # progbar = Progbar(logger=logging, title='Training', target=len(train_data_loader), batch_size=train_data_loader.batch_size,total_examples=len(train_data_loader.dataset.examples))
        for batch_i, batch in enumerate(train_data_loader):
            total_batch += 1
            if perturb_decay_mode == 0:  # do not decay
                perturb_std = init_perturb_std
            elif perturb_decay_mode == 1:  # exponential decay
                perturb_std = final_perturb_std + (init_perturb_std - final_perturb_std) * math.exp(-1. * total_batch * perturb_decay_factor)
            elif perturb_decay_mode == 2:  # steps decay
                perturb_std = init_perturb_std * math.pow(perturb_decay_factor, math.floor((1+total_batch)/4000))

            #no bert
            #batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt, perturb_std)
            #bert
            batch_reward_stat, log_selected_token_dist = train_one_batch(batch, generator, optimizer_rl, opt,
                                                                         perturb_std, bert)
            report_train_reward_statistics.update(batch_reward_stat)
            total_train_reward_statistics.update(batch_reward_stat)

            # Checkpoint, decay the learning rate if validation loss stop dropping, apply early stopping if stop decreasing for several epochs.
            # Save the model parameters if the validation loss improved.
            if total_batch % 4000 == 0:
                from datetime import datetime
                now_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
                print(f'[{now_time}]Epoch {epoch}; batch: {batch_i}; total batch: {total_batch}')
                sys.stdout.flush()

            if epoch >= opt.start_checkpoint_at:
                if (opt.checkpoint_interval == -1 and batch_i == len(train_data_loader) - 1) or \
                        (opt.checkpoint_interval > -1 and total_batch > 1 and total_batch % opt.checkpoint_interval == 0):

                    valid_reward_stat = evaluate_reward(valid_data_loader, generator, opt, bert=bert)
                    model.train()
                    current_valid_reward = valid_reward_stat.reward()
                    print("Enter check point!")
                    sys.stdout.flush()

                    current_train_reward = report_train_reward_statistics.reward()
                    current_train_pg_loss = report_train_reward_statistics.loss()

                    if current_valid_reward > best_valid_reward:
                        print("Valid reward increases")
                        sys.stdout.flush()
                        best_valid_reward = current_valid_reward
                        num_stop_increasing = 0

                        check_pt_model_path = os.path.join(opt.model_path, '%s.epoch=%d.batch=%d.total_batch=%d' % (
                            opt.exp, epoch, batch_i, total_batch) + '.model')
                        torch.save(  # save model parameters
                            model.state_dict(),
                            open(check_pt_model_path, 'wb')
                        )
                        logging.info('Saving checkpoint to %s' % check_pt_model_path)
                    else:
                        print("Valid reward does not increase")
                        sys.stdout.flush()
                        num_stop_increasing += 1
                        # decay the learning rate by the factor specified by opt.learning_rate_decay
                        if opt.learning_rate_decay_rl:
                            for i, param_group in enumerate(optimizer_rl.param_groups):
                                old_lr = float(param_group['lr'])
                                new_lr = old_lr * opt.learning_rate_decay
                                if old_lr - new_lr > EPS:
                                    param_group['lr'] = new_lr

                    logging.info('Epoch: %d; batch idx: %d; total batches: %d' % (epoch, batch_i, total_batch))
                    logging.info(
                        'avg training reward: %.4f; avg training loss: %.4f; avg validation reward: %.4f; best validation reward: %.4f' % (
                            current_train_reward, current_train_pg_loss, current_valid_reward, best_valid_reward))

                    report_train_reward.append(current_train_reward)
                    report_valid_reward.append(current_valid_reward)

                    if not opt.disable_early_stop_rl:
                        if num_stop_increasing >= opt.early_stop_tolerance:
                            logging.info('Have not increased for %d check points, early stop training' % num_stop_increasing)
                            early_stop_flag = True
                            break
                    report_train_reward_statistics.clear()

    # export the training curve
    train_valid_curve_path = opt.exp_path + '/train_valid_curve'
    export_train_and_valid_reward(report_train_reward, report_valid_reward, opt.checkpoint_interval, train_valid_curve_path)


def train_one_batch(one2many_batch, generator, optimizer, opt, perturb_std=0, bert=None):
    src, src_lens, src_mask, src_oov, oov_lists, src_str_list, trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, title, title_oov, title_lens, title_mask = one2many_batch
    """
    src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
    src_lens: a list containing the length of src sequences for each batch, with len=batch
    src_mask: a FloatTensor, [batch, src_seq_len]
    src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
    oov_lists: a list of oov words for each src, 2dlist
    """

    one2many = opt.one2many
    one2many_mode = opt.one2many_mode
    if one2many and one2many_mode > 1:
        num_predictions = opt.num_predictions
    else:
        num_predictions = 1

    # move data to GPU if available
    src = src.to(opt.device)
    src_mask = src_mask.to(opt.device)
    src_oov = src_oov.to(opt.device)
    bert = bert.to(opt.device)
    # trg = trg.to(opt.device)
    # trg_mask = trg_mask.to(opt.device)
    # trg_oov = trg_oov.to(opt.device)

    if opt.title_guided:
        title = title.to(opt.device)
        title_mask = title_mask.to(opt.device)
        #title_oov = title_oov.to(opt.device)

    optimizer.zero_grad()

    eos_idx = opt.word2idx[pykp.io.EOS_WORD]
    delimiter_word = opt.delimiter_word
    batch_size = src.size(0)
    topk = opt.topk
    reward_type = opt.reward_type
    reward_shaping = opt.reward_shaping
    baseline = opt.baseline
    match_type = opt.match_type
    regularization_type = opt.regularization_type
    regularization_factor = opt.regularization_factor

    if regularization_type == 2:
        entropy_regularize = True
    else:
        entropy_regularize = False

    if opt.perturb_baseline:
        baseline_perturb_std = perturb_std
    else:
        baseline_perturb_std = 0

    #generator.model.train()

    # sample a sequence from the model
    # sample_list is a list of dict, {"prediction": [], "scores": [], "attention": [], "done": True}, prediction is a list of 0 dim tensors
    # log_selected_token_dist: size: [batch, output_seq_len]
    start_time = time.time()
    sample_list, log_selected_token_dist, output_mask, pred_eos_idx_mask, entropy, location_of_eos_for_each_batch, location_of_peos_for_each_batch = generator.sample(
        src, src_lens, src_oov, src_mask, oov_lists, opt.max_length, greedy=False, one2many=one2many,
        one2many_mode=one2many_mode, num_predictions=num_predictions, perturb_std=perturb_std, entropy_regularize=entropy_regularize, title=title, title_lens=title_lens, title_mask=title_mask)
    pred_str_2dlist = sample_list_to_str_2dlist(sample_list, oov_lists, opt.idx2word, opt.vocab_size, eos_idx, delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                              src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
    sample_time = time_since(start_time)
    max_pred_seq_len = log_selected_token_dist.size(1)

    if entropy_regularize:
        entropy_array = entropy.data.cpu().numpy()
    else:
        entropy_array = None

    # if use self critical as baseline, greedily decode a sequence from the model
    if baseline == 'self':
        generator.model.eval()
        with torch.no_grad():
            start_time = time.time()
            greedy_sample_list, _, _, greedy_eos_idx_mask, _, _, _ = generator.sample(src, src_lens, src_oov, src_mask,
                                                                             oov_lists, opt.max_length,
                                                                             greedy=True, one2many=one2many,
                                                                             one2many_mode=one2many_mode,
                                                                             num_predictions=num_predictions,
                                                                             perturb_std=baseline_perturb_std,
                                                                                      title=title,
                                                                                      title_lens=title_lens,
                                                                                      title_mask=title_mask)
            greedy_str_2dlist = sample_list_to_str_2dlist(greedy_sample_list, oov_lists, opt.idx2word, opt.vocab_size,
                                                          eos_idx,
                                                          delimiter_word, opt.word2idx[pykp.io.UNK_WORD], opt.replace_unk,
                                                        src_str_list, opt.separate_present_absent, pykp.io.PEOS_WORD)
        generator.model.train()

    # Compute the reward for each predicted keyphrase
    # if using reward shaping, each keyphrase will have its own reward, else, only the last keyphrase will get a reward
    # In addition, we adds a regularization terms to the reward

    if reward_shaping:
        max_num_pred_phrases = max([len(pred_str_list) for pred_str_list in pred_str_2dlist])

        # compute the reward for each phrase, np array with size: [batch_size, num_predictions]
        phrase_reward = compute_phrase_reward(pred_str_2dlist, trg_str_2dlist, batch_size, max_num_pred_phrases, reward_shaping,
                              reward_type, topk, match_type, regularization_factor, regularization_type, entropy_array)
        # store the sum of cumulative reward for the experiment log
        cumulative_reward = phrase_reward[:, -1]
        cumulative_reward_sum = cumulative_reward.sum(0)

        # Subtract reward by a baseline if needed
        if opt.baseline == 'self':
            max_num_greedy_phrases = max([len(greedy_str_list) for greedy_str_list in greedy_str_2dlist])
            assert max_num_pred_phrases == max_num_greedy_phrases, "if you use self-critical training with reward shaping, make sure the number of phrases sampled from the policy and that decoded by greedy are the same."
            # use the reward of greedy decoding as baseline
            phrase_baseline = compute_phrase_reward(greedy_str_2dlist, trg_str_2dlist, batch_size, max_num_greedy_phrases, reward_shaping,
                              reward_type, topk, match_type, regularization_factor, regularization_type, entropy_array)
            phrase_reward = phrase_reward - phrase_baseline

        # convert each phrase reward to its improvement in reward
        phrase_reward = shape_reward(phrase_reward)

        # convert to reward received at each decoding step
        stepwise_reward = phrase_reward_to_stepwise_reward(phrase_reward, pred_eos_idx_mask)
        q_value_estimate_array = np.cumsum(stepwise_reward[:, ::-1], axis=1)[:, ::-1].copy()

    elif opt.separate_present_absent:

        ####no bert
        #present_absent_reward = compute_present_absent_reward(pred_str_2dlist, trg_str_2dlist, reward_type=reward_type, topk=topk, match_type=match_type,
        #               regularization_factor=regularization_factor, regularization_type=regularization_type, entropy=entropy_array)
        ####bert
        present_absent_reward = compute_present_absent_reward(pred_str_2dlist, trg_str_2dlist, reward_type=reward_type, topk=topk, match_type=match_type,
                    regularization_factor = regularization_factor, regularization_type = regularization_type, entropy = entropy_array, bert = bert)

        cumulative_reward = present_absent_reward.sum(1)
        cumulative_reward_sum = cumulative_reward.sum(0)
        # Subtract reward by a baseline if needed
        if opt.baseline == 'self':

            ####no bert
            #present_absent_baseline = compute_present_absent_reward(greedy_str_2dlist, trg_str_2dlist, reward_type=reward_type, topk=topk, match_type=match_type,
            #           regularization_factor=regularization_factor, regularization_type=regularization_type, entropy=entropy_array)
            ####bert
            present_absent_baseline = compute_present_absent_reward(greedy_str_2dlist, trg_str_2dlist, reward_type=reward_type, topk=topk, match_type=match_type,
                       regularization_factor = regularization_factor, regularization_type = regularization_type, entropy = entropy_array, bert = bert)


            present_absent_reward = present_absent_reward - present_absent_baseline
        stepwise_reward = present_absent_reward_to_stepwise_reward(present_absent_reward, max_pred_seq_len, location_of_peos_for_each_batch, location_of_eos_for_each_batch)
        q_value_estimate_array = np.cumsum(stepwise_reward[:, ::-1], axis=1)[:, ::-1].copy()

    else:  # neither using reward shaping
        # only receive reward at the end of whole sequence, np array: [batch_size]
        cumulative_reward = compute_batch_reward(pred_str_2dlist, trg_str_2dlist, batch_size, reward_type=reward_type, topk=topk, match_type=match_type,
                       regularization_factor=regularization_factor, regularization_type=regularization_type, entropy=entropy_array)
        # store the sum of cumulative reward (before baseline) for the experiment log
        cumulative_reward_sum = cumulative_reward.sum(0)
        # Subtract the cumulative reward by a baseline if needed
        if opt.baseline == 'self':
            baseline = compute_batch_reward(greedy_str_2dlist, trg_str_2dlist, batch_size, reward_type=reward_type, topk=topk, match_type=match_type,
                       regularization_factor=regularization_factor, regularization_type=regularization_type, entropy=entropy_array)
            cumulative_reward = cumulative_reward - baseline
        # q value estimation for each time step equals to the (baselined) cumulative reward
        q_value_estimate_array = np.tile(cumulative_reward.reshape([-1, 1]), [1, max_pred_seq_len])  # [batch, max_pred_seq_len]

    #shapped_baselined_reward = torch.gather(shapped_baselined_phrase_reward, dim=1, index=pred_phrase_idx_mask)

    # use the return as the estimation of q_value at each step

    q_value_estimate = torch.from_numpy(q_value_estimate_array).type(torch.FloatTensor).to(src.device)
    q_value_estimate.requires_grad_(True)
    q_estimate_compute_time = time_since(start_time)

    # compute the policy gradient objective
    pg_loss = compute_pg_loss(log_selected_token_dist, output_mask, q_value_estimate)

    # back propagation to compute the gradient
    start_time = time.time()
    pg_loss.backward()
    backward_time = time_since(start_time)

    if opt.max_grad_norm > 0:
        grad_norm_before_clipping = nn.utils.clip_grad_norm_(generator.model.parameters(), opt.max_grad_norm)

    # take a step of gradient descent
    optimizer.step()

    stat = RewardStatistics(cumulative_reward_sum, pg_loss.item(), batch_size, sample_time, q_estimate_compute_time, backward_time)
    # (final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0)
    # reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0

    return stat, log_selected_token_dist.detach()

'''
def preprocess_sample_list(sample_list, idx2word, vocab_size, oov_lists, eos_idx):
    for sample, oov in zip(sample_list, oov_lists):
        sample['sentence'] = prediction_to_sentence(sample['prediction'], idx2word, vocab_size, oov, eos_idx)
    return
'''
