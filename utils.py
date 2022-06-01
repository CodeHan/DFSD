import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
import pickle as pkl
import numpy as np
from copy import deepcopy
import os
from glob import glob
import logging
import pathlib
from collections import OrderedDict
from settings import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from settings import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS
from multiprocessing import Pool
import sys
import time
import quadprog
import io
#import torchsnooper
from pyemd import emd_with_flow
from torch.nn import CrossEntropyLoss, MSELoss,KLDivLoss
import GPUtil
from parallel import DataParallelCriterion
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def get_gen_token(task):
    if args.add_task_tokens:
        return '__' + task + '__'
    else:
        return '__gen__'


def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type != "multitask" else args.model_dir_root



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def get_new_layer_weight(trans_matrix, distance_matrix, stu_layer_num, tea_layer_num, T, type_update='att'):
    if type_update == 'att':
        global att_student_weight, att_teacher_weight
        student_layer_weight = np.copy(att_student_weight)
        teacher_layer_weight = np.copy(att_teacher_weight)
    else:
        global rep_student_weight, rep_teacher_weight
        student_layer_weight = np.copy(rep_student_weight)
        teacher_layer_weight = np.copy(rep_teacher_weight)

    distance_matrix = distance_matrix.detach().cpu().numpy().astype('float64')
    trans_weight = np.sum(trans_matrix * distance_matrix, -1)
    
    for i in range(stu_layer_num):
        student_layer_weight[i] = trans_weight[i] / student_layer_weight[i]
    weight_sum = np.sum(student_layer_weight)
    for i in range(stu_layer_num):
        if student_layer_weight[i] != 0:
            student_layer_weight[i] = weight_sum / student_layer_weight[i]

    trans_weight = np.sum(np.transpose(trans_matrix) * distance_matrix, -1)
    for j in range(tea_layer_num):
        teacher_layer_weight[j] = trans_weight[j + stu_layer_num] / teacher_layer_weight[j]
    weight_sum = np.sum(teacher_layer_weight)
    for i in range(tea_layer_num):
        if teacher_layer_weight[i] != 0:
            teacher_layer_weight[i] = weight_sum / teacher_layer_weight[i]

    student_layer_weight = softmax(student_layer_weight / T)
    teacher_layer_weight = softmax(teacher_layer_weight / T)

    if type_update == 'att':
        att_student_weight = student_layer_weight
        att_teacher_weight = teacher_layer_weight
    else:
        rep_student_weight = student_layer_weight
        rep_teacher_weight = teacher_layer_weight


def sd_emd_loss(student_atts, teacher_atts, student_reps, teacher_reps,
                     device, loss_mse, args, global_step, T=1):
    
    global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight
    def embedding_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j]
                tmp_loss = loss_mse(student_rep, teacher_rep,mode='mse') if args.jsloss else loss_mse(student_rep, teacher_rep)
                
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_rep_loss(student_reps, teacher_reps, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):
        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_rep = student_reps[i+1]
            for j in range(tea_layer_num):
                teacher_rep = teacher_reps[j + 1]
                tmp_loss = loss_mse(student_rep, teacher_rep,mode='log_softmax') if args.jsloss else loss_mse(student_rep, teacher_rep)
                
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss

        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        
        rep_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return rep_loss, trans_matrix, distance_matrix

    def emd_att_loss(student_atts, teacher_atts, student_layer_weight, teacher_layer_weight,
                     stu_layer_num, tea_layer_num, device, loss_mse):

        student_layer_weight = np.concatenate((student_layer_weight, np.zeros(tea_layer_num)))
        teacher_layer_weight = np.concatenate((np.zeros(stu_layer_num), teacher_layer_weight))
        totol_num = stu_layer_num + tea_layer_num
        distance_matrix = torch.zeros([totol_num, totol_num]).cuda()
        for i in range(stu_layer_num):
            student_att = student_atts[i]
            for j in range(tea_layer_num):
                teacher_att = teacher_atts[j]
                student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                          student_att)
                teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                          teacher_att)

                tmp_loss = loss_mse(student_att, teacher_att,mode='log_softmax') if args.jsloss else loss_mse(student_att, teacher_att)
                distance_matrix[i][j + stu_layer_num] = distance_matrix[j + stu_layer_num][i] = tmp_loss
        _, trans_matrix = emd_with_flow(student_layer_weight, teacher_layer_weight,
                                        distance_matrix.detach().cpu().numpy().astype('float64'))
        att_loss = torch.sum(torch.tensor(trans_matrix).cuda() * distance_matrix)
        return att_loss, trans_matrix, distance_matrix

    stu_layer_num = len(student_atts)
    tea_layer_num = len(teacher_atts)
    att_trans_matrix, att_distance_matrix,rep_trans_matrix, rep_distance_matrix=None,None,None,None
    if args.use_att:
        att_loss, att_trans_matrix, att_distance_matrix = \
            emd_att_loss(student_atts, teacher_atts, att_student_weight, att_teacher_weight,
                         stu_layer_num, tea_layer_num, device, loss_mse)
        if args.update_weight:
            get_new_layer_weight(att_trans_matrix, att_distance_matrix, stu_layer_num, tea_layer_num, T=T)
        att_loss = att_loss.to(device)
    else:
        att_loss = torch.tensor(0)
    if args.use_rep:
        if args.embedding_emd:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                embedding_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num+1, tea_layer_num+1, device, loss_mse)
            if args.update_weight:
                get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num+1, tea_layer_num+1, T=T, type_update='xx')
        else:
            rep_loss, rep_trans_matrix, rep_distance_matrix = \
                emd_rep_loss(student_reps, teacher_reps, rep_student_weight, rep_teacher_weight,
                             stu_layer_num, tea_layer_num, device, loss_mse)

            if args.update_weight:
                get_new_layer_weight(rep_trans_matrix, rep_distance_matrix, stu_layer_num, tea_layer_num, T=T, type_update='xx')
        rep_loss = rep_loss.to(device)
    else:
        rep_loss = torch.tensor(0)


    if not args.seperate:
        student_weight = np.mean(np.stack([att_student_weight, rep_student_weight]), 0)
        teacher_weight = np.mean(np.stack([att_teacher_weight, rep_teacher_weight]), 0)
        #if global_step % args.eval_step == 0:
            #logger.info('all_student_weight:{}'.format(student_weight))
            #logger.info('all_teacher_weight:{}'.format(teacher_weight))
        att_student_weight = student_weight
        att_teacher_weight = teacher_weight
        rep_student_weight = student_weight
        rep_teacher_weight = teacher_weight
    
    else:
        if global_step % args.eval_step == 0:
            logger.info('att_student_weight:{}'.format(att_student_weight))
            logger.info('att_teacher_weight:{}'.format(att_teacher_weight))
            logger.info('rep_student_weight:{}'.format(rep_student_weight))
            logger.info('rep_teacher_weight:{}'.format(rep_teacher_weight))
    
    if args.add_softmax:
        att_student_weight = softmax(att_student_weight)
        att_teacher_weight = softmax(att_teacher_weight)

        rep_student_weight = softmax(rep_student_weight)
        rep_teacher_weight = softmax(rep_teacher_weight)
    return att_loss.requires_grad_(), rep_loss.requires_grad_(),[att_trans_matrix, att_distance_matrix,rep_trans_matrix, rep_distance_matrix]

def seperate_old_and_new_data(cqa, Y, gen_X, gen_Y,special_task_token):
    final_new_cqa, final_new_Y, final_new_gen_X, final_new_gen_Y = [], [], [], []
    final_old_cqa, final_old_Y, final_old_gen_X, final_old_gen_Y = [], [], [], []
    
    if args.add_task_tokens:
        for i in range(len(cqa)):
            for j in range(len(cqa[i])):
                new_cqa, new_Y, new_gen_X, new_gen_Y = [], [], [], []
                old_cqa, old_Y, old_gen_X, old_gen_Y = [], [], [], []
                for k in range(len(cqa[i][j])):
                    if gen_X[i][j][k][0]==special_task_token:
                        new_cqa.append(cqa[i][j][k])
                        new_Y.append(Y[j][k])
                        new_gen_X.append(gen_X[i][j][k])
                        new_gen_Y.append(gen_Y[j][k])
                    else:
                        old_cqa.append(cqa[i][j][k])
                        old_Y.append(Y[j][k])
                        old_gen_X.append(gen_X[i][j][k])
                        old_gen_Y.append(gen_Y[j][k])
                if len(new_cqa)>0:
                    final_new_cqa.append(torch.cat(new_cqa).view([len(new_cqa),-1]))
                    final_new_Y.append(torch.cat(new_Y).view([len(new_Y), -1]))
                    final_new_gen_X.append(torch.cat(new_gen_X).view([len(new_gen_X), -1]))
                    final_new_gen_Y.append(torch.cat(new_gen_Y).view([len(new_gen_Y), -1]))
                if len(old_cqa)>0:
                    final_old_cqa.append(torch.cat(old_cqa).view([len(old_cqa), -1]))
                    final_old_Y.append(torch.cat(old_Y).view([len(old_Y), -1]))
                    final_old_gen_X.append(torch.cat(old_gen_X).view([len(old_gen_X), -1]))
                    final_old_gen_Y.append(torch.cat(old_gen_Y).view([len(old_gen_Y), -1]))


    else:
        raise ValueError('args.add_task_tokens=%s, must be True')

    return [final_old_cqa, final_old_Y, final_old_gen_X, final_old_gen_Y],[final_new_cqa, final_new_Y, final_new_gen_X, final_new_gen_Y]
	



def hda_loss(y, labels, teacher_scores, T=1.0, alpha=0.8,reduction_kd='mean', reduction_nll='mean', reduce_T=1, is_teacher=True):
    teacher_T = T if is_teacher else 1
    rt = T*T/reduce_T if is_teacher else 1
	
    if teacher_scores is not None:
        student_likelihood = torch.nn.functional.log_softmax(y / T, dim=-1)
        targets_prob = torch.nn.functional.softmax(teacher_scores / T, dim=-1)
        stu_shape = list(student_likelihood.shape)     
        stu_shape[-1]=1            
        add_tensor = torch.zeros(stu_shape).to(args.device_ids[0])           
        targets_prob = torch.cat([targets_prob,add_tensor],dim=-1)
        d_loss = (- targets_prob * student_likelihood).mean() * T * T / reduce_T
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = torch.nn.functional.cross_entropy(torch.transpose(y,-1,-2), labels, reduction=reduction_nll,ignore_index=FILL_VAL)#nll_loss = torch.nn.functional.cross_entropy(torch.unsqueeze(y,0), labels, reduction=reduction_nll)
    
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    return tol_loss, d_loss, nll_loss

def get_loss_dict(qa_loss, lm_loss,tran_loss=None,hda_loss=None,qa_matrix=[],lm_matrix=[],trans_loss_all=[],hda_loss_all=[]):
    loss_dict={}
    loss_dict['qa_loss']=qa_loss
    loss_dict['lm_loss'] = lm_loss
    loss_dict['tran_loss'] = tran_loss
    loss_dict['hda_loss'] = hda_loss
    loss_dict['qa_matrix'] = qa_matrix
    loss_dict['lm_matrix'] = lm_matrix
    loss_dict['trans_loss_all'] = trans_loss_all
    loss_dict['hda_loss_all'] = hda_loss_all

    return loss_dict
    
import scipy.special as special
class JSLoss(torch.nn.Module):
    def __init__(self):
        super(JSLoss,self).__init__()
        self.kl_loss=KLDivLoss(reduction='batchmean')
        self.mse=MSELoss()
    def forward(self, input, target,mode='log_softmax'):
        
        if mode=='log':
            return 0.5*(self.kl_loss(torch.log(input),target)+self.kl_loss(torch.log(target),input))
        elif mode=='log_softmax':
            return 0.5*(self.kl_loss(F.log_softmax(input,dim=-1),F.softmax(target,dim=-1))+self.kl_loss(F.log_softmax(target,dim=-1),F.softmax(input,dim=-1)))
        else:
            return self.mse(input,target)
        



def get_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct,global_step,layer_weight,special_task_token,teacher_model=None):
    
    if "lll" in args.seq_train_type:
        
        global att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight
        att_student_weight, rep_student_weight, att_teacher_weight, rep_teacher_weight = layer_weight
        
        emd_loss_func= JSLoss() if args.jsloss else MSELoss()
        loss_mse=DataParallelCriterion(emd_loss_func, args.device_ids).to(torch.device('cuda:%s' % args.device_ids[0]))
        
        device = torch.device("cuda:%s"%args.device_ids[0] if torch.cuda.is_available() else "cpu")
        tran_loss=torch.zeros([],requires_grad=True).to(torch.device('cuda:%s' % args.device_ids[0]))
        trans_loss_all = []
        qa_matrix=None
        lm_matrix=None
        
        _trans_qa_att_loss, _trans_qa_rep_loss, _trans_lm_att_loss, _trans_lm_rep_loss, _embedding_qa_loss, _embedding_lm_loss=[],[],[],[],[],[]
        hda_loss=torch.zeros([],requires_grad=True).to(torch.device('cuda:%s' % args.device_ids[0]))
        hda_loss_all = []
        
        if teacher_model!=None:
            old_data, new_data = seperate_old_and_new_data(cqa, Y, gen_X, gen_Y,special_task_token)
            
            if len(old_data[0]) != 0:
                
                tran_loss=None
                hda_loss=None
                tran_loss_temp=None
                hda_loss_temp = None
                
                for i in range(1):
                    
                    old_qa_logits, old_qa_presents, old_qa_hidden_states, old_qa_attentions=parallel_model([old_data[0]])[0]#parallel_model([old_data[0][0][i].view(1,-1)])[0]
                    old_lm_logits, old_lm_presents, old_lm_hidden_states, old_lm_attentions=parallel_model([old_data[2]])[0]#parallel_model([old_data[2][0][i].view(1,-1)])[0]
                    
                    with torch.no_grad():
                        teacher_qa_logits, teacher_qa_presents, teacher_qa_hidden_states, teacher_qa_attentions = teacher_model([old_data[0]])[0]#teacher_model([old_data[0][0][i].view(1,-1)])[0]
                        teacher_lm_logits, teacher_lm_presents, teacher_lm_hidden_states, teacher_lm_attentions = teacher_model([old_data[2]])[0]#teacher_model([old_data[2][0][i].view(1,-1)])[0]
                    
                    
                    trans_qa_att_loss ,trans_qa_rep_loss, qa_matrix =  sd_emd_loss(old_qa_attentions, teacher_qa_attentions, old_qa_hidden_states, teacher_qa_hidden_states,
													device, loss_mse, args, global_step, T=args.emd_temp)

                    trans_lm_att_loss ,trans_lm_rep_loss, lm_matrix = sd_emd_loss(old_lm_attentions, teacher_lm_attentions, old_lm_hidden_states,
													 teacher_lm_hidden_states,
													 device, loss_mse, args, global_step, T=args.emd_temp)
                    embedding_qa_loss = loss_mse(old_qa_hidden_states[0], teacher_qa_hidden_states[0])
                    embedding_lm_loss = loss_mse(old_lm_hidden_states[0], teacher_lm_hidden_states[0])
                    
                    hda_qa_loss=hda_loss(old_qa_logits,old_data[1][0],teacher_qa_logits,T=args.emd_temp,alpha=args.alpha)
                    hda_lm_loss = hda_loss(old_lm_logits, old_data[3][0], teacher_lm_logits,T=args.emd_temp,alpha=args.alpha)
                    
                    tran_loss_temp=trans_qa_att_loss +trans_qa_rep_loss+trans_lm_att_loss +trans_lm_rep_loss+embedding_qa_loss+embedding_lm_loss
                    tran_loss = tran_loss_temp if tran_loss is None else torch.add(tran_loss,tran_loss_temp)			
                    hda_loss_temp=hda_qa_loss[0]+hda_lm_loss[0]
                    hda_loss = hda_loss_temp if hda_loss is None else torch.add(hda_loss,hda_loss_temp)
                    trans_loss_all.extend([torch.sum(trans_qa_att_loss).item(),# / len(old_data[0]),
                        torch.sum(trans_qa_rep_loss).item(),# / len(old_data[0]),
                        torch.sum(trans_lm_att_loss).item(),# / len(old_data[0]),
                        torch.sum(trans_lm_rep_loss).item(),# / len(old_data[0]),
                        torch.sum(embedding_qa_loss).item(),# / len(old_data[0]),
                        torch.sum(embedding_lm_loss).item()])# / len(old_data[0])])
					
                    hda_loss_all.extend([torch.sum(hda_qa_loss[i]).item() for i in range(len(hda_qa_loss))])
                    hda_loss_all.extend([torch.sum(hda_lm_loss[i]).item() for i in range(len(hda_lm_loss))])
                    
                    
        qa_logits, qa_presents, qa_hidden_states, qa_attentions=parallel_model(cqa)[0]
        lm_logits, lm_presents, lm_hidden_states, lm_attentions=parallel_model(gen_X)[0]
        
        qa_logits=[qa_logits] 
        lm_logits=[lm_logits] 
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y )
        lm_loss = loss_fct([torch.transpose(l, 1, 2) for l in lm_logits], gen_Y)
        
        new_layer_weight = (att_student_weight,rep_student_weight,att_teacher_weight,rep_teacher_weight)
        if teacher_model==None:
            
            return get_loss_dict(torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)),new_layer_weight
        else:           
            return get_loss_dict(torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss),args.beta*torch.mean(tran_loss),args.gama*torch.mean(hda_loss),qa_matrix,lm_matrix,trans_loss_all,hda_loss_all),new_layer_weight
    else:
        qa_logits = parallel_model(cqa)
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        return torch.mean(qa_loss), torch.tensor(0.)


def pad_to_max_len(l, pad_len, val):
    return l + [val] * pad_len


def pad_all_to_max_len(ls, val):
    max_len = max(len(l) for l in ls)
    return [pad_to_max_len(l, max_len-len(l), val) for l in ls]


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits


def varlen_collate_fn(data):
    batch_size = (len(data) + args.n_gpus - 1) // args.n_gpus
    cqs = torch.tensor(pad_all_to_max_len([datum[0] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqs = torch.tensor([datum[1] for datum in data]).split(batch_size)
    cqas = torch.tensor(pad_all_to_max_len([datum[2] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    len_cqas = torch.tensor([datum[3] for datum in data]).split(batch_size)
    Ys = torch.tensor(pad_all_to_max_len([datum[4] for datum in data], FILL_VAL)).split(batch_size)
    gen_Xs = torch.tensor(pad_all_to_max_len([datum[5] for datum in data], SPECIAL_TOKEN_IDS["pad_token"])).split(batch_size)
    gen_Ys = torch.tensor(pad_all_to_max_len([datum[6] for datum in data], FILL_VAL)).split(batch_size)
    return list(cqs), list(len_cqs), list(cqas), list(len_cqas), list(Ys), list(gen_Xs), list(gen_Ys)


def dynamic_collate_fn(data, batch_size):

    def local_collate():
        null_counter = 0
        _cqs, _len_cqs, _cqas, _len_cqas, _Ys, _gen_Xs, _gen_Ys = [], [], [], [], [], [], []
        Y_max_len = max(len(data[j][4]) for j in range(st, ed))
        cq_max_len = max(len(data[j][0]) for j in range(st, ed))
        for j in range(st, ed):
            if None in data[j] or [] in data[j]:
                null_counter+=1
                logger.warning('null example in collate_fn, count: {}'.format(null_counter))
                continue

            pad_len = cqa_max_len - len(data[j][2])

            _cqs.append(pad_to_max_len(data[j][0], cq_max_len-len(data[j][0]), SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqs.append(data[j][1])
            _cqas.append(pad_to_max_len(data[j][2], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _len_cqas.append(data[j][3])
            _Ys.append(pad_to_max_len(data[j][4], Y_max_len - len(data[j][4]), FILL_VAL))
            _gen_Xs.append(pad_to_max_len(data[j][5], pad_len, SPECIAL_TOKEN_IDS["pad_token"]))
            _gen_Ys.append(pad_to_max_len(data[j][6], pad_len, FILL_VAL))

        cqs.append(torch.tensor(_cqs))
        len_cqs.append(torch.tensor(_len_cqs))
        cqas.append(torch.tensor(_cqas))
        len_cqas.append(torch.tensor(_len_cqas))
        Ys.append(torch.tensor(_Ys))
        gen_Xs.append(torch.tensor(_gen_Xs))
        gen_Ys.append(torch.tensor(_gen_Ys))

    cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys = [], [], [], [], [], [], []
    cqa_max_len, cnt, st = 0, 0, 0
    ed=len(data)-1
    for ed, datum in enumerate(data):
        ln = len(datum[2]) # use cqas to calibrate
        if max(cqa_max_len, ln)**LEN_FACTOR * (ed - st + 1) > batch_size[cnt]:
            local_collate()
            cnt += 1
            cqa_max_len = 0
            st = ed
        cqa_max_len = max(cqa_max_len, ln)
    ed += 1  # otherwise ed will be len(data)-1
    local_collate()

    return cqs, len_cqs, cqas, len_cqas, Ys, gen_Xs, gen_Ys


class QADataset(Dataset):
    def __init__(self, task_name,data_paths, data_type, gen_token, extra_data=[],mix=False):
        self.data_type = data_type
        self.gen_token = gen_token
        if args.use_sep:
            self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d
        
        self.data = []
        self.max_a_len = 0
        if len(data_paths)==1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            #data = self._sort_by_index(data)
            #args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json" 
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json" 
            with open(os.path.join(args.data_dir,answers_file),"r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)

        model_dir = get_model_dir([task_name])  
        gen_path = os.path.join(model_dir, "%s_%s.csv" %(task_name,self.data_type))
        real_data = [TOKENIZER.decode(res[-3]) for res in self.data]
        write_extra_data(gen_path, real_data)

        if len(extra_data) > 0:
            extra_data = map(lambda x: self.etl_single_extra_data(x), extra_data)
            extra_data = list(filter(lambda x:x, extra_data))
            if args.gen_lm_sample_percentage > 0. and len(extra_data) == 0:
                logger.warning("No good extra data but sample percentage > 0!")
            self.data += extra_data


    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            if args.use_sep:
                context, qa = re.split(str(SPECIAL_TOKEN_IDS["sep_token"]), data)
            else:
                context = ""
                qa = data
            question, answer = re.split(str(SPECIAL_TOKEN_IDS["ans_token"]), qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(str(SPECIAL_TOKEN_IDS["eos_token"]), "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        if args.use_sep:
            cq_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [])
        else:
            cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        if cqa_example is None:
            return
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        if args.use_sep:
            gen_X_example = self.concat_example([gen_token], context, [self.sep_token], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [self.eos_token])
        else:
            gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        examples = []
        context = TOKENIZER.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            question = TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(TOKENIZER.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            max_a_len = max(max_a_len, len(answer))
            cur_data = self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0))
            if cur_data is None:
                continue
            examples.append(cur_data)
        return examples, max_a_len

    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    #def _sort_by_index(self,data):
    #    datum = []
    #    for d in data:
    #        for qa in d["qas"]:
    #            datum.append({"context":d["context"], "qas":[qa]})
    #    datum.sort(key=lambda x:x["qas"][0]["id"])
    #    return datum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



class EarlyStopping:
    def __init__(self, logger,  patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
        elif score < self.best_score:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(model_dir)
        TOKENIZER.save_pretrained(model_dir)
        self.val_loss_min = val_loss


class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, loss, scheduler_steps):
        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)

        if "gem" in args.seq_train_type and self.model.task_id >0: 
            store_grad(self.model.parameters, self.model.grads, self.model.grad_dims,self.model.task_id)
            indx = torch.cuda.LongTensor([i for i in range(self.model.task_id)])
            dotp = torch.mm(self.model.grads[:, self.model.task_id].unsqueeze(0),
                            self.model.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.model.grads[:, self.model.task_id].unsqueeze(1),
                              self.model.grads.index_select(1, indx), args.qp_margin)
                # copy gradients back
                overwrite_grad(self.model.parameters,
                               self.model.grads[:, self.model.task_id],
                               self.model.grad_dims)
            
        if args.seq_train_type in args.REG_TYPE_KEYS:
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()


class GEMStep:
    def __init__(self, model, parallel_model, train_loss_fct, optimizer):
        self.model = model
        self.parallel_model = parallel_model
        self.train_loss_fct = train_loss_fct
        self.optimizer = optimizer

    def __call__(self,current_task_id):
        for past_task_id, md in enumerate(args.memory_data):
            # Not saving current task's grads.
            if past_task_id >= current_task_id: return
            qadata = QADataset(None, "test", "gen", md)
            dataloader = create_dataloader(qadata, "test")
            grads_tmp = torch.zeros(sum(self.model.grad_dims),).cuda()
            if not args.fp32:
                grads_tmp = grads_tmp.half() 
            for _, _, cqa, _, Y, gen_X, gen_Y in dataloader:
                #CHECK
                n_inputs = sum(_cqa.shape[0] for _cqa in cqa)
                self.optimizer.zero_grad()
                for i in range(len(cqa)):
                    cqa[i] = (cqa[i].to(args.device_ids[i]),)
                    Y[i] = Y[i].to(args.device_ids[i])
                    gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                    gen_Y[i] = gen_Y[i].to(args.device_ids[i])

                losses = get_losses(self.parallel_model, cqa, Y, gen_X, gen_Y, self.train_loss_fct)
                loss = sum(losses)
                if not args.fp32:
                    self.optimizer.backward(loss, update_master_grads=False)
                else:
                    loss.backward()

                if not args.fp32:
                    #copy fp16 grads to fp32 grads  
                    self.optimizer.update_master_grads()
                    self.optimizer.clip_master_grads(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                i = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        beg = 0 if i == 0 else sum(self.model.grad_dims[:i])
                        end = sum(self.model.grad_dims[:i+1])
                        grads_tmp[beg: end] += param.grad.data.view(-1)*n_inputs
                    i += 1

            grads_tmp /= len(qadata)
            self.model.grads[:, past_task_id].copy_(grads_tmp)
            self.optimizer.zero_grad()


class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if args.debug or self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][2])
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=args.n_workers,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids,output_attentions=args.output_attentions,output_hidden_states=args.output_hidden_states)
        
        return outputs


def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(MODEL_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0

#@torchsnooper.snoop()
def sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens):
    while len(need_process) > 0:
        first_id = next(iter(need_process))
        shortest_len = len(qa_results[first_id])
        decode_batch_size = int(args.memory_sizes[0] * MEMORY_FACTOR[args.seq_train_type] // (shortest_len+1)**LEN_FACTOR)
        it = iter(need_process)
        stop = False
        remove_ids = []
        while not stop:
            batch_ids, input_ids, past = [], [], [[] for _ in range(MODEL_CONFIG.n_layer)]
            while True:
                try:
                    cur_id = next(it)
                    if len(qa_results[cur_id]) > shortest_len:
                        stop = True
                        break
                    batch_ids.append(cur_id)
                    if args.model_name == "gpt2":
                        input_ids.append(qa_results[cur_id][-1:])
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            past[layer_id].append(all_pasts[layer_id][cur_id])
                    else:
                        input_ids.append(qa_results[cur_id])
                    if len(input_ids) == decode_batch_size:
                        break
                except StopIteration:
                    stop = True
                    break

            n_inputs = len(input_ids)
            if n_inputs == 0:
                break
            input_ids = torch.stack(input_ids)
            if args.model_name == "gpt2":
                for layer_id in range(MODEL_CONFIG.n_layer):
                    past[layer_id] = torch.stack(past[layer_id], dim=1)
                all_outputs = model(input_ids=input_ids.cuda(), past=past)
            else:
                all_outputs = model(input_ids=input_ids.cuda())

            outputs = all_outputs[0]
            if args.model_name == "gpt2":
                pasts = all_outputs[1]

            next_logits = outputs[..., -1, :] / args.temperature_qa
            next_tokens = logits_to_tokens(next_logits).cpu()

            for i, cur_id in enumerate(batch_ids):
                if next_tokens[i] == SPECIAL_TOKEN_IDS["eos_token"]:
                    remove_ids.append(cur_id)
                else:
                    qa_results[cur_id] = torch.cat((qa_results[cur_id], next_tokens[i]))
                    if len(qa_results[cur_id]) in [max_tot_lens[cur_id], args.max_len]:
                        remove_ids.append(cur_id)
                    elif args.model_name == "gpt2":
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cur_id] = pasts[layer_id][:, i].type(torch.float if args.fp32 else torch.half)
        for idx in remove_ids:
            remove_id(idx, need_process, all_pasts)


def write_extra_data(dump_path, qa_results):
    logger.info(f"writing extra data in {dump_path} ...")
    with open(dump_path,"w",newline="",encoding="utf-8") as f:
        lm_writer = csv.writer(f,delimiter=',')
        lm_writer.writerow(["gen"])
        for l in qa_results:
            lm_writer.writerow([l])


def parse_single_real_data(data,task):
    c = data["paragraphs"][0]["context"]
    q = data["paragraphs"][0]["qas"][0]["question"]
    a = data["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    if args.use_sep:
        data = "{}{}{}{}{}{}{}".format(SPECIAL_TOKENS[task],c,SPECIAL_TOKENS["sep_token"],q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    else:
        data = "{}{} {}{}{}{}".format(SPECIAL_TOKENS[task],c,q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    return data


def get_real_data(task, train_extra_data, accum=True, encode=True):
    task_idx = args.tasks.index(task)
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    if accum:
        prev_tasks = args.tasks[:task_idx]
        gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))//len(prev_tasks)
    else:
        prev_tasks = [args.tasks[task_idx-1]]
        gen_size = int(gen_size * args.gen_lm_sample_percentage)

    datum = []
    for prev_task in prev_tasks:
        with open(TASK_DICT[prev_task]["train"],"r") as f:
            data = data_expand(json.load(f)["data"])
        indices = np.random.choice(range(len(data)), gen_size)
        for i in indices:
            d = parse_single_real_data(data[i],prev_task)
            datum.append(d)
            if encode:
                train_extra_data.append(TOKENIZER.encode(d))
        
    model_dir = get_model_dir([prev_task])
    dump_path = os.path.join(model_dir,"real.csv")
    write_extra_data(dump_path, datum)
    return dump_path


def read_extra_data(gen_path, train_extra_data):
    with open(gen_path,"r") as lm_file:
        reader = csv.reader(lm_file,delimiter=',')
        next(reader)
        for row in reader: 
            row = TOKENIZER.encode(row[0].strip()) 
            train_extra_data.append(row)


def create_extra_data(task, prev_task, model, train_extra_data):
    if args.real_sample:
        logger.info(f"using real data as extra data")
        return get_real_data(task, train_extra_data)
    begin=0
    tasks=args.tasks
    if len(args.tasks)>5:
        tasks=args.tasks[-5:]    
    task_cnt = tasks.index(task)
    model_dir = get_model_dir([prev_task])
    gen_path = os.path.join(model_dir,"lm.csv")
    if os.path.exists(gen_path):
        logger.info(f"extra data exists in {gen_path}, read it!")
        return read_extra_data(gen_path, train_extra_data) 
    gen_size = DATA_ATTRS[task]["train"]["data_size"]
    gen_size = int(np.ceil(gen_size * args.gen_lm_sample_percentage))
    gen_size -= (gen_size % task_cnt)

    if args.debug:
        gen_size = task_cnt

    model.eval()

    need_process = OrderedDict()
    qa_results = []
    for task_name in tasks[:task_cnt]:
        qa_results.extend([torch.tensor([SPECIAL_TOKEN_IDS[task_name]]) for _ in range(gen_size//task_cnt)])
    qr_np = np.array(qa_results)
    
    all_pasts = [[
        torch.empty(2, MODEL_CONFIG.n_head, 0, MODEL_CONFIG.n_embd//MODEL_CONFIG.n_head,
            dtype=torch.float if args.fp32 else torch.half).cuda()
        for _ in range(gen_size)
    ] for __ in range(MODEL_CONFIG.n_layer)]
    
    max_tot_lens = [args.max_len for _ in range(gen_size)]

    for i in range(gen_size):
        need_process.update([[i, None]])
        if len(need_process) > int(args.memory_sizes[0] * 0.12):
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)
    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    model.train()

    qa_results = [res.tolist() for res in qa_results]
    train_extra_data.extend(qa_results)
    qa_results = [TOKENIZER.decode(res) for res in qa_results]

    write_extra_data(gen_path, qa_results)


def logits_to_tokens(next_logits):
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_qa, top_p=args.top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    return next_tokens

 
def lll_unbound_setting(split_size=10,data_type="train",test_target="self"):
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    if data_type == "test":
        args.splitted_tasks = [f"task_{i}" for i in range(split_size)]
        args.n_train_epochs = {task: args.n_train_epochs for task in args.splitted_tasks}
        if test_target in ["self","all"]:
            for no in range(split_size):  
                task = f"task_{no}" 
                test_data_path = os.path.join(data_dir,f"{task}-test.json")
                TASK_DICT[task] = {}
                TASK_DICT[task]["test"] = test_data_path
            if test_target == "all":
                args.tasks += args.splitted_tasks
            else:
                args.tasks = args.splitted_tasks
    elif data_type == "train":
        create_lll_unbound_data(split_size)
        args.n_train_epochs = {task: args.n_train_epochs for task in args.tasks}
    return TASK_DICT


def create_lll_unbound_data(split_size=10): 
    data_dir = os.path.join(args.data_dir,"{}_{}".format("_".join(args.tasks),args.gen_lm_sample_percentage))
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    datum = [] 
    test_datum = []
    data_sizes = [] 
    chunk_sizes = []
    for task in args.tasks:
        train_data_path = TASK_DICT[task]["train"]
        with open(train_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            data_sizes.append(len(data))
            datum += data
        test_data_path = TASK_DICT[task]["test"]
        with open(test_data_path, "r") as f:
            data = json.load(f)["data"]
            data = data_expand(data)
            test_datum.append(data) 
    chunk_size = int(np.ceil(len(datum)/split_size))

    tasks = []
    for no, i in enumerate(range(0, len(datum), chunk_size)):  
        task = f"task_{no}" 
        tasks.append(task)
        chunk = datum[i:i + chunk_size] if i < len(datum)-chunk_size else datum[i:]
        chunk_sizes.append(len(chunk))
        DATA_ATTRS[task] = {"train":{"data_size":None}}
        DATA_ATTRS[task]["train"]["data_size"] = len(chunk)
        train_data_path = os.path.join(data_dir,f"{task}-train.json")
        with open(train_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task] = {}
        TASK_DICT[task]["train"] = train_data_path
    args.tasks = tasks

    sis = get_split_indices(data_sizes,chunk_sizes)
    test_split = []
    for dic in sis.values():
        merged_data = []
        for k,v in dic.items():
            from_index = int(len(test_datum[k])*v[0])
            to_index = int(len(test_datum[k])*v[1])
            merged_data+= test_datum[k][from_index:to_index]
        test_split.append(merged_data)

    for no, chunk in enumerate(test_split):  
        task = f"task_{no}" 
        test_data_path = os.path.join(data_dir,f"{task}-test.json")
        with open(test_data_path,"w") as f:
            json.dump({"data":chunk},f)
        TASK_DICT[task]["test"] = test_data_path


def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para: 
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs":[d]})
    return datum


def get_split_indices(data_sizes,chunk_sizes):
    ds = deepcopy(data_sizes)
    records = {}
    tmp = {}
    order = 0 # data_sizes index
    i = 0 # chunk_sizes index
    while len(data_sizes)>0:
        d0 = data_sizes[0]
        c0 = chunk_sizes[0]
        if d0>c0:
            val = c0/ds[order]
        else:
            val = d0/ds[order]

        if order not in tmp:
            rec = (0,val)
            tmp[order] = val
        else:
            rec = (tmp[order],tmp[order]+val)
            tmp[order] += val
        if i in records:
            records[i][order] = rec
        else:
            records[i] = {order: rec}

        if d0>c0:
            data_sizes[0]-=c0
            chunk_sizes.pop(0)
            i+=1
        else:
            chunk_sizes[0]-=d0
            data_sizes.pop(0)
            order+=1
            if d0==c0:
                chunk_sizes.pop(0)
                i+=1
    return records


def store_grad(get_ps, grads, grad_dims, task_id): 
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[:i+1])
            grads[beg: end, task_id].copy_(param.grad.data.view(-1))
        i += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
