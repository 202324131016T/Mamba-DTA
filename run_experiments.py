## This program is based on the DeepDTA model(https://github.com/hkmztrk/DeepDTA)
## The program requires pytorch and gpu support.

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import os
from tqdm import tqdm
import matplotlib
from datahelper import *
import time
from copy import deepcopy
from emetrics import *
from sklearn.metrics import roc_auc_score


import numpy as np
import torch
import random
import os
seed_value = 0 # 设定随机数种子
np.random.seed(seed_value) # 设置 numpy random seed
random.seed(seed_value) # 设置 random seed
os.environ['PYTHONHASHSEED'] = str(seed_value) # 为了禁止 hash 随机化，使得实验可复现。
torch.manual_seed(seed_value) # 为 CPU 设置随机种子
torch.cuda.manual_seed(seed_value) # 为当前 GPU 设置随机种子（只用一块 GPU ）
torch.cuda.manual_seed_all(seed_value) # 为所有 GPU 设置随机种子（多块 GPU ）
torch.backends.cudnn.deterministic = True # cuda random 保证可重复性, cuda中对卷积操作进行了优化，牺牲了精度来换取计算效率。


from config.arguments import *
# include import net
print("current model:", net)
# my_experiment_net
'''
model.model1_att3
model.model1_att5
model.model2_att_cross_scale
model.model3_no_att
model.model4_att3_cross_scale
model.model4_att5_cross_scale
model.model_att3
model.model_att5
model.model_block
model.model_CoVAE
'''

def get_random_folds(tsize, foldcount):
    folds = []
    indices = set(range(tsize))
    foldsize = tsize / foldcount
    leftover = tsize % foldcount
    for i in range(foldcount):
        sample_size = foldsize
        if leftover > 0:
            sample_size += 1
            leftover -= 1
        fold = random.sample(indices, int(sample_size))
        indices = indices.difference(fold)
        folds.append(fold)

    # assert stuff
    foldunion = set([])
    for find in range(len(folds)):
        fold = set(folds[find])
        assert len(fold & foldunion) == 0, str(find)
        foldunion = foldunion | fold
    assert len(foldunion & set(range(tsize))) == tsize

    return folds
    # 随机进行drug index的划分, 划分为[[], [], foldcount个[]]

def get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount):
    assert len(np.array(label_row_inds).shape) == 1, 'label_row_inds should be one dimensional array'
    row_to_indlist = {}
    rows = sorted(list(set(label_row_inds)))
    for rind in rows: # 行[0, 1, 2, ..., n-1] n==亲和度有数值的行数
        alloccs = np.where(np.array(label_row_inds) == rind)[0]
        row_to_indlist[rind] = alloccs # 每一行表示：Y的对应行有几个数据，其他为nan
        # 在此将所有 affinity!=nan 进行编号 0~n-1，从Y的逐行进行编号
    drugfolds = get_random_folds(drugcount, foldcount) # 随机进行drug index的划分
    folds = []
    for foldind in range(foldcount):
        fold = []
        drugfold = drugfolds[foldind] # drug划分为foldcount个，每一个小数据集的drug index
        for drugind in drugfold:
            fold = fold + row_to_indlist[drugind].tolist() # 上述编号后的affinity index，按照划分后的drug index排列
        folds.append(fold)
    return folds
    # folds [[], [], [], [], [], []] 划分数据集[], 每个[]包含按照划分后的drug index排列的affinity index

def get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount):
    assert len(np.array(label_col_inds).shape) == 1, 'label_col_inds should be one dimensional array'
    col_to_indlist = {}
    cols = sorted(list(set(label_col_inds)))
    for cind in cols:
        alloccs = np.where(np.array(label_col_inds) == cind)[0]
        col_to_indlist[cind] = alloccs
    target_ind_folds = get_random_folds(targetcount, foldcount)
    folds = []
    for foldind in range(foldcount):
        fold = []
        targetfold = target_ind_folds[foldind]
        for targetind in targetfold:
            fold = fold + col_to_indlist[targetind].tolist()
        folds.append(fold)
    return folds


def loss_f(recon_x, x, mu, logvar):

    cit = nn.CrossEntropyLoss(reduction='none')
    cr_loss = torch.sum(cit(recon_x.permute(0, 2, 1), x), 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(cr_loss + KLD)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
        if isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight.data, 1)
            init.constant_(m.bias.data, 0)
        if isinstance(m, nn.LSTM):
            init.orthogonal_(m.all_weights[0][0])
            init.orthogonal_(m.all_weights[0][1])
        if isinstance(m, nn.Conv1d):
            init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0)

def prepare_interaction_pairs(XD, XT, Y, rows, cols):
    dataset = [[]]
    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        dataset[pair_ind].append(np.array(drug, dtype=np.float32))
        target = XT[cols[pair_ind]]
        dataset[pair_ind].append(np.array(target, dtype=np.float32))
        dataset[pair_ind].append(np.array(Y[rows[pair_ind], cols[pair_ind]], dtype=np.float32))
        if pair_ind < len(rows) - 1:
            dataset.append([])
    return dataset

# train model
def train(train_loader, model, FLAGS, lamda=-5):
    model.train()
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr) # learning rate default:0.001

    with tqdm(train_loader) as t:
        for drug_SMILES, target_protein, affinity in t:
            drug_SMILES = torch.Tensor(drug_SMILES)
            target_protein = torch.Tensor(target_protein)
            affinity = torch.Tensor(affinity)
            optimizer.zero_grad()

            affinity = Variable(affinity).cuda()
            if "CoVAE" in str(net): # CoVAE
                pre_affinity, new_drug, new_target, drug, target, mu_drug, logvar_drug, mu_target, logvar_target = \
                    model(drug_SMILES, target_protein, FLAGS)
                loss_affinity = loss_func(pre_affinity, affinity)
                loss_drug = loss_f(new_drug, drug, mu_drug, logvar_drug)
                loss_target = loss_f(new_target, target, mu_target, logvar_target)
                loss = loss_affinity + 10 ** lamda * (loss_drug + FLAGS.max_smi_len / FLAGS.max_seq_len * loss_target)
            else:
                pre_affinity = model(drug_SMILES, target_protein)
                loss_affinity = loss_func(pre_affinity, affinity)
                loss = loss_affinity
            loss.backward()
            optimizer.step()
            mse = loss_affinity.item()
            c_index = get_cindex(affinity.cpu().detach().numpy(), pre_affinity.cpu().detach().numpy())
            t.set_postfix(train_loss=loss.item(), mse=mse, train_cindex=c_index)
    return model

# test model
def test(model,test_loader, FLAGS):
    model.eval()
    loss_func = nn.MSELoss()
    affinity_ls = []
    pre_affinity_ls = []

    with torch.no_grad():
        for i, (drug_SMILES, target_protein, affinity) in enumerate(test_loader):
            if "CoVAE" in str(net):
                pre_affinity, _, _, _, _, _, _, _, _ = model(drug_SMILES, target_protein, FLAGS)
            else:
                pre_affinity = model(drug_SMILES, target_protein)
            pre_affinity_ls += pre_affinity.cpu().detach().numpy().tolist()
            affinity_ls += affinity.cpu().detach().numpy().tolist()

        pre_affinity_ls = np.array(pre_affinity_ls)
        affinity_ls = np.array(affinity_ls)
        loss = loss_func(torch.Tensor(pre_affinity_ls), torch.Tensor(affinity_ls))
        cindex = get_cindex(affinity_ls,pre_affinity_ls)
        rm2 = get_rm2(affinity_ls, pre_affinity_ls)
        if 'davis' in FLAGS.dataset_path:
            auc = roc_auc_score(np.int32(affinity_ls > 7), pre_affinity_ls)
        if 'kiba' in FLAGS.dataset_path:
            auc = roc_auc_score(np.int32(affinity_ls > 12.1), pre_affinity_ls)
    return cindex, loss, rm2, auc

# only test
def n_fold_only_test(XD, XT, Y, label_row_inds, label_col_inds, FLAGS, test_sets, i):
    # i: random_seed
    print("changed: only test begin: n_fold_only_test")

    folds = len(test_sets)  # 5 folds
    all_CI = [0 for x in range(folds)]
    all_MSE = [0 for x in range(folds)]
    all_MAE = [0 for x in range(folds)]
    all_rm2 = [0 for x in range(folds)]
    all_auc = [0 for x in range(folds)]
    all_aupr = [0 for x in range(folds)]
    logging("---test-----", FLAGS)

    for fold_ind in range(len(test_sets)):
        # test_dataset
        val_inds = test_sets[fold_ind]
        terows = label_row_inds[val_inds]
        tecols = label_col_inds[val_inds]
        test_dataset = prepare_interaction_pairs(XD, XT, Y, terows, tecols)
        if len(test_dataset) % FLAGS.batch_size == 1:
            test_loader = DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size, drop_last=True)
        else:
            test_loader = DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)

        # test
        affinity_ls = []
        pre_affinity_ls = []
        path = FLAGS.checkpoint_path + "fold_" + str(fold_ind) + "_checkpoint.pth"
        model = torch.load(path)
        model.eval() # net
        for drug_SMILES, target_protein, affinity in test_loader:
            # predicted affinity
            if "CoVAE" in str(net):
                pre_affinity, _, _, _, _, _, _, _, _ = model(drug_SMILES, target_protein, FLAGS)
            else:
                pre_affinity = model(drug_SMILES, target_protein)
            pre_affinity_ls += pre_affinity.cpu().detach().numpy().tolist()
            affinity_ls += affinity.cpu().detach().numpy().tolist()
        # every fold best epoch affinity
        pre_affinity_ls = np.array(pre_affinity_ls)
        affinity_ls = np.array(affinity_ls)

        # save affinity_ls and pre_affinity_ls for other work
        path = FLAGS.result_path
        print("save affinity_ls and pre_affinity_ls in ", path)
        path_real = path + "fold_" + str(fold_ind) + "_affinity_ls.txt"
        np.savetxt(path_real, affinity_ls)
        path_pred = path + "fold_" + str(fold_ind) + "_pre_affinity_ls.txt"
        np.savetxt(path_pred, pre_affinity_ls)

        CI = get_cindex(affinity_ls, pre_affinity_ls)  # CI
        mse = get_mse(affinity_ls, pre_affinity_ls)  # mse
        mae = get_mae(affinity_ls, pre_affinity_ls)  # mae
        rm2 = get_rm2(affinity_ls, pre_affinity_ls)  # rm2
        if 'davis' in FLAGS.dataset_path:
            pre_label = pre_affinity_ls
            label = np.int32(affinity_ls > 7.0)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        if 'kiba' in FLAGS.dataset_path:
            pre_label = pre_affinity_ls
            label = np.int32(affinity_ls > 12.1)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        print('test: Fold={:d}, CI={:.5f}, MSE={:.5f}, MAE={:.5f}, rm2={:.5f}, auc={:.5f}, aupr={:.5f}'
              .format(fold_ind, CI, mse, mae, rm2, auc, aupr))
        logging("test: Fold=%d, CI=%f, MSE=%f, MAE=%f, rm2=%f, auc=%f, aupr=%f"
                % (fold_ind, CI, mse, mae, rm2, auc, aupr), FLAGS)

        # every fold best epoch result
        all_CI[fold_ind] = CI
        all_MSE[fold_ind] = mse
        all_MAE[fold_ind] = mae
        all_rm2[fold_ind] = rm2
        all_auc[fold_ind] = auc
        all_aupr[fold_ind] = aupr
        print("fold_index=%d test over" % fold_ind)
    # folds over return n_folds results
    return all_CI, all_MSE, all_MAE, all_rm2, all_auc, all_aupr

# train and test
def n_fold_train_test(XD, XT, Y, label_row_inds, label_col_inds, FLAGS, train_sets, test_sets, i):
    # i: random_seed
    print("changed: train and test begin: n_fold_train_test")

    folds = len(test_sets) # 5 folds
    all_CI = [0 for x in range(folds)]
    all_MSE = [0 for x in range(folds)]
    all_MAE = [0 for x in range(folds)]
    all_rm2 = [0 for x in range(folds)]
    all_auc = [0 for x in range(folds)]
    all_aupr = [0 for x in range(folds)]
    logging("---train and val(test)-----", FLAGS)

    for fold_ind in range(len(test_sets)):
        # get dataloader
        labeled_inds = train_sets[fold_ind]
        trrows = label_row_inds[labeled_inds]
        trcols = label_col_inds[labeled_inds]
        train_dataset = prepare_interaction_pairs(XD, XT, Y, trrows, trcols)

        val_inds = test_sets[fold_ind]
        terows = label_row_inds[val_inds]
        tecols = label_col_inds[val_inds]
        test_dataset = prepare_interaction_pairs(XD, XT, Y, terows, tecols)
        # solve the problem of the last batch_size==1
        if len(train_dataset) % FLAGS.batch_size == 1:
            train_loader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
        if len(test_dataset) % FLAGS.batch_size == 1:
            test_loader = DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size, drop_last=True)
        else:
            test_loader = DataLoader(dataset=test_dataset, batch_size=FLAGS.batch_size)
        # model eval
        model = net(FLAGS).cuda()
        model.apply(weights_init)
        # train
        CI_list = []
        for epochind in range(FLAGS.num_epoch):
            model = train(train_loader, model, FLAGS)
            CI, loss, rm2, auc = test(model, test_loader, FLAGS)
            CI_list.append(CI)
            print('val: epoch={}, loss={:.5f}, CI={:.5f}, rm2={:.5f}, auc={:.5f}'.format(epochind, loss, CI, rm2, auc))
            if CI >= max(CI_list):
                path = FLAGS.checkpoint_path + "fold_" + str(fold_ind) + "_checkpoint.pth"
                torch.save(model, path)
        # print("fold_index=%d train over" % fold_ind)

        # save epoch and CI for other work
        path = FLAGS.result_path + "fold_" + str(fold_ind) + "_CI.txt"
        # print("save epoch and CI in ", path)
        np.savetxt(path, np.array(CI_list))

        # test
        affinity_ls = []
        pre_affinity_ls = []
        # Test with the best checkpoint obtained from this fold
        path = FLAGS.checkpoint_path + "fold_" + str(fold_ind) + "_checkpoint.pth"
        model = torch.load(path)
        model.eval()
        for drug_SMILES, target_protein, affinity in test_loader:
            # predicted affinity
            if "CoVAE" in str(net):
                pre_affinity, _, _, _, _, _, _, _, _ = model(drug_SMILES, target_protein, FLAGS)
            else:
                pre_affinity = model(drug_SMILES, target_protein)
            pre_affinity_ls += pre_affinity.cpu().detach().numpy().tolist()
            affinity_ls += affinity.cpu().detach().numpy().tolist()
        # every fold best epoch affinity
        pre_affinity_ls = np.array(pre_affinity_ls)
        affinity_ls = np.array(affinity_ls)

        # save affinity_ls and pre_affinity_ls for other work
        path = FLAGS.result_path
        # print("save affinity_ls and pre_affinity_ls in ", path)
        path_real = path + "fold_" + str(fold_ind) + "_affinity_ls.txt"
        np.savetxt(path_real, affinity_ls)
        path_pred = path + "fold_" + str(fold_ind) + "_pre_affinity_ls.txt"
        np.savetxt(path_pred, pre_affinity_ls)

        CI = get_cindex(affinity_ls, pre_affinity_ls)  # CI
        mse = get_mse(affinity_ls, pre_affinity_ls)  # mse
        mae = get_mae(affinity_ls, pre_affinity_ls)  # mae
        rm2 = get_rm2(affinity_ls, pre_affinity_ls)  # rm2
        if 'davis' in FLAGS.dataset_path:
            pre_label = pre_affinity_ls
            label = np.int32(affinity_ls > 7.0)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        if 'kiba' in FLAGS.dataset_path:
            pre_label = pre_affinity_ls
            label = np.int32(affinity_ls > 12.1)
            auc = roc_auc_score(label, pre_label)
            aupr = get_aupr(label, pre_label)
        print('test: Fold={:d}, CI={:.5f}, MSE={:.5f}, MAE={:.5f}, rm2={:.5f}, auc={:.5f}, aupr={:.5f}'
              .format(fold_ind, CI, mse, mae, rm2, auc, aupr))
        logging("test: Fold=%d, CI=%f, MSE=%f, MAE=%f, rm2=%f, auc=%f, aupr=%f"
                % (fold_ind, CI, mse, mae, rm2, auc, aupr), FLAGS)

        # every fold best epoch result (n_folds results)
        all_CI[fold_ind] = CI
        all_MSE[fold_ind] = mse
        all_MAE[fold_ind] = mae
        all_rm2[fold_ind] = rm2
        all_auc[fold_ind] = auc
        all_aupr[fold_ind] = aupr
        print("fold_index=%d test over" % fold_ind, "\n")
    # folds over return n_folds results
    return all_CI, all_MSE, all_MAE, all_rm2, all_auc, all_aupr

def n_fold_setting(XD, XT, Y, label_row_inds, label_col_inds, FLAGS, nfolds,i):
    test_set = nfolds[5] # 划分的最后一个数据集，作为测试集
    outer_train_sets = nfolds[0:5] # 前面的数据集

    foldinds = len(outer_train_sets)
    ## TRAIN AND VAL
    val_sets = []
    train_sets = []
    test_sets = []
    for val_foldind in range(foldinds):
        val_fold = outer_train_sets[val_foldind]
        val_sets.append(val_fold)
        otherfolds = deepcopy(outer_train_sets) # 前面的数据集，5
        otherfolds.pop(val_foldind) # 删去一个ls，4
        otherfoldsinds = [item for sublist in otherfolds for item in sublist]
        train_sets.append(otherfoldsinds) # 删去最后一个和某一个后，将所有的放进一个列表
        test_sets.append(test_set) # 选取最后一个
        print("total dataset split: (val, train, test)", foldinds+1)
        print("val set", str(len(val_sets)), ",\t set nfolds index: ", val_foldind) # 验证集：依次选择0 1 2 3 4
        print("train set", str(len(train_sets)), ",\t set nfolds len: ", foldinds+1-2) # 训练集：删去验证集和测试集后的数据集
        print("test set", str(len(test_sets)), ",\t set nfolds index: ", foldinds) # 测试集：nfolds[n-1]

    # debug
    if FLAGS.only_test: # only test
        all_CI, all_MSE, all_MAE, all_rm2, all_auc, all_aupr = \
            n_fold_only_test(XD, XT, Y, label_row_inds,label_col_inds, FLAGS, test_sets, i)
    else: # train + test
        all_CI, all_MSE, all_MAE, all_rm2, all_auc, all_aupr = \
            n_fold_train_test(XD, XT, Y, label_row_inds, label_col_inds, FLAGS, train_sets, test_sets, i)

    logging("---FINAL RESULTS-----", FLAGS)

    # result CI MSE MAE rm2 auc aupr
    result_CI = np.mean(all_CI)
    result_MSE = np.mean(all_MSE)
    result_MAE = np.mean(all_MAE)
    result_rm2 = np.mean(all_rm2)
    result_auc = np.mean(all_auc)
    result_aupr = np.mean(all_aupr)

    std_CI = np.std(all_CI)
    std_MSE = np.std(all_MSE)
    std_MAE = np.std(all_MAE)
    std_rm2 = np.std(all_rm2)

    logging("i=%d Test Performance: (CI=%f std_CI=%f) (MSE=%f std_MSE=%f) (MAE=%f std_MAE=%f) (rm2=%f std_rm2=%f) (auc=%f aupr=%f)" %
            (i, result_CI, std_CI, result_MSE, std_MSE, result_MAE, std_MAE, result_rm2, std_rm2, result_auc, result_aupr), FLAGS)
    logging("i=%d Test Performance: %.3f(%.3f)%.3f(%.3f)%.3f(%.3f)%.3f %.3f %.3f %.3f" %
        (i, result_CI, std_CI, result_MSE, std_MSE, result_MAE, std_MAE, result_rm2, std_rm2, result_auc, result_aupr),FLAGS)
    # return best result
    return result_CI, result_MSE, result_MAE, result_rm2, result_auc, result_aupr, std_CI, std_MSE, std_MAE, std_rm2

def experiment(FLAGS, foldcount=6):  # 5-fold cross validation + test 5折交叉验证+测试

    # Input
    # XD: [drugs, features] sized array (features may also be similarities with other drugs
    # XT: [targets, features] sized array (features may also be similarities with other targets
    # Y: interaction values, can be real values or binary (+1, -1), insert value float("nan") for unknown entries
    # perfmeasure: function that takes as input a list of correct and predicted outputs, and returns performance
    # higher values should be better, so if using error measures use instead e.g. the inverse -error(Y, P)
    # foldcount: number of cross-validation folds for settings 1-3, setting 4 always runs 3x3 cross-validation

    dataset = DataSet(fpath=FLAGS.dataset_path,  ### BUNU ARGS DA GUNCELLE
                      setting_no=FLAGS.problem_type,  ##BUNU ARGS A EKLE
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    # set character set size
    FLAGS.charseqset_size = dataset.charseqset_size # proteins序列
    FLAGS.charsmiset_size = dataset.charsmiset_size # drug SMILES序列

    XD, XT, Y = dataset.parse_data(FLAGS)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    drugcount = XD.shape[0]
    print("drug count after remove", drugcount)
    targetcount = XT.shape[0]
    print("target count after remove", targetcount)

    FLAGS.drug_count = drugcount
    FLAGS.target_count = targetcount

    label_row_inds, label_col_inds = np.where(np.isnan(Y) == False)
    # np.isnan(Y) == False: [num: true, nan: false]
    # Y[label_row_inds][label_col_inds]: 为数据(true)

    # ls_i = [i for i in range(0, 10)]
    ls_i = [0]
    # 10次循环后的结果
    if len(ls_i) > 1:
        CI = []
        MSE = []
        MAE = []
        rm2 = []
        AUC = []
        AUPR = []

    for i in ls_i:
        print("current seed:", i+1000)
        random.seed(i+1000)
        # nfolds表示 affinity!=nan 的affinity index，并进行train test val的划分
        if FLAGS.problem_type == 1:
            nfolds = get_random_folds(len(label_row_inds),foldcount)
        if FLAGS.problem_type == 2:
            nfolds = get_drugwise_folds(label_row_inds, label_col_inds, drugcount, foldcount) # 5-fold cross validation + test 5折交叉验证+测试
            # folds [[], [], [], [], [], []] 划分数据集[], 每个[]包含按照划分后的drug index排列的affinity index
        if FLAGS.problem_type == 3:
            nfolds = get_targetwise_folds(label_row_inds, label_col_inds, targetcount, foldcount)
        # debug
        result_CI, result_MSE, result_MAE, result_rm2, result_auc, result_aupr, std_CI, std_MSE, std_MAE, std_rm2 = \
            n_fold_setting(XD, XT, Y, label_row_inds, label_col_inds, FLAGS, nfolds, i)
        # 10次循环后的结果
        if len(ls_i) > 1:
            CI.append(result_CI)
            MSE.append(result_MSE)
            MAE.append(result_MAE)
            rm2.append(result_rm2)
            AUC.append(result_auc)
            AUPR.append(result_aupr)

    # 10次循环后的结果
    if len(ls_i) > 1:
        # save numpy result
        path = FLAGS.result_path + "result.txt"
        ls = [CI, MSE, MAE, rm2, AUC, AUPR]
        np.savetxt(path, np.asarray(ls))

        # result CI MSE MAE rm2 auc aupr
        result_CI = np.mean(CI)
        result_MSE = np.mean(MSE)
        result_MAE = np.mean(MAE)
        result_rm2 = np.mean(rm2)
        result_auc = np.mean(AUC)
        result_aupr = np.mean(AUPR)
        std_CI = np.std(CI)
        std_MSE = np.std(MSE)
        std_MAE = np.std(MAE)
        std_rm2 = np.std(rm2)
        logging(("-----Finally-----"), FLAGS)
        logging("i=end Test Performance: (CI=%f std_CI=%f) (MSE=%f std_MSE=%f) (MAE=%f std_MAE=%f) (rm2=%f std_rm2=%f) (auc=%f aupr=%f)" %
            (result_CI, std_CI, result_MSE, std_MSE, result_MAE, std_MAE, result_rm2, std_rm2, result_auc, result_aupr), FLAGS)
        logging("i=end Test Performance: %.3f(%.3f)%.3f(%.3f)%.3f(%.3f)%.3f %.3f %.3f %.3f" %
                (result_CI, std_CI, result_MSE, std_MSE, result_MAE, std_MAE, result_rm2, std_rm2, result_auc, result_aupr), FLAGS)

    print(FLAGS.log_dir)

if __name__ == "__main__":

    # begin_time
    TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    print(TIME)
    begin_time = time.time()

    FLAGS = argparser()

    FLAGS.dataset_path = FLAGS.dataset_path + FLAGS.dataset + "/"

    if 'davis' in FLAGS.dataset_path:
        FLAGS.max_smi_len = 85
        FLAGS.max_seq_len = 1200
    if 'kiba' in FLAGS.dataset_path:
        FLAGS.max_smi_len = 100
        FLAGS.max_seq_len = 1000
    # CoVAE
    if 'CoVAE' in str(net):
        FLAGS.batch_size = 256
        FLAGS.lr = 0.001

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    FLAGS.log_dir = os.path.join(FLAGS.log_dir, "log.txt")

    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)

    logging("current model:%s" % net, FLAGS)
    logging("arguments: sequence_Len=(%d %d), epoch=%d, batch_size=%d"
            % (FLAGS.max_smi_len, FLAGS.max_seq_len, FLAGS.num_epoch, FLAGS.batch_size), FLAGS)
    logging("arguments: log_dir=%s, checkpoint_path=%s, result_path=%s"
            % (FLAGS.log_dir, FLAGS.checkpoint_path, FLAGS.result_path), FLAGS)
    logging("arguments: dataset_path=%s, problem_setting(2:drug 3:proteins)=%d, only_test=%s"
            % (FLAGS.dataset_path, FLAGS.problem_type, FLAGS.only_test), FLAGS)

    for i in str(FLAGS).split(', '):
        print("FLAGS: ", i)
        logging("FLAGS: %s" % i, FLAGS)
    experiment(FLAGS)

    # end_time
    TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    print(TIME)
    end_time = time.time()
    running_time = end_time - begin_time
    hour = running_time // 3600
    minute = (running_time - 3600 * hour) // 60
    second = running_time - 3600 * hour - 60 * minute
    print('\n time cost : %d:%d:%d \n' % (hour, minute, second))