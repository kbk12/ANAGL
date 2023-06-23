import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pickle
from model import ANAGL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 60
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lambda_3 = args.lambda3
dropout = args.dropout
lr = args.lr
svd_q = args.q

# load data  A 
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
#train_np = train.toarray()
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)

path = 'data/' + args.data + '/'
f_A = open(path+ 'kuai_train.pkl', 'rb')
train_A = pickle.load(f_A)
train_csr_A = (train!=0).astype(np.float32)

epoch_user = min(train.shape[0], 30000)
svd_q = 7
adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda()
adj_A = scipy_sparse_mat_to_torch_sparse_tensor(train_A).coalesce().cuda()
svd_u_A, s_A, svd_v_A = torch.svd_lowrank(adj_A, q=svd_q)
u_mul_s_A = svd_u_A @ torch.diag(s_A)
v_mul_s_A = svd_v_A @ torch.diag(s_A)
del adj_A 
del s_A
svd_u,s,svd_v = torch.svd_lowrank(adj,q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del adj
del s

rowD = np.array(train.sum(1)).squeeze()  
colD = np.array(train.sum(0)).squeeze() 

rowD_A = np.array(train_A.sum(1)).squeeze()
colD_A = np.array(train_A.sum(0)).squeeze()


for i in range(len(train.data)):             
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda()

for i in range(len(train_A.data)):
    train_A.data[i] = train.data[i] / pow(rowD_A[train_A.row[i]]*colD_A[train_A.col[i]], 0.5)

adj_norm_A = scipy_sparse_mat_to_torch_sparse_tensor(train_A)
adj_norm_A = adj_norm.coalesce().cuda()


test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []
model = GCL(adj_norm.shape[0], adj_norm.shape[1], adj_norm_A.shape[0], adj_norm_A.shape[1], d, u_mul_s, v_mul_s, u_mul_s_A, v_mul_s_A, \
                svd_u.T, svd_v.T, svd_u_A.T, svd_v_A.T, train_csr, train_csr_A, adj_norm, adj_norm_A, l, temp, lambda_1, lambda_2, lambda_3, dropout, batch_user )
model.cuda()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=lambda_2,lr=lr)
def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*0.98
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr
current_lr = lr
for epoch in range(epoch_no):
    current_lr = learning_rate_decay(optimizer)  
    e_users = np.random.permutation(adj_norm.shape[0])[:epoch_user]
    batch_no = int(np.ceil(epoch_user/batch_user))

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    for batch in tqdm(range(batch_no)):
        start = batch*batch_user
        end = min((batch+1)*batch_user,epoch_user)
        batch_users = e_users[start:end]
        pos = []
        neg = []
        iids = set()
        pos_A = []
        neg_A = []
        iids_A = set()
        for i in range(len(batch_users)):
            u = batch_users[i]
            u_interact = train_csr[u].toarray()[0]
            positive_items = np.random.permutation(np.where(u_interact==1)[0])
            negative_items = np.random.permutation(np.where(u_interact==0)[0])
            item_num = min(max_samp,len(positive_items))
            positive_items = positive_items[:item_num]
            negative_items = negative_items[:item_num]
            pos.append(torch.LongTensor(positive_items).cuda())
            neg.append(torch.LongTensor(negative_items).cuda())
            iids = iids.union(set(positive_items))
            iids = iids.union(set(negative_items))

            u_A = batch_users[i]
            u_interact_A = train_csr_A[u].toarray()[0]
            positive_items_A = np.random.permutation(np.where(u_interact_A==1)[0])
            negative_items_A = np.random.permutation(np.where(u_interact_A==0)[0])
            item_num_A = min(max_samp,len(positive_items_A))
            positive_items_A = positive_items_A[:item_num_A]
            negative_items_A = negative_items_A[:item_num_A]
            pos.append(torch.LongTensor(positive_items_A).cuda())
            neg.append(torch.LongTensor(negative_items_A).cuda())
            iids_A = iids_A.union(set(positive_items_A))
            iids_A = iids_A.union(set(negative_items_A))

        iids = torch.LongTensor(list(iids)).cuda()
        uids = torch.LongTensor(batch_users).cuda()
        
        uids_A = torch.LongTensor(batch_users).cuda()
        iids_A = torch.LongTensor(list(iids)).cuda()
        optimizer.zero_grad()
 
        uids_A = torch.LongTensor(batch_users).cuda()
        iids_A = torch.LongTensor(list(iids)).cuda()
        
        
        loss, loss_r, loss_s = model.forward_CL(uids, iids, uids_A, iids_A, pos, neg)
        
        loss.backward()
        optimizer.step()
  
        torch.cuda.empty_cache()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()
    
    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    if epoch % 3 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda()
            predictions = model.predict(test_uids_input)
            predictions = np.array(predictions.cpu())

            #top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('*'*20)
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda()
    predictions = model.predict(test_uids_input)
    predictions = np.array(predictions.cpu())

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('*'*20)
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)
metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y
})
current_t = time.gmtime()


