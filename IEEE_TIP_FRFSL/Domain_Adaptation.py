import numpy as np
import torch
from functools import partial

from numpy.ma.core import shape
from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb

def cos_batch_torch(x, y):
	'''Returns the cosine distance batchwise'''
	# x is the Sourse feature: bs * d * m * m
	# y is the Target feature: bs * d * nF
	# return: bs * n * m
	bs = x.shape[0]
	D = x.shape[1]
	assert(x.shape[1]==y.shape[1])
	#x = x.contiguous().view(bs, D, -1) # bs * d * m^2
	x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
	y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
	cos_dis = torch.bmm(torch.transpose(x,1,2), y)#.transpose(1,2)
	cos_dis = 1 - cos_dis # to minimize this value
	# TODO:
	beta = 0.1
	min_score = cos_dis.min()
	max_score = cos_dis.max()
	threshold = min_score + beta * (max_score - min_score)
	res = cos_dis - threshold
	return torch.nn.functional.relu(res.transpose(2,1))

def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50):
	C = C.float().cuda()
	T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
	temp = torch.bmm(torch.transpose(C,1,2), T)
	distance = batch_trace(temp, m, bs)
	return -distance


def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50):
	# C is the distance matrix
	# c: bs by n by m
	sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
	T = torch.ones(bs, n, m).cuda()
	A = torch.exp(-C/beta).float().cuda()
	for t in range(iteration):
		Q = A * T # bs * n * m
		for k in range(1):
			delta = 1 / (n * torch.bmm(Q, sigma))
			a = torch.bmm(torch.transpose(Q,1,2), delta)
			sigma = 1 / (float(m) * a)
		T = delta * Q * sigma.transpose(2,1)

	return T#.detach()

def GW_distance(X, Y, p, q, lamda=0.5, iteration=5, OT_iteration=20):
	'''
	:param X, Y: Source and target embeddings , batchsize by embed_dim by n
	:param p, q: probability vectors
	:param lamda: regularization
	:return: GW distance
	'''
	Cs = cos_batch_torch(X, X).float().cuda()
	Ct = cos_batch_torch(Y, Y).float().cuda()
	bs = Cs.size(0)
	m = Ct.size(2)
	n = Cs.size(2)
	T, Cst = GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
	temp = torch.bmm(torch.transpose(Cst,1,2), T)
	distance = batch_trace(temp, m, bs)
	return distance

def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
	one_m = torch.ones(bs, m, 1).float()
	one_n = torch.ones(bs, n, 1).float()
	Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
	      torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
	gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
	for i in range(iteration):
		C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))####改这一个
		gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
	Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
	return gamma.detach(), Cgamma

def GW_distance_uniform(X, Y, lamda=1e-1, iteration=5, OT_iteration=20):
	m = X.size(2)
	n = Y.size(2)
	bs = X.size(0)
	p = (torch.ones(bs, m, 1)/m)
	q = (torch.ones(bs, n, 1)/n)
	return GW_distance(X, Y, p, q, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

def batch_diag(a_emb, n, bs):
	a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1) # bs * n * n
	b = (a_emb.unsqueeze(1).repeat(1,n,1))# bs * n * n
	return a*b
	# diagonal bs by n by n

def batch_trace(input_matrix, n, bs):
	a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
	b = a * input_matrix
	return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)
#真正用的时候记得加CUDA！！！！！！！！！！
class GOT(nn.Module):
	def __init__(self,in_channelsx,in_channelsy,beta=0.5, iteration=5, OT_iteration=20,lamda = 1e-1):#改的这儿1e-1
		super(GOT, self).__init__()
		self.MLP1 = nn.Conv1d(in_channelsx,in_channelsx,kernel_size = 3,padding = 1)
		self.MLP2 = nn.Conv1d(in_channelsy,in_channelsy,kernel_size = 3,padding = 1)
		self.beta = 0.5#zheer0.5
		self.iter = iteration
		self.OT = OT_iteration
		self.lamda = lamda
	def forward(self,X,Y):
		Cs = cos_batch_torch(X, X).float().cuda()
		Ct = cos_batch_torch(Y, Y).float().cuda()
		Xi = self.MLP1(X)
		Yi = self.MLP2(Y)
		Cst_w = cos_batch_torch(Xi,Yi).float().cuda()#Cst_w==>F_st
		bs = Cs.size(0)
		m = Ct.size(2)#5
		n = Cs.size(2)#9
		p = (torch.ones(bs,m,1)/m).cuda()
		q = (torch.ones(bs,n,1)/n).cuda()
		one_m = torch.ones(bs, m, 1).float().cuda()
		one_n = torch.ones(bs, n, 1).float().cuda()
		#gamma就是T
		gamma = torch.bmm(p, q.transpose(2, 1))
		Cst_G = torch.bmm(torch.bmm(Cs ** 2, q), torch.transpose(one_m, 1, 2)) + torch.bmm(one_n, torch.bmm(torch.transpose(p, 1, 2), torch.transpose(Ct ** 2, 1, 2)))#Cst_G==>F_g
		#在顺一遍！！！！！！！！！！！！！！！！
		for i in range(self.iter):
			C_gamma = Cst_G - 2 * torch.bmm(torch.bmm(Cs, gamma.transpose(1,2)), torch.transpose(Ct, 1, 2))
			C_gamma = self.lamda * Cst_w.transpose(1,2) + (1 - self.lamda) * C_gamma###对应的是损失
			gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=self.beta, iteration=self.OT)###gamma==>H
			gamma = torch.transpose(gamma,1,2)
		C_gamma = Cst_G - 2 * torch.bmm(torch.bmm(Cs, gamma.transpose(1,2)), torch.transpose(Ct, 1, 2))
		temp = torch.bmm(torch.transpose(Cst_w, 1, 2), gamma)
		wd = batch_trace(temp, n, bs)
		temp = torch.bmm(gamma, C_gamma)
		gwd = batch_trace(temp, m, bs)#return wd
		return self.lamda * wd + (1-self.lamda) * gwd
if __name__ == '__main__':
	x = torch.rand(10,10,9)
	y = torch.rand(10,10,5)
	model = GOT(90,50)
	z = model(x,y)
	print(z)
