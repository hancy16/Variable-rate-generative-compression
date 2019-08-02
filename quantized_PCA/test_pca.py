#!/usr/bin/python3

import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import math
def load_Data():

	
	f = np.load('./train.npy')
	f = np.squeeze(f)
	#print(f.shape)
	Data_list = []
	for i in range(f.shape[0]):
		Data_list += num_split(f[i,:])   
	#print(len(Data_list))
	Data = np.array(Data_list)
	Data = np.squeeze(Data)
	#print(Data.shape)
	print("Data Loaded!")
	#print(Data.shape)
	return Data

def num_split(Data):
	Data = Data.reshape(32,64,16)
	list_tmp = []
	for i in range(0,32,8):
		for j in range(0,64,16):
			if i+8<32 & j+16<64:
				Data_tmp = Data[i:i+8,j:j+16,:]
				Data_tmp = Data_tmp.reshape(1,-1)
			#print(Data_tmp.shape)
				list_tmp.append(Data_tmp)
	#print(len(list_tmp))
	return list_tmp


def tsne_visualize(Data_list):
	Data = np.concatenate((Data_list[0],Data_list[1],Data_list[2],Data_list[3]),axis=0)
	#target_tmp = np.ones((300))
	#target = np.concatenate((target_tmp,2*target_tmp,target_tmp*3,target_tmp*4),axis=0)
	print(Data.shape)
	#print(target.shape)
	##print(Data[0:1,:])
	legend = ['1','2','3','4']
	tsne_data = TSNE(n_components=2).fit_transform(Data)
	f = plt.figure()
	plt.plot(tsne_data[0:300,0],tsne_data[0:300,1],'ro')
	plt.plot(tsne_data[300:600,0],tsne_data[300:600,1],'gx')
	plt.plot(tsne_data[600:900,0],tsne_data[600:900,1],'b*')
	plt.plot(tsne_data[900:1200,0],tsne_data[900:1200,1],'ko')
	plt.legend(legend)
	#plt.scatter(tsne_data[:,0],tsne_data[:,1],c=target)
	f.savefig('tsne', format='png', dpi=720, bbox_inches='tight', pad_inches=0)
	plt.gcf().clear()
	plt.close(f)

def MMD_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
	batch_size = int(source.shape[0])
	kernels = guassian_kernel(source, target,kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma )
	XX = kernels[:batch_size,:batch_size]
	YY = kernels[batch_size:,batch_size:]
	XY = kernels[:batch_size,batch_size:]
	YX = kernels[batch_size:,:batch_size]
	loss = torch.mean(XX+YY-XY-YX)
	return loss


 
def guassian_kernel(source_data, target_data, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

	source = Variable(torch.Tensor(source_data))
	target = Variable(torch.Tensor(target_data))

	n_samples = int(source.size()[0])+int(target.size()[0])
	total = torch.cat([source, target], dim=0)
	
	total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
	
	total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
	
	L2_distance = ((total0-total1)**2).sum(2)
	
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
	
	return sum(kernel_val)



def PCA_ana(Data_train):

	#Data_train = load_Data()
	pca = PCA(n_components=0.9)
	pca.fit(Data_train)
	print("PCA Model Trained!")
	f = open("./pca.pkl",'wb')
	pickle.dump(pca,f)
	f.close()
	return pca

def Redim(Data,level):
	f = open("numpyfile/pca.pkl",'rb')
	pca = pickle.load(f)
	f.close()

	Data_Cons = np.zeros_like(Data)
	Code = []
	Data_list = []
	for i in range(int(Data.shape[1]/8)):
		for j in range(int(Data.shape[2]/16)):
			Data_tmp = Data[:,8*i:8*i+8,16*j:16*j+16,:]
			Data_tmp = Data_tmp.reshape(1,-1)
			Data_list.append(Data_tmp)

			#print(Data_tmp.shape)
	Cons_X,NewX = Redim_split(pca,np.squeeze(np.array(Data_list)),level)
	var = 0
	for i in range(int(Data.shape[1]/8)):
		for j in range(int(Data.shape[2]/16)):
			Data_Cons[:,8*i:8*i+8,16*j:16*j+16,:] = Cons_X[var,:].reshape(1,8,16,16)
			var += 1
			
	Code = NewX.reshape(1,-1)
	#Code = Code.reshape(1,-1)
	return Data_Cons, Code

def Redim_split(pca,Data,level):
	#g = open("numpyfile/y_estimate.pkl",'rb')
	#y_estimate = pickle.load(g)
	#g.close()
	#ori_shape = Data.shape
	#Data = Data.reshape(1,-1)
	NewX = pca.transform(Data)
	#y_estimate_m = np.array([y_estimate for _ in range(NewX.shape[0])])
	NewX = np.round(NewX*level)
	#print(NewX.shape)
	#NewX_Z = NewX + np.min(NewX)
	#print(np.max(NewX_Z))
	Cons_X = pca.inverse_transform(NewX/level)
	#Cons_X = Cons_X.reshape(ori_shape)
	#print(Data)	
	#print(NewX)
	#print(np.log2(2*np.max(np.abs(NewX)))*NewX.shape[1]/(Data.shape[1]*np.log2(5)))
	return Cons_X,NewX


def SGD_t(pca,Data):
	Data_pca = pca.transform(Data)
	Sj = pca.inverse_transform(Data_pca)
	#Initisualize t 
	
	W =pca.components_
	W_matrix = W.T
	#Sj_t = Sj.copy()
	#kernels = guassian_kernel(Data, Sj , kernel_num=1, fix_sigma=fix_sigma)
	res = float("inf")
	thres = 10e-16
	max_iter = 100
	iter_num = 0
	mini_batch = 100
	#y = np.random.rand(Data_pca.shape[1])
	y = np.ones_like(Data_pca[0,:])
	
	SGD_y = np.zeros_like(y)

	while (res > thres) & (iter_num < max_iter):
		last_y = y.copy()
		#t = 
		last_t = np.matmul(W_matrix,last_y)
		rad1  = np.random.permutation(mini_batch)
		Data_new = Data[rad1,:]
		rad2  = np.random.permutation(mini_batch)
		Sj_new = Sj[rad2,:]
		#t_matrix = np.array([last_t for _ in range(Data_new.shape[0])])
		#Sj_t = Sj_new + t_matrix
		#kernels = guassian_kernel(Data_new, Sj_new , kernel_num=1)
		sum_kernel = 0.0
		lamda_tmp = 0.0
		k=9.0
		S_tmp = np.zeros_like(y)
		mu = 1.0/len(last_y)

		n_samples = float((1.0/Data_new.shape[0]**2)) 

		ru = []
		for i in range(Data_new.shape[0]):
			for j in range(Sj_new.shape[0]):
				ru.append(np.linalg.norm(Data_new[i,:]-Sj_new[j,:]-last_t))
		ru_array = np.array(ru)
		ru_mid   = 1.0/float(np.median(ru_array))
		delta_y = np.zeros_like(y)
		

		for i in range(Data_new.shape[0]):
			for j in range(Sj_new.shape[0]):
				Dis = math.exp(-np.linalg.norm(Data_new[i,:] - Sj_new[j,:] - last_t)**2*ru_mid)
				#print(Dis)
				lamda_tmp += (1.0-1.0/k*np.matmul(last_t,Data_new[i,:] - Sj_new[j,:]))*Dis
				S_tmp += np.matmul(W_matrix.T,Data_new[i,:] - Sj_new[j,:])*Dis
				sum_kernel += Dis
				delta_y += (last_y - np.matmul(W_matrix.T,Data_new[i,:] - Sj_new[j,:]))*Dis
				#print(i)
		#print(sum_kernel- lamda_tmp)	
		#print(Data_new.shape[0]**2)
		
		mul_t = float(n_samples*ru_mid)
		#print(mul_t)
		#print(mul_t*sum_kernel)
		
		lamda = mul_t*lamda_tmp - mu*len(last_y)/(2*k)
		print(sum_kernel*n_samples)
		
		print(lamda)
		#if lamda<0:
			#lamda = 0
		#print((1/Data_new.shape[0]**2)*sum_kernel-lamda)
		SGD_y = 0.1*SGD_y + 0.9*(-2.0*mul_t*delta_y+2.0*lamda*last_y+mu*1.0/y)
		y = 1.0*last_y + 0.6*SGD_y
		#y = 0.0*last_y + 1.0*(mul_t*S_tmp+0.5*mu*(1.0/last_y))/(mul_t*float(sum_kernel)-lamda)
		res = np.linalg.norm(last_y-y)
		iter_num += 1
		print("Current Iter: {} Loss: {} y_norm: {} MMD: {}".format(iter_num,res,np.linalg.norm(y),sum_kernel*float((1.0/Data_new.shape[0]**2))))
		#print(np.linalg.norm(y))
		
	print("Training Finished!")
	print(y)
	y = abs(y)
	#y = 1.0/y
	#print(np.min(y))
	y= y/np.min(y)
	print(np.max(y))
	print(np.linalg.norm(y))
	f = plt.figure()
	plt.hist(y)
	f.savefig('hist_y', format='png', dpi=720, bbox_inches='tight', pad_inches=0)
	plt.gcf().clear()
	plt.close(f)

	
	

def PCA_MMD(pca,Data):

	W =pca.components_
	W_matrix = W.T
	u,s,v = np.linalg.svd(W_matrix)
	#print(W_matrix.shape)
	Data_pca = pca.transform(Data)
	Sj = pca.inverse_transform(Data_pca)
	Datalen = Data.shape[0]
	#= np.zeros_like(Data[0,:])
	
	Data_mean = np.mean(Data-Sj,axis = 0)
	#print(Data_mean.shape)
	Data_nor = Data_mean/np.linalg.norm(Data_mean)
	Data_nor = np.squeeze(Data_nor)
	#print(Data_nor.shape)
	#Data_matrix = np.mat(Data_nor)	
	S_inv = np.zeros([v.shape[0],u.shape[0]])
	for i in range(len(s)):
		if s[i]:
			S_inv[i,i] = 1/s[i]
	#print(S_inv.shape)

	y_estimate = np.matmul(np.matmul(v.T,S_inv),np.matmul(u.T,Data_nor))
	y_estimate = abs(y_estimate)
	
	y_estimate = np.round(y_estimate/np.min(y_estimate))
	
	y_sort = y_estimate.copy()
	y_sort.sort()
	#print(y_sort)
	n_q = 16
	round_len = round(len(y_sort)/n_q)
	va_list = []
	va_list.append(0)
	for i in range(1,n_q):
		if i*round_len < len(y_sort):
			va_list.append(int(y_sort[i*round_len]))
	va_array = np.zeros(int(np.max(y_estimate))+1)
	va_list.append(len(va_array))
	#print(va_list)
	for j in range(len(va_list)-1):
		
		va_array[va_list[j]:va_list[j+1]] = va_list[j+1]
	va_array = va_array/np.min(va_array)
	#print(va_array)
	for n in range(len(y_estimate)):

		y_estimate[n] = va_array[int(y_estimate[n])]
	#print(y_estimate)
	#y_estimate = np.log(y_estimate)
	#print(y_estimate)
	#plt.legend(legend)
	
	#print(y_estimate.shape)
	#print(np.max(y_estimate))
	print("Round Matrix Trained!")
	#print(np.mean(y_estimate))
	#print(np.max(y_estimate))
	#print(np.min(y_estimate))
	g = open("./y_estimate.pkl",'wb')
	pickle.dump(y_estimate,g)
	g.close()
	#y_estimate = W_matrix.I*Data_matrix

def cos_vec(a,b):

	La = np.sqrt(a.dot(a))
	Lb = np.sqrt(b.dot(b))
	angle = a.dot(b)/(La*Lb)
	angle = np.arccos(angle)*360/2/np.pi
	print(angle)



if __name__ == '__main__':

	Data_list = load_Data()
	PCA_ana(Data_list)
	#f = open("./pca.pkl",'rb')
	#pca = pickle.load(f)
	#f.close()
	#SGD_t(pca,Data_list)
	#Cons_X,NewX = Redim(Data_list[0,:])
	#print(NewX.shape)
	#print(np.linalg.norm(Data_list[0,:]-Cons_X))
	#Cons_X1 = PCA_ana(Data_list[1])
	#tsne_visualize([Data_list[1],Data_list[1],Cons_X1,Cons_X1])
	#loss1 = MMD_rbf(Data_list[0],Data_list[1])
	#loss2 = MMD_rbf(Data_list[0],Data_list[2])
	#loss3 = MMD_rbf(Data_list[0],Data_list[3])
	#loss4 = MMD_rbf(Data_list[1],Data_list[2])
	#loss5 = MMD_rbf(Data_list[1],Data_list[3])
	#print(" loss1: {}\n loss2: {}\n loss3: {}\n loss4: {}\n loss5: {}".format(loss1,loss2,loss3,loss4,loss5))
	#loss =	MMD_rbf(Data_list[1],Cons_X1)
	#print(loss)

 
 

