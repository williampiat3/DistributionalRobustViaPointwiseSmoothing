import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
from functools import reduce

# sys.path.append('/home/wpiat/Documents/gitlab/lipEstimation-master')

# from lipschitz_approximations import lipschitz_second_order_ub
# from lipschitz_utils import *

# sys.path.append('/home/wpiat/Documents/gitlab/lipopt-master')
# import old_code as po
# from old_code import utils

# sys.path.append('/home/wpiat/Documents/gitlab/lipopt-master')
# import polyopt as pol
# from polyopt import utilsl


import random
from deap import creator, base, tools, algorithms
from copy import deepcopy
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy
from copy import deepcopy
import argparse


def deactivate(model):
	for p in model.parameters():
		p.requires_grad=False

def activate(model):
	for p in model.parameters():
		p.requires_grad=True


def testing_convergence():
	#testing the convergence of the layer to 1/pi when he layer size increases
	for N in [100]:
		epochs= 30000
		lr = 0.1
		v = torch.randn(N)
		v = v/torch.norm(v,p=2)
		u = torch.randn(N)
		u = u/torch.norm(u,p=2)
		sigma = torch.rand(N)
		sigma.requires_grad=True
		for epoch in tqdm(range(epochs)):
			loss = torch.sum(torch.abs(v*u*sigma))
			loss.backward()
			with torch.no_grad():
				sigma.data = torch.clamp(sigma.data + lr*sigma.grad,0,1)
				sigma.grad.zero_()
		print(loss.item())

def compute_spectral_norm(weight,n_power_iterations,do_power_iteration=True,eps=1e-12):
	weight_mat = weight
	h, w = weight_mat.size()
	# randomly initialize `u` and `v`
	u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=eps)
	v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=eps)
	if do_power_iteration:
			with torch.no_grad():
				for _ in range(n_power_iterations):
					# Spectral norm of weight equals to `u^T W v`, where `u` and `v`
					# are the first left and right singular vectors.
					# This power iteration produces approximations of `u` and `v`.
					v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
					u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=eps, out=u)
				if n_power_iterations > 0:
					# See above on why we need to clone
					u = u.clone()
					v = v.clone()

	sigma = torch.dot(u, torch.mv(weight_mat, v))
	return sigma


def rvs(dim=3):
	 random_state = np.random
	 H = np.eye(dim)
	 D = np.ones((dim,))
	 for n in range(1, dim):
		 x = random_state.normal(size=(dim-n+1,))
		 D[n-1] = np.sign(x[0])
		 x[0] -= D[n-1]*np.sqrt((x*x).sum())
		 # Householder transformation
		 Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
		 mat = np.eye(dim)
		 mat[n-1:, n-1:] = Hx
		 H = np.dot(H, mat)
		 # Fix the last sign such that the determinant is 1
	 D[-1] = (-1)**(1-(dim % 2))*D.prod()
	 # Equivalent to np.dot(np.diag(D), H) but faster, apparently
	 H = (D*H.T).T
	 return H



class MLP(nn.Module):
	def __init__(self,number_layers,size,non_linearity=nn.ReLU):
		super(MLP,self).__init__()
		self.model =nn.Sequential(*[nn.Sequential(nn.Linear(size,size),non_linearity()) for _ in range(number_layers-1)])
		self.last_layer = nn.Linear(size,size)

	def forward(self,x):
		return self.last_layer(self.model(x))

	def compute_spectral_norm_(self,n_power_iterations):
		lip=1
		for module in self.modules():
			if isinstance(module,nn.Linear):
				lip *= compute_spectral_norm(module.weight,n_power_iterations)
		return lip





def test_virmaux_on_nns():
	n_power_iterations=20
	for N in [100]:
		model = MLP(N,100)
		lx1 = model.compute_spectral_norm_(n_power_iterations).item()
		model.double()
		X_train=torch.rand(1,100)
		input_size = X_train[[0]].size()
		#print(input_size)
		compute_module_input_sizes(model, input_size)
		lx = lipschitz_second_order_ub(model, algo='greedy')
		print("_____________________________")
		print("number of layers:",N)
		print(lx/lx1)

def reconfigure_ortho(ortho,index=0):
	"""
	function to positivise the column of index "index"
	"""

	return np.diag(sign_star(ortho[:,index]))@ortho


def generate_weight_pt(sizes):
	weights = []
	for i in range(len(sizes)-1):
		weight = torch.empty(sizes[i+1],sizes[i])
		nn.init.xavier_normal_(weight)
		if i ==0:
			
			u_old,s_old,v_old = np.linalg.svd(weight.cpu().numpy())
			s_old = np.ones(len(s_old))
			s_old[0]=2.
			u_old = reconfigure_ortho(u_old)
			v_old = reconfigure_ortho(v_old)
			weights.append(torch.tensor(u_old[:,:sizes[i]]@ np.diag(s_old)@ v_old[:sizes[i+1],:]))
			main_direct = torch.tensor(v_old[0])
		else:
			u_new,s_new,v_new =  np.linalg.svd(weight.cpu().numpy())
			s_new = np.ones(len(s_new))
			s_new[0]=2.
			u_new = reconfigure_ortho(u_new)
			v_new = reconfigure_ortho(v_new)
			weights.append(torch.tensor(u_new[:,:sizes[i]]@ np.diag(s_new)@ u_old.T[:sizes[i+1],:]))
			u_old,s_old,v_old = u_new,s_new,v_new
	return weights,main_direct

def sign_star(array):
	"""
	Small change in the sign function so that 0 values are counted as positive
	"""
	return np.sign(np.sign(array)*2+1)



def generate_weight_matrix(size,number):
	weights = []
	previous_orth = rvs(size)
	# previous_orth = reconfigure_ortho(previous_orth)

	main_direct = previous_orth.T[0]
	for k in range(number):
		new_orth = rvs(size)
		new_orth = reconfigure_ortho(new_orth)
		sig = np.random.rand(size)
		sig[0]=1.
		weights.append((new_orth@ np.diag(sig)@ previous_orth.T))
		previous_orth = new_orth
	return weights,main_direct

def generate_weight_matrix_last_not_aligned(size,number):
	weights = []
	previous_orth = rvs(size)
	# previous_orth = reconfigure_ortho(previous_orth)

	main_direct = previous_orth.T[0]
	for k in range(number-1):
		new_orth = rvs(size)
		new_orth = reconfigure_ortho(new_orth)
		sig = np.ones(size)
		sig[0]=2.
		weights.append((new_orth@ np.diag(sig)@ previous_orth.T))
		previous_orth = new_orth
	vect = previous_orth.T[0]
	new_orth1 = rvs(size)
	new_orth2 = rvs(size)
	sig = np.ones(size)
	sig[0]=2.
	new_weight = (new_orth2@ np.diag(sig)@ new_orth1.T)

	last_multiply = np.sqrt(np.sum((np.dot(new_weight.T,vect)**2)))/np.sqrt(np.sum(vect**2))
	weights.append(new_weight)

	return weights,main_direct,last_multiply


class PlaceHolder():
	def __init__(self,size,device,dtype):
		self._size=size
		self.dtype = dtype
		self.device = device
		self.values = torch.eye(size)

	def assign(self,list_values):
		diag = torch.Tensor(list_values[:self._size])
		diag = diag.to(dtype=self.dtype,device=self.device)
		self.values = torch.diag(diag)
		return list_values[self._size:]

	@property
	def size(self):
		return self._size

	@size.setter
	def size(self,new_size):
		self._size = new_size
		self.values = torch.eye(new_size)

	@property
	def weight(self):
		return self.values
	


	
	



def build_place_holders(layers,dtype=None,device=None):
	layers.reverse()
	matrixes = []
	for i,layer in enumerate(layers):
		if isinstance(layer,nn.Linear):
			input_size = layer.in_features
			matrixes.append(layer)
		if isinstance(layer,nn.ReLU) or isinstance(layer,nn.ELU):
			matrixes.append(PlaceHolder(size=input_size,device=device,dtype=dtype))
	return matrixes

def affect_values_and_process(matrixes,activations):
	for layer in matrixes:
		if isinstance(layer,PlaceHolder):
			activations= layer.assign(activations)
	evaluation = reduce(torch.mm,[layer.weight for layer in matrixes])
	return evaluation

		

def testing_virmaux_on_crafted_model():
	size=10
	layer1 = nn.Linear(size,size,bias=False)
	layer2 = nn.Linear(size,size,bias=False)
	model = nn.Sequential(layer1,nn.ReLU(),layer2)
	model.to(dtype=torch.double)
	model.eval()
	weights,main_direct = generate_weight_matrix(size)
	layer1.weight = nn.Parameter(torch.DoubleTensor(weights[0]))
	layer2.weight = nn.Parameter(torch.DoubleTensor(weights[1]))


	# print("input main direction norm")
	value_max=0

	for _ in range(1024):
		sigma = np.diag(np.random.randint(0,2,size))
		weights_prod = np.dot(np.dot(weights[0].T,sigma),weights[1].T)

		value = compute_spectral_norm(torch.tensor(weights_prod),n_power_iterations=100,do_power_iteration=True,eps=1e-12)
		if value > value_max:
			value_max=value
			sigma_max = sigma
	print(value_max)
	print(sigma_max)
	
	# print(torch.norm(torch.DoubleTensor(main_direct)).item())
	# print(torch.norm(model(torch.DoubleTensor(main_direct))).item())


	# print('main direction')
	# print(torch.norm(torch.DoubleTensor(main_direct)))
	# print(torch.norm(layer1(torch.DoubleTensor(main_direct))))


	X_train=torch.rand(1,size)
	input_size = X_train[[0]].size()
	compute_module_input_sizes(model, input_size)

	print("computation lipschitz constant")
	print(lipschitz_second_order_ub(model, algo='exact'))


def get_lengths(matrixes):
	l=[]
	for layer in matrixes:
		if isinstance(layer,PlaceHolder):
			l.append(layer.size)
	return l

def cross_over(ind1,ind2,sizes):
	activation=random.choice(range(len(sizes)))
	indexes = [sum(sizes[:i]) for i in range(len(sizes)+1)]

	ind1[indexes[activation]:indexes[activation+1]],ind2[indexes[activation]:indexes[activation+1]] = tools.cxTwoPoint(ind2[indexes[activation]:indexes[activation+1]], ind1[indexes[activation]:indexes[activation+1]])
	return ind1,ind2

def evalOneMax(individual,matrixes,n_power_iterations,do_power_iteration,eps):
	with torch.no_grad():
		return compute_spectral_norm(affect_values_and_process(matrixes,individual),n_power_iterations=n_power_iterations,do_power_iteration=do_power_iteration,eps=eps),




def run_lipschitz_algorithm(list_item,nb_individuals=20,NGEN=100,cxpb=0.5, mutpb=0.2,tournsize=3,indpb=0.02,dtype=None,device=None,verbose=False,matrixes=None):
	if matrixes is None:
		matrixes = build_place_holders(list_item,dtype=dtype,device=device)

	#print([layer.weight for layer in matrixes])
	#print(compute_spectral_norm(affect_values_and_process(matrixes,activations),n_power_iterations=100,do_power_iteration=True,eps=1e-12))
	sizes = get_lengths(matrixes)
	length = sum(sizes)

	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMax)

	toolbox = base.Toolbox()

	toolbox.register("attr_bool", random.randint, 0, 1)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=length)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	
	toolbox.register("evaluate", evalOneMax,matrixes=matrixes,n_power_iterations=10,do_power_iteration=True,eps=1e-12)
	toolbox.register("mate", cross_over,sizes=sizes)
	toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
	toolbox.register("select", tools.selTournament, tournsize=tournsize)

	population = toolbox.population(n=nb_individuals)

	for gen in range(NGEN):
		offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
		fits = toolbox.map(toolbox.evaluate, offspring)
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
		best_indiv = tools.selBest(population,k=1)[0]
		if verbose:
			print(best_indiv,best_indiv.fitness.values)
		population = toolbox.select(offspring, k=len(population))
	return best_indiv.fitness.values[0].item()
	# top10 = tools.selBest(population, k=10)

def run_lipschitz_on_different_layers(list_item,device=None,dtype=None):
	list_item.reverse()
	values=1
	for i,layer in enumerate(list_item):
		
		if i == 0:
			u,s,v = torch.svd(layer.weight.detach())
			sigma = s
			vk = v.t()
		elif isinstance(layer,nn.ReLU):
			u,s,v = torch.svd(list_item[i+1].weight.detach())

			uk_1 = u
			if i==(len(list_item)-2):
				sigma_1=s
			else:
				sigma_1 = torch.sqrt(s)

			left_member = torch.mm(torch.diag(sigma),vk)
			input_feature = left_member.shape[1]
			output_feature = left_member.shape[0]
			layer1 = nn.Linear(input_feature,output_feature,bias=False)
			layer1.weight = nn.Parameter(left_member)
			right_member = torch.mm(uk_1,torch.diag(sigma_1))

			layer2 = nn.Linear(output_feature,list_item[i+1].out_features,bias=False)
			layer2.weight = nn.Parameter(right_member)

			value = run_lipschitz_algorithm(list_item=[],matrixes=[layer1,PlaceHolder(size=input_feature,device=device,dtype=dtype),layer2],nb_individuals=60,NGEN=200,cxpb=0.5, mutpb=0.2,tournsize=3,indpb=0.02,dtype=dtype,device=device)
			values=values*value



			#run algorithm here to ge the constant
			sigma = sigma_1
			vk = v.t()
	return values


def build_M(weights,x,rho,alpha,beta):
	width,height = zip(*[(weight.shape[1],weight.shape[0]) for weight in weights])
	rows_A=[]
	rows_B = []

	for i in range(len(width)-1):
		row_A = []
		
		row_B = [np.zeros((width[i+1],width[0]))]

		for j in range(len(width)-1):		   
			if i!=j:
				row_A.append(np.zeros((height[j],width[j])))
				row_B.append(np.zeros((width[i+1],height[j])))
			if i==j:
				row_A.append(weights[j])
				row_B.append(np.eye(width[i+1]))


			
		row_A.append(np.zeros((height[i],width[-1])))
		rows_A.append(cp.atoms.affine.hstack.hstack(row_A))
		rows_B.append(cp.atoms.affine.hstack.hstack(row_B))

	A = cp.atoms.affine.vstack.vstack(rows_A)
	B = cp.atoms.affine.vstack.vstack(rows_B)

	AB = cp.atoms.affine.vstack.vstack((A,B))


	ucl= -2*alpha*beta*x
	ucr = (alpha+beta)*x
	lcl = (alpha+beta)*x
	lcr = -2*x
	uc = cp.atoms.affine.hstack.hstack((ucl,ucr))
	lc = cp.atoms.affine.hstack.hstack((lcl,lcr))
	interior_matrix = cp.atoms.affine.vstack.vstack((uc,lc))


	D = [np.zeros((w,w)) for w in width]
	C = deepcopy(D)
	C[0]=np.eye(width[0])
	D[-1]=weights[-1].T@weights[-1]

	return AB.T@interior_matrix@AB -rho*scipy.linalg.block_diag(*C) + scipy.linalg.block_diag(*D)

def run_lipSDP(list_layers,alpha,beta):
	weights = []
	size = 0
	first_layer=True
	for layer in list_layers:
		if isinstance(layer,nn.Linear):
			w = layer.weight.data.cpu().numpy().astype(np.float32)
			weights.append(w)
			if not first_layer:
				size += w.shape[1]
			else:
				first_layer=False
	x = cp.Variable((size,size),symmetric=True)
	rho = cp.Variable()
	rho.value = 10000
	M=build_M(weights,x,rho,alpha,beta)
	# exit()


	objective = cp.Minimize(rho)
	constraints = [M << 0]
	# constraints+= [cp.atoms.affine.upper_tri.upper_tri(x)<=0]
	constraints += [cp.atoms.affine.diag.diag(x)>=0]

	prob =  cp.Problem(objective,constraints)
	result = prob.solve(solver=cp.MOSEK,warm_start=True)
	return np.sqrt(rho.value)

def run_random(model,dim_input,sample=5000,dtype=None,device=None):
	X = (torch.rand((sample,dim_input),device=device,dtype=dtype)-0.5)
	X = X/(torch.sqrt(torch.sum(X**2,dim=1,keepdim=True)))
	with torch.no_grad():
		Y = model(X)
		norm = torch.sqrt(torch.sum(Y**2,dim=1))
	return torch.max(norm).item()

def run_CLIP(model,dim_input,nb_iter,iter_eps,dtype=None,device=None):
	X = torch.randn(dim_input,device=device,dtype=dtype)*0.01
	X = X/torch.sqrt(torch.sum(X**2))
	X.requires_grad=True

	for _ in range(nb_iter):
		loss = torch.sqrt(torch.sum(model(X)**2))
		loss.backward()
		with torch.no_grad():
			X.data = X + iter_eps*X.grad
			if torch.sum(X**2)>1:
					X.data = X.data/torch.sqrt(torch.sum(X.data**2))
			X.grad.zero_()
		print(loss.item())
	return loss.item()

def run_lipopt(model,dim_input,depth):
	lb = np.repeat(-1., dim_input)  # lower bounds for domain
	ub = np.repeat(1., dim_input)  # upper bounds for domain
	weights, biases = utils.weights_from_pytorch(model)
	fc = po.FullyConnected(weights, biases)
	f = fc.grad_poly
	g, lb, ub = fc.new_krivine_constr(p=1, lb=lb, ub=ub)
	m = po.KrivineOptimizer.new_maximize_serial(
			f, g, lb=lb, ub=ub, deg=len(weights),
			start_indices=fc.start_indices,
			layer_config=[dim_input]*depth,
			solver='gurobi', name='')
	lp_bound = m[0].objVal
	return lp_bound*np.sqrt(dim_input)

def compare_bounds(model):
    results = dict()
    weights, biases = utilsl.weights_from_pytorch(model)
    fc = pol.FullyConnected(weights, biases)
    f = fc.grad_poly
    g = fc.krivine_constr(p=1)
    m = pol.KrivineOptimizer.maximize(
            f, g, deg=len(weights)+8, start_indices=fc.start_indices,
            solver='gurobi', sparse=True, n_jobs=-1, name='')
    lp_bound = m.objVal
    return lp_bound

def compare_seqlip_genlip():
	device = torch.device("cuda")
	dtype = torch.double
	sizes = [1000]#,30,50,100,300,1000]
	nb_layers =[20]#5,10,15,20]
	
	x,y = np.meshgrid(sizes,nb_layers)
	# z_gen_layer_wise = np.zeros_like(x).astype(np.float32)
	z_gen_network_wise = np.zeros_like(x).astype(np.float32)
	z_lipsdp = np.zeros_like(x).astype(np.float32)
	z_virmaux = np.zeros_like(x).astype(np.float32)
	values = np.stack((x,y),axis=2)


	for k in tqdm(range(values.shape[0])):
		for j in range(values.shape[1]):
			list_layers = []
			size = values[k,j,0]
			nb_layer = values[k,j,1]
			weights,main_direct = generate_weight_matrix(size,nb_layer)
			for i in range(nb_layer-1):
				linear_layer = nn.Linear(size,size,bias=False)
				linear_layer.weight = nn.Parameter(torch.DoubleTensor(weights[i]))
				# linear_layer.weight.data.normal_(0.,1.)
				list_layers.append(linear_layer)
				list_layers.append(nn.ReLU())

			linear_layer = nn.Linear(size,size,bias=False)
			# linear_layer.weight.data.normal_(0.,1.)
			linear_layer.weight = nn.Parameter(torch.DoubleTensor(weights[-1]))
			list_layers.append(linear_layer)

			model = nn.Sequential(*list_layers)
			model.to(device=device,dtype=dtype)
			# print("__________________________")
			# print("number layers")
			try:
				z_lipsdp[k,j]=run_lipSDP(list_layers,alpha=0,beta=1)
			except:
				z_lipsdp[k,j]=0

			# z_lipsdp[k,j]=run_lipopt(model,dim_input=size,depth=nb_layer)/2**nb_layer
			# print(z_lipsdp[k,j])
			#print(size,nb_layer,z_virmaux[k,j])

			# z_gen_layer_wise[k,j]=run_lipschitz_on_different_layers(deepcopy(list_layers),device=device,dtype=dtype)/2**nb_layer
	
			try:
				z_gen_network_wise[k,j]=run_lipschitz_algorithm(deepcopy(list_layers),nb_individuals=100,NGEN=800,cxpb=0.5, mutpb=0.2,tournsize=3,indpb=0.02,dtype=dtype,device=device)
			except:
				z_gen_network_wise[k,j]=0
			# X_train=torch.rand(1,size)
			# input_size = X_train[[0]].size()
			# compute_module_input_sizes(model, input_size)

			# # # # print("computation lipschitz constant by Virmaux")
			# z_virmaux[k,j] = lipschitz_second_order_ub(model, algo='greedy')/2**nb_layer
			# # if size == 3:
			# # 	print(z_virmaux[k,j])
	print(z_lipsdp)
	print(z_gen_network_wise)
	exit()

	# print(z_gen_layer_wise)

	levels = [0,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1.01,1.1]
	cs = plt.contourf(x, y, z_lipsdp,levels=levels)
	# plt.contour(x,y,z_lipsdp,levels=levels)
	plt.xlabel(r"Size layer")
	plt.ylabel(r"Number of layers")
	plt.title(r"$L_{LipSDP}$")
	plt.colorbar(cs)
	plt.savefig("LipSDP_large.png")
	plt.clf()
	# cs = plt.contourf(x, y, z_gen_layer_wise,levels=levels)
	# plt.contour(x,y,z_gen_layer_wise,levels=levels)
	# plt.xlabel(r"Size layer")
	# plt.ylabel(r"Number of layers")
	# plt.title(r"$L_{SeqLip}$")
	# plt.colorbar(cs)
	# plt.savefig("seqlip.png")
	# plt.clf()
	cs = plt.contourf(x, y, z_gen_network_wise,levels=levels)
	# plt.contour(x,y,z_gen_network_wise,levels=levels)
	plt.xlabel(r"Size layer")
	plt.ylabel(r"Number of layers")
	plt.title(r"$L_{SeqLip}$")
	plt.colorbar(cs)
	plt.savefig("seqlip_network_large.png")
	plt.clf()
	# cs = plt.contourf(x, y, z_virmaux,levels=levels)
	# # plt.contour(x,y,z_virmaux,levels=levels)
	# plt.xlabel(r"Size layer")
	# plt.ylabel(r"Number of layers")
	# plt.title(r"$L_{VirSeqLip}$")
	# plt.colorbar(cs)
	# plt.savefig("seqlip_virmaux.png")
	# plt.clf()





	# layer1 = nn.Linear(10,5,bias=False)
	# layer2 = nn.Linear(5,40,bias=False)
	# layer3 = nn.Linear(40,3,bias=False)
	# list_item2 =[layer1,nn.ReLU(),layer2,nn.ReLU(),layer3]
	# model = nn.Sequential(*list_item2)
	# model.eval()


	# weights,main_direct = generate_weight_pt(sizes)
	# layer1.weight = nn.Parameter(torch.DoubleTensor(weights[0]))
	# layer2.weight = nn.Parameter(torch.DoubleTensor(weights[1]))
	# layer3.weight = nn.Parameter(torch.DoubleTensor(weights[2]))
	# model.to(dtype=torch.double,device=device)
	# print(run_lipschitz_on_different_layers(deepcopy(list_item2),device=device,dtype=dtype))
	

	# print(run_lipschitz_algorithm(deepcopy(list_item2),nb_individuals=50,NGEN=50,cxpb=0.5, mutpb=0.2,tournsize=3,indpb=0.02,dtype=dtype,device=device))

	# size = sizes[0]
	# X_train=torch.rand(1,size)
	# input_size = X_train[[0]].size()
	# compute_module_input_sizes(model, input_size)

	# print("computation lipschitz constant by Virmaux")
	# print(lipschitz_second_order_ub(model, algo='greedy'))

def run_function(size,nb_layer,method):
	device = torch.device("cuda")
	dtype = torch.double
	
	list_layers = []
	weights,main_direct = generate_weight_matrix(size,nb_layer)
	for i in range(nb_layer-1):
		linear_layer = nn.Linear(size,size,bias=False)
		linear_layer.weight = nn.Parameter(torch.DoubleTensor(weights[i]))

		list_layers.append(linear_layer)
		list_layers.append(nn.ReLU())

	linear_layer = nn.Linear(size,size,bias=False)
	linear_layer.weight = nn.Parameter(torch.DoubleTensor(weights[-1]))
	list_layers.append(linear_layer)

	model = nn.Sequential(*list_layers)
	model.to(device=device,dtype=dtype)
	deactivate(model)


	if method == "lipSDP":
		value=run_lipSDP(list_layers,alpha=0,beta=1)
	elif method == "SeqLip":
		X_train=torch.rand(1,size)
		input_size = X_train[[0]].size()
		compute_module_input_sizes(model, input_size)
		value = lipschitz_second_order_ub(model, algo='greedy')
	elif method == "genSeqLip":
		value = run_lipschitz_algorithm(deepcopy(list_layers),nb_individuals=100,NGEN=800,cxpb=0.5, mutpb=0.2,tournsize=3,indpb=0.02,dtype=dtype,device=device)
	elif method == "random":
		value = run_random(model,size,sample=50000,dtype=dtype,device=device)
	elif method == "CLIP":
		value = run_CLIP(model,size,nb_iter=100,iter_eps=0.01,dtype=dtype,device=device)
	return value
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-m","--method",type=str,choices=["lipSDP","SeqLip","genSeqLip","random","CLIP"],help="Method to execute")
	parser.add_argument("-s","--size",type=int,help="Size of the layers")
	parser.add_argument("-d","--depth",type=int,help="Depth")
	parser.add_argument("-f","--folder",type=str, help="Folder where to save the tests")

	args = parser.parse_args()

	save_file = "{0}/{1}_{2}_{3}.txt".format(args.folder,args.method,args.size,args.depth)

	with open(save_file,"w") as handle:
		handle.write("")

	value = run_function(args.size,args.depth,args.method)

	with open(save_file,"w") as handle:
		handle.write(str(value))






	# compare_seqlip_genlip()
	# exit()
	# # sizes=[10,20,40]
	# # weights,main_direct = generate_weight_pt(sizes)
	# # print(main_direct.shape)
	# # main_direct = main_direct.double()
	# # print(torch.norm(main_direct))
	# # print(torch.norm(main_direct.matmul(weights[0].t()).matmul(weights[1].t())))
	# device = torch.device("cuda")
	# dtype = torch.double
	# list_layers = []
	# size = 30
	# nb_layer = 4
	# weights,main_direct= generate_weight_matrix(size,nb_layer)
	# main_direct = torch.Tensor(main_direct)
	# main_direct = main_direct.to(device=device,dtype=dtype)
	# for i in range(nb_layer-1):
	# 	linear_layer = nn.Linear(size,size,bias=False)
	# 	w = torch.DoubleTensor(weights[i])
	# 	w.to(device=device,dtype=dtype)
	# 	linear_layer.weight = nn.Parameter(w)
	# 	# linear_layer.weight.data.normal_(0.,1.)
	# 	list_layers.append(linear_layer)
	# 	list_layers.append(nn.ReLU())

	# linear_layer = nn.Linear(size,size,bias=False)
	# # linear_layer.weight.data.normal_(0.,1.)
	# w = torch.DoubleTensor(weights[-1])
	# w.to(device=device,dtype=dtype)
	# linear_layer.weight = nn.Parameter(w)
	# list_layers.append(linear_layer)

	# model = nn.Sequential(*list_layers)
	# model.to(device=device,dtype=dtype)
	# print(run_lipSDP(list_layers,alpha=0,beta=1))
	# print(run_random(model,dim_input=size,sample=50000,dtype=dtype,device=device))
	# print(run_CLIP(model,dim_input=size,nb_iter=4000,iter_eps=0.0001,dtype=dtype,device=device))
	# X_train=torch.rand(1,size)
	# input_size = X_train[[0]].size()
	# compute_module_input_sizes(model, input_size)
	# print(lipschitz_second_order_ub(model, algo='greedy'))
	# with torch.no_grad():
	# 	print(torch.norm(model(main_direct)).item())


