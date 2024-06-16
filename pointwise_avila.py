import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from ruszczynski import Ruszczynski


import numpy as np
import matplotlib.pyplot as plt



import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.distributions as distrib
from modified_attacks import LinfPGDAttackNew

from tqdm import tqdm

import argparse

import pandas as pd
from sklearn import preprocessing

def deactivate(model):
	for p in model.parameters():
		p.requires_grad=False

def activate(model):
	for p in model.parameters():
		p.requires_grad=True


def n_function(x):
	y=torch.zeros_like(x)
	y[(x>=0)*(x<1)] = x[(x>0)*(x<1)]
	y[(x>=1)*(x<2)] = 2- x[(x>=1)*(x<2)]
	y[(x>=2)*(x<3)] = -2 + x[(x>=2)*(x<3)]
	return y



def create_fake_data(device,dtype,kind="gaussians",**kwargs):
	"""
	Function to create a toy dataset for experimenting on the robust objectives
	Arguments: kind string, name of hte dataset that you want to sample
			   kwargs: extra parameters for the distribution in case you want to change the shape, variance...
	"""
	if kind == "gaussians":
		mean1 = kwargs.get("mean1") if kwargs.get("mean1") is not None else torch.ones(2)
		mean2 = kwargs.get("mean2") if kwargs.get("mean2") is not None else torch.zeros(2)
		sigma1 = kwargs.get("sigma1") if kwargs.get("sigma1") is not None else 1.
		sigma2 = kwargs.get("sigma2") if kwargs.get("sigma2") is not None else 1.
		N = kwargs.get("N") if kwargs.get("N") is not None else 1000
		N_test = kwargs.get("N_test") if kwargs.get("N_test") is not None else 40

		X_train = torch.cat([torch.randn(int(N/2),2,device=device,dtype=dtype)*sigma1+mean1,torch.randn(N-int(N/2),2,device=device,dtype=dtype)*sigma2 + mean2],dim=0)
		Y_train = torch.cat([torch.ones(int(N/2),device=device,dtype=dtype),torch.zeros(N-int(N/2),device=device,dtype=dtype)],dim=0)

		X_test = torch.cat([torch.randn(int(N_test/2),2,device=device,dtype=dtype)*sigma1+mean1,torch.randn(N_test-int(N_test/2),2,device=device,dtype=dtype)*sigma2 + mean2],dim=0)
		Y_test = torch.cat([torch.ones(int(N_test/2),device=device,dtype=dtype),torch.zeros(N_test-int(N_test/2),device=device,dtype=dtype)],dim=0)

	elif kind =="doughnut":
		sigma1 = kwargs.get("sigma1") if kwargs.get("sigma1") is not None else 1.
		sigma2 = kwargs.get("sigma2") if kwargs.get("sigma2") is not None else 1.
		radius = kwargs.get("radius") if kwargs.get("radius") is not None else 2.
		N = kwargs.get("N") if kwargs.get("N") is not None else 1000
		N_test = kwargs.get("N_test") if kwargs.get("N_test") is not None else 40

		X_int = torch.randn(int(N/2),2,device=device,dtype=dtype)*sigma1
		X_ext = torch.randn(N-int(N/2),2,device=device,dtype=dtype)*sigma2 + sample_sphere(N-int(N/2),2,device=device,dtype=dtype)*radius

		X_train = torch.cat([X_int,X_ext],dim=0)
		Y_train = torch.cat([torch.ones(int(N/2),device=device,dtype=dtype),torch.zeros(N-int(N/2),device=device,dtype=dtype)],dim=0)


		X_int_test = torch.randn(int(N_test/2),2,device=device,dtype=dtype)*sigma1
		X_ext_test = torch.randn(N_test-int(N_test/2),2,device=device,dtype=dtype)*sigma2 + sample_sphere(N_test-int(N_test/2),2,device=device,dtype=dtype)*radius

		X_test= torch.cat([X_int_test,X_ext_test],dim=0)
		Y_test = torch.cat([torch.ones(int(N_test/2),device=device,dtype=dtype),torch.zeros(N_test-int(N_test/2),device=device,dtype=dtype)],dim=0)

	elif kind =="two moons":

		sigma = kwargs.get("sigma") if kwargs.get("sigma") is not None else 1.	
		radius = kwargs.get("radius") if kwargs.get("radius") is not None else 2.
		shift = kwargs.get("shift") if kwargs.get("shift") is not None else torch.tensor([[1.,-0.6]]).to(device=device,dtype=dtype)*radius
		N = kwargs.get("N") if kwargs.get("N") is not None else 1000
		N_test = kwargs.get("N_test") if kwargs.get("N_test") is not None else 40

		X_train = torch.randn(N,2,device=device,dtype=dtype)*sigma + sample_sphere(N,2,device=device,dtype=dtype)*radius
		Y_train = (X_train[:,1]>0).to(device=device,dtype=dtype)
		X_train[X_train[:,1]>0] += shift

		


		X_test = torch.randn(N_test,2,device=device,dtype=dtype)*sigma + sample_sphere(N_test,2,device=device,dtype=dtype)*radius
		Y_test = (X_test[:,1]>0).to(device=device,dtype=dtype)
		X_test[X_test[:,1]>0] += shift

	elif kind =="saw":
		sigma = kwargs.get("sigma") if kwargs.get("sigma") is not None else 1.	
		shift = kwargs.get("shift") if kwargs.get("shift") is not None else torch.tensor([[0.,1.]]).to(device=device,dtype=dtype)
		N = kwargs.get("N") if kwargs.get("N") is not None else 1000
		N_test = kwargs.get("N_test") if kwargs.get("N_test") is not None else 40
		x = torch.rand(N//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X0 = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+ torch.randn(N//2,2,device=device,dtype=dtype)*sigma
		x = torch.rand(N//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X1= torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+torch.randn(N//2,2,device=device,dtype=dtype)*sigma
		X_train = torch.cat((X0,X1+shift),dim=0)
		Y_train = torch.cat((torch.zeros(N//2,device=device,dtype=dtype),torch.ones(N//2,device=device,dtype=dtype)),dim=0)

		x = torch.rand(N_test//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X0 = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+ torch.randn(N_test//2,2,device=device,dtype=dtype)*sigma
		x = torch.rand(N_test//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X1= torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+torch.randn(N_test//2,2,device=device,dtype=dtype)*sigma
		X_test = torch.cat((X0,X1+shift),dim=0)
		Y_test = torch.cat((torch.zeros(N_test//2,device=device,dtype=dtype),torch.ones(N_test//2,device=device,dtype=dtype)),dim=0)
	elif kind == "mix":
		sigma = kwargs.get("sigma") if kwargs.get("sigma") is not None else 1.	
		shift = kwargs.get("shift") if kwargs.get("shift") is not None else torch.tensor([[1.5,1.5]]).to(device=device,dtype=dtype)
		N = kwargs.get("N") if kwargs.get("N") is not None else 1000
		N_test = kwargs.get("N_test") if kwargs.get("N_test") is not None else 40
		x = torch.rand(N//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X0 = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+ torch.randn(N//2,2,device=device,dtype=dtype)*sigma
		X1 = torch.randn(N//2,2,device=device,dtype=dtype)*sigma*3
		X_train = torch.cat((X0,X1+shift),dim=0)
		Y_train = torch.cat((torch.zeros(N//2,device=device,dtype=dtype),torch.ones(N//2,device=device,dtype=dtype)),dim=0)

		x = torch.rand(N_test//2,device=device,dtype=dtype)*3
		y = n_function(x)
		X0 = torch.cat((x.unsqueeze(1),y.unsqueeze(1)),dim=1)+ torch.randn(N_test//2,2,device=device,dtype=dtype)*sigma
		X1 = torch.randn(N_test//2,2,device=device,dtype=dtype)*sigma*3
		X_test = torch.cat((X0,X1+shift),dim=0)
		Y_test = torch.cat((torch.zeros(N_test//2,device=device,dtype=dtype),torch.ones(N_test//2,device=device,dtype=dtype)),dim=0)


		
	else:
		raise NotImplementedError
	return X_train,Y_train.unsqueeze(1),X_test,Y_test.unsqueeze(1)

def load_generic_base(base,dtype,device):
	if base == "avila":
		data = pd.read_csv("data/avila/avila-tr.txt")
		scaler = preprocessing.StandardScaler()
		preprocessor = preprocessing.LabelEncoder()
		X_train = data[["X"+str(i) for i in range(10) if i not in [5,9]]].values
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		preprocessor.fit(data['Y'])
		Y_train = preprocessor.transform(data['Y'])
		data = pd.read_csv("data/avila/avila-ts.txt")
		X_test = data[["X"+str(i) for i in range(10) if i not in [5,9]]].values
		Y_test = preprocessor.transform(data['Y'])
		X_test = scaler.transform(X_test)
		X_train,Y_train,X_test,Y_test = torch.Tensor(X_train).to(dtype=dtype,device=device),torch.Tensor(Y_train).to(device=device,dtype=torch.long),torch.Tensor(X_test).to(dtype=dtype,device=device),torch.Tensor(Y_test).to(device=device,dtype=torch.long)
	
	elif base == "grid":
		data = pd.read_csv("data/grid/Data_for_UCI_named.csv")
		X_pd = data[["tau1","tau2","tau3","tau4","p1","p2","p3","p4","g1","g2","g3","g4"]]
		preprocessor = preprocessing.LabelEncoder()
		preprocessor.fit(data['stabf'])
		Y_train = preprocessor.transform(data['stabf'].head(int(len(data)*0.8)))
		Y_test = preprocessor.transform(data['stabf'].tail(int(len(data)*0.2)))
		X_train = X_pd.head(int(len(data)*0.8)).values
		X_test = X_pd.tail(int(len(data)*0.2)).values
		X_train,Y_train,X_test,Y_test = torch.Tensor(X_train).to(dtype=dtype,device=device),torch.Tensor(Y_train).to(device=device,dtype=torch.long),torch.Tensor(X_test).to(dtype=dtype,device=device),torch.Tensor(Y_test).to(device=device,dtype=torch.long)

	elif base == "MNIST":
		train_dataset = torchvision.datasets.MNIST(**{"root":"data", "train":True, "transform":transforms.Compose([transforms.Resize(size=(14,14)),transforms.ToTensor()]), "download":True})
		X_train,Y_train = zip(*train_dataset)
		X_train = torch.stack(X_train,dim=0).to(device=device,dtype=dtype)
		X_train = X_train.reshape(X_train.shape[0],196)
		Y_train = torch.tensor(Y_train).to(device=device)
		test_dataset = torchvision.datasets.MNIST(**{"root":"data", "train":False, "transform":transforms.Compose([transforms.Resize(size=(14,14)),transforms.ToTensor()]), "download":True})
		X_test,Y_test = zip(*test_dataset)
		X_test = torch.stack(X_test,dim=0).to(device=device,dtype=dtype)
		X_test = X_test.reshape(X_test.shape[0],196)
		Y_test = torch.tensor(Y_test).to(device=device)
	elif base == "MNISTlatent":
		
		X_train,Y_train = torch.load("data/mnist_latent_train.pt"),torch.load("data/mnist_latent_train_label.pt")
		X_test,Y_test = torch.load("data/mnist_latent_test.pt"),torch.load("data/mnist_latent_test_label.pt")
		X_train = X_train.to(dtype=dtype,device=device)
		Y_train = Y_train.to(device=device,dtype=torch.long)
		X_test = X_test.to(dtype=dtype,device=device)
		Y_test = Y_test.to(device=device,dtype=torch.long)
	
	# 	pass

	# elif base == "segmentation":
	# 	pass
	else:
		raise NotImplementedError("This set is not implemented yet")

	return X_train,Y_train,X_test,Y_test


def select_base(base,dtype,device):
	"""
	Simple overlay of the create_fake data function with kwargs hardcoded so as to cclean the main functions
	"""

	if base=="gaussian":

		gaussian_kwargs ={
		"mean1" : torch.ones(2),
		"mean2" : torch.zeros(2),
		"sigma1" : 0.5,
		"sigma2" : 0.5,
		"N" : 1000,
		"N_test":40,

		}
		X_train,Y_train,X_test,Y_test = create_fake_data(device,dtype,kind="gaussians",**gaussian_kwargs)

	if base == "doughnut":

		doughnut_kwargs ={
		"radius":2,
		"sigma1" : 0.3,
		"sigma2" : 0.3,
		"N" : 1000,
		"N_test":300,

		}
		X_train,Y_train,X_test,Y_test = create_fake_data(device,dtype,kind="doughnut",**doughnut_kwargs)

	if base=="two moons":

		two_moons_kwargs = {

		"radius":2,
		"sigma" : 0.2,
		"N" : 2000,
		"N_test":3000,
		"shift": torch.tensor([[1.,-0.5]]).to(device=device,dtype=dtype)*2,


		}
		X_train,Y_train,X_test,Y_test = create_fake_data(device,dtype,kind="two moons",**two_moons_kwargs)

	if base == "saw":
		saw_kwargs = {
		"sigma" : 0.05,
		"N" : 2000,
		"N_test":3000,
		"shift": torch.tensor([[0.,1.]]).to(device=device,dtype=dtype),


		}
		X_train,Y_train,X_test,Y_test = create_fake_data(device,dtype,kind="saw",**saw_kwargs)
	if base == "mix":
		saw_kwargs = {
		"sigma" : 0.05,
		"N" : 2000,
		"N_test":3000,
		"shift": torch.tensor([[1.5,1.5]]).to(device=device,dtype=dtype),


		}
		X_train,Y_train,X_test,Y_test = create_fake_data(device,dtype,kind="mix",**saw_kwargs)
	return X_train,Y_train,X_test,Y_test

	


class GeneralizedRelu(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 0] = 1
        grad_input[input == 0] = torch.rand(*grad_input[input == 0].shape,dtype=input.dtype,device=input.device )
        return grad_input*grad_output

class MLPModel(nn.Module):
	"""
	Simple MLP for testing the new approaches
	"""
	def __init__(self,output_classes,non_linearity):
		super(MLPModel,self).__init__()
		self.layer1 = nn.Linear(8,200)
		self.layer2 = nn.Linear(200,200)
		self.layer3= nn.Linear(200,output_classes)
		self.input_sizes = (-1,2)
		self.relu = non_linearity


	def forward(self,x):
		return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

	def compute_lipschitz_constant(self):

		with torch.no_grad():
			return compute_spectral_norm(self.layer1.weight,n_power_iterations=100,do_power_iteration=True,eps=1e-12)*compute_spectral_norm(self.layer2.weight,n_power_iterations=100,do_power_iteration=True,eps=1e-12)*compute_spectral_norm(self.layer3.weight,n_power_iterations=100,do_power_iteration=True,eps=1e-12)




def sample_sphere(*shape,device=None,dtype=None):
	"""
	This function samples uniformly the sphere given a certain shape
	"""
	noise = torch.randn(*shape,device=device,dtype=dtype)
	#projecting on the sphere
	noise_sphere = noise/torch.sqrt(torch.sum(noise**2,dim=-1,keepdim=True))

	return noise_sphere




def build_adversarial_noise(X,Y,attack_class,**kwargs):
	"""
	Function using an attack too build adversarial noise
	Arguments: X,Y input data
			   model NN to fool
			   attack_class: attack class
	"""
	attack = attack_class(**kwargs)
	X_pert = attack.perturb(X,Y)
	# plt.plot(attack.monitored_loss)
	return X_pert

def build_l2_noise(points,shape,radius=1.,device=None,dtype=None):
	"""
	Function to sample the L2 ball around points given, process that we kept differentiable
	Arguments: points: torch tensor to perturb
			   shape: extra shape to add 
			   radius: saclar value of the ball to sample
			   device and dtype for sampling in the right data format
	"""
	noise_sphere = sample_sphere(*shape,*points.shape,device=device,dtype=dtype)
	# Scaling to the ball with an uniform draw to the d dimensionnal ball
	noise_ball = noise_sphere*torch.rand(*noise_sphere.shape[:-1],1,device=device,dtype=dtype)**(1/points.shape[-1])
	return noise_ball*radius + points.view(*[1 for _ in shape],*points.shape)

def sample_lp_ball(*shape,p=2,device=None,dtype=None):
	"""
	Function to sample the p dimension ball, the samples are normalised according to the last dimension
	Arguments: shape: sample shape
				p : norm selection

	"""
	if p==2:
		noise = torch.randn(*shape,device=device,dtype=dtype)
		noise = noise/((noise**2.).sum(-1).unsqueeze(-1))*torch.rand(*shape[:-1],1,device=device,dtype=dtype)**(1/shape[-1])
	elif p!=np.inf:
		noise = sample_gen_gaussian(*shape,p=p,device=device,dtype=dtype)
		z = torch.tensor(np.random.exponential(size=shape[:-1]),device=device,dtype=dtype)
		norm = (torch.sum(torch.abs(noise)**p,dim=-1)+z)**(1/p)
		noise = noise/norm.unsqueeze(-1)
	elif p==np.inf:
		noise = (torch.rand(*shape,device=device,dtype=dtype)-0.5)*2

	return noise
def sample_gen_gaussian(*shape,p=2,device=None,dtype=None):
	"""
	Function to sample a generalised  normalised centrered gaussian law: pdf proportionnal to exp(-|x|^p)
	if p=2 we have a gausian distribution for instance
	Even thoug the generalised gaussian distribution is implemented in scipy their implementation for sampling is based on
	the inverse cdf algorithm that is quite slow
	Here we have chosen to follow a faster sampling method presented here:
	https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function
	Usig a Bernoulli and gamma distribution
	"""
	gamma_dist = distrib.gamma.Gamma(torch.tensor([1.0])/p,torch.tensor([1.0]))
	bern_dist = distrib.bernoulli.Bernoulli(probs=torch.tensor([.5]))
	Y = gamma_dist.sample(sample_shape=torch.Size(shape)).to(device=device,dtype=dtype).squeeze()
	S = (bern_dist.sample(sample_shape=torch.Size(shape)).to(device=device,dtype=dtype).squeeze()-0.5)*2
	return S*(Y**(1/p))


def hinge_loss(output,label,reduction="mean",relu=torch.relu):
	"""
	function to compute the hinge loss, it converge in case the data is seperable
	"""

	y=(label-0.5)*2
	losses = relu(1-y*output)
	if reduction=="mean":

		return torch.mean(losses)
	if reduction == "sum":
		return torch.sum(losses)
	if reduction == "none":
		return losses

def noise_randomly(ball_points_number,x,radius,p,device=None,dtype=None):
	if p == np.inf:
		noisy_inputs = x + (torch.rand(*(ball_points_number,*x.shape),device=device,dtype=dtype)-0.5)*2*radius
	elif p != 2 :
		noisy_inputs = x + sample_lp_ball(*(ball_points_number,*x.shape),p=p,device=device,dtype=dtype)*radius
	elif p == 2:
		noisy_inputs = build_l2_noise(x,(ball_points_number,),radius=radius,device=device,dtype=dtype)
	return noisy_inputs

def noise_sphere_randomly(ball_points_number,x,radius,p,device=None,dtype=None):
	if p == np.inf:
		noisy_inputs = x + (torch.rand(*(ball_points_number,*x.shape),device=device,dtype=dtype)-0.5)*2*radius
	elif p != 2 :
		noisy_inputs = x + sample_lp_ball(*(ball_points_number,*x.shape),p=p,device=device,dtype=dtype)*radius
	elif p == 2:
		noisy_inputs = build_l2_noise(x,(ball_points_number,),radius=radius,device=device,dtype=dtype)
	return noisy_inputs


def train(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,scheduler,**kwargs):
	"""
	Training Loop
	"""

	dataset = TensorDataset(X_train,Y_train)
	train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
	for epoch in tqdm(range(epochs)):
		for data in train_loader:
			x,y = data
			output = model(x)
			loss = criterion(output,y)
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()



def adversarial_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,scheduler,attack_cls=None,kwargs_attack=None,**kwargs):
	"""
	Training Loop for an adversarial training that perturbs the dataset at each step
	"""
	dataset = TensorDataset(X_train,Y_train)
	train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
	for epoch in range(epochs):
		for data in train_loader:
			x,y = data
			#Changing locally the kwargs of the attack as the computation is far too long
			# kwargs_attack["nb_iter"]//=10
			# kwargs_attack["eps_iter"]*=10
			if epoch > 2:
				deactivate(model)
				x_adv = build_adversarial_noise(x,y,attack_cls,predict=model,**kwargs_attack)
				activate(model)
				output = model(x_adv)

			else:
				output = model(x)
			
			loss = criterion(output,y)
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

def langevin_sampling(model,data,label,criterion,gamma,delta,radius,eps_iter,iterations,device=None,dtype=None):
	"""
	Function to use Langevin sampling in order to boost the regions of the space explored
	"""
	x = data.clone()
	x.requires_grad=True
	model.requires_grad_(False)
	datas = [x.detach().clone()]
	for i in range(iterations-1):
		loss = criterion(model(x),label)
		loss.backward()
		with torch.no_grad():
			x.data= x.data + (gamma*x.grad + delta*(torch.rand(*x.shape,device=device,dtype=dtype)-0.5)*2)*eps_iter
			x.grad.zero_()
			x.data =  torch.clamp(x.data-data,-radius,radius) + data
			datas.append(x.detach().clone())
	
	model.requires_grad_(True)
	return torch.stack(datas,dim=0)



def log_sum_exp_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,ball_points_number,radius,temperature_generator,scheduler,p=2,device=None,dtype=None,**kwargs):
	"""
	Training computing the logsumexp computation of the supremum:
	Extra Arguments compared to classic training:
		ball_points_number: number of points to assess the supremum value
		radius: radius of the ball that is sampled
		temperature_generator: scaling factor for the logsumexp operation in the form of a generator, allows some fine tuning of the temperature decay
	"""
	dataset = TensorDataset(X_train,Y_train)
	train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
	for epoch in tqdm(range(epochs)):
		for data in train_loader:
			x,y = data
			with torch.no_grad():
				noisy_inputs = noise_randomly(ball_points_number,x,radius,p,device=device,dtype=dtype)

			output = model(noisy_inputs)
			batch_nb = x.shape[0]
			

			loss = criterion(output.view(ball_points_number*batch_nb,output.shape[-1]),y.unsqueeze(0).repeat(ball_points_number,*[1 for _ in y.shape]).view(ball_points_number*batch_nb)).view(ball_points_number,batch_nb)
			temperature = next(temperature_generator)
			loss = torch.mean(temperature*torch.logsumexp(loss/temperature,dim=0))
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()

def langevin_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,radius,scheduler,gamma,delta,eps_iter,iterations,criterion_langevin,temperature_generator,p=np.inf,device=None,dtype=None,**kwargs):
	"""
	Training computing the logsumexp computation of the supremum but with langevin sampling:
	Extra Arguments compared to classic training:
		ball_points_number: number of points to assess the supremum value
		radius: radius of the ball that is sampled
		
	"""
	dataset = TensorDataset(X_train,Y_train)
	train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0)
	for epoch in tqdm(range(epochs)):
		for data in train_loader:
			x,y = data
			noisy_inputs = langevin_sampling(model,x,y,criterion_langevin,gamma,delta,radius,eps_iter,iterations,device=device,dtype=dtype)
			optimizer.zero_grad()
			output = model(noisy_inputs)
			batch_nb = x.shape[0]


			criterion(output.view(iterations*batch_nb,output.shape[-1]),y.unsqueeze(0).repeat(iterations,*[1 for _ in y.shape]).view(iterations*batch_nb))

			loss = criterion(output.view(iterations*batch_nb,output.shape[-1]),y.unsqueeze(0).repeat(iterations,*[1 for _ in y.shape]).view(iterations*batch_nb)).view(iterations,batch_nb)
			temperature = next(temperature_generator)
			loss = torch.mean(temperature*torch.logsumexp(loss/temperature,dim=0))
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()


def max_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,ball_points_number,radius,p,scheduler,device=None,dtype=None,**kargs):
	"""
	This function does the max training for a specific case as we cannot have the max location for every possible case
	here the max is implemented for the swiss rolls, we have that the worst case is when we push the samples toward the decision boundary
	this is made for the infinite ball
	"""
	dataset = TensorDataset(X_train,Y_train)
	train_loader = DataLoader(dataset,batch_size=25,shuffle=True,num_workers=0)
	for epoch in range(epochs):
		for data in train_loader:
			x,y = data
			with torch.no_grad():
				noisy_inputs = noise_randomly(ball_points_number,x,radius,p,device=device,dtype=dtype)

			output = model(noisy_inputs)
			loss = criterion(output,y.unsqueeze(0).repeat(ball_points_number,*[1 for _ in y.shape]))
			loss = torch.mean(torch.max(loss,dim=0)[0])
			loss.backward()
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()


def training_lee(model, criterion, X_train,Y_train, optimizer,epochs,radius,alpha,steps,device=None,dtype=None,**kwargs ):
	"""
	Function to train the model based on Solving a Class of Non-Convex Min-Max Games Using Iterative First Order Methods
	instead of making the network train on random adversaries we make it train against targeted attacks
	"""

	for epoch in tqdm(range(epochs)):
		
		truth =  Y_train
		input_model = X_train
		# creating perturbated data by multiplicating first dim
		targeted_adversaries = generate_targeted_adversaries(model,input_model,truth,radius,alpha,steps)
		output = model(targeted_adversaries)
		
		loss = criterion(output,Y_train.unsqueeze(0).repeat(2,*[1 for _ in Y_train.shape]))
		loss = torch.mean(torch.max(loss,dim=0)[0])
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



def compute_accuracy(model,X_test,Y_test):
	"""
	Function to compute the accuracy of the model
	"""
	with torch.no_grad():
		output = model(X_test)
		inter = (output>0).long()-Y_test.long()

		accuracy = torch.sum((inter==0.)).float()/X_test.shape[0]
	return accuracy

#function to compute the accuracy of the model on a test set multiple labels
def computing_accuracy(model,x,y,batch_size):
	model.eval()
	eps=0.
	size = x.shape[0]
	count=0
	if batch_size < x.shape[0]:
		with torch.no_grad():
			for i in range(0,x.shape[0],batch_size):
				count += batch_size
				output = model(x[i:i+batch_size])
				eps+= torch.sum(torch.argmax(output,dim=1)==y[i:i+batch_size]).item()
			return eps/count
	else:
		output = model(x)
		return torch.mean((torch.argmax(output,dim=1)==y).float()).item()


#function to compute the confusion matrix of the model on a test set multiple labels
def computing_cm(model,x,y,batch_size):
	model.eval()
	size  = int((torch.max(y)+1).item())

	cm = torch.zeros(size,size,device=x.device)
	if batch_size < x.shape[0]:
		with torch.no_grad():
			for i in range(0,x.shape[0],batch_size):
				output = torch.argmax(model(x[i:i+batch_size]),dim=1)
				truth = y[i:i+batch_size]
				for k,l in zip(output,truth):
					cm[k,l]+=1
			
	else:
		output = torch.argmax(model(x),dim=1)
		for k,l in zip(output,y):
			cm[k,l]+=1
	return cm/torch.sum(cm,dim=1).unsqueeze(1)





def compute_worst_case_accuracy(model,X_test,Y_test,ball_points_number,radius,p=2,device=None,dtype=None):
	dataset = TensorDataset(X_test,Y_test)
	train_loader = DataLoader(dataset,batch_size=25,shuffle=True,num_workers=0)
	results = []
	for data in train_loader:
		x,y = data
		with torch.no_grad():
			noisy_inputs = noise_randomly(ball_points_number,x,radius,p,device=device,dtype=dtype)
			output = model(noisy_inputs)
			inter = (torch.abs((output>0).long()-y.unsqueeze(0).repeat(ball_points_number,*[1 for _ in y.shape]).long())==0.).float()
			pred = torch.min(inter,dim=0)[0]
			acc = torch.sum((pred)).float()/x.shape[0]
			results.append(acc.item())
	return np.mean(results)



def compute_adversarial_accuracy(model,X_test,Y_test,attack_cls,kwargs_attack):
	"""
	Function to compute the adversarial accuracy on the test set
	"""
	X_test_adv = build_adversarial_noise(X_test,Y_test,attack_cls,predict=model,**kwargs_attack)
	return compute_accuracy(model,X_test_adv,Y_test)


def compute_robust_accuracy(model,X_test,Y_test,ball_points_number,radius,p=2,device=None,dtype=None):
	"""
	Function to compute the robustness of the network by computing the accuracy over a ball around the samples
	Arguments:
		model: network to probe
		X_test: test base
		Y_test: test label
		ball_points_number: number of points for the assesment of the robust accuracy
		radius: float for the assement of the robust accuracy
		p: integer norm used for the ball
	"""
	with torch.no_grad():
		noisy_inputs = X_test + sample_lp_ball(*(ball_points_number,*X_test.shape),p=p,device=device,dtype=dtype)*radius
		output = model(noisy_inputs)
		inter = (torch.abs((output>0).long()-Y_test.unsqueeze(0).repeat(ball_points_number,*[1 for _ in Y_test.shape]).long())==0.).float()
		pred = torch.mean(inter,dim=0)
		acc = torch.sum((pred)).float()/X_test.shape[0]
		return acc

		
def compute_bound(rx,lx,dx,tau,volx,volbx,d):
	"""
	Function to compute the bound using robust training over max training
	"""
	return tau*lx*(rx+dx) + tau*np.log(volx/volbx)- tau*np.log(tau)*d




def decaying_temperature(initial_value,final_value,decay):
	"""
	Python Generator for the temperature: allows fine tuning of the temperation during the iterations
	here this is a simple linear decreasing temperature
	"""
	t=0
	while True:
		yield initial_value - (initial_value-final_value)*min(1,decay*t)
		t+=1

def train_macro(loss,mode,model,X_train,Y_train,optimizer,epochs,batch_size,device,dtype,**kwargs):
	"""
	Function to simplify main functions and remove forks
	"""



	if mode == "regular":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss()
		elif loss =="hinge":
			criterion =  lambda x,y: hinge_loss(x,y,reduction="mean",relu=F.elu)
		elif loss == "Xent":
			criterion = nn.CrossEntropyLoss()
		return train(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,**kwargs)
	

	if mode == "robust":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss(reduction="none")
		elif loss =="hinge":
			criterion = lambda x,y: hinge_loss(x,y,reduction="none",relu=F.elu)
		elif loss == "Xent":
			criterion = nn.CrossEntropyLoss(reduction="none")

		# Be careful of Out Of Memory error when increasing too much the batch size
		log_sum_exp_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,device=device,dtype=dtype,**kwargs)
		# else:
		# 	log_sum_exp_training_extended_batch(model,X_train,Y_train,optimizer,criterion,epochs,batch_size=200000,repetitions=math.ceil(batch_size/200000),device=device,dtype=dtype,**kwargs)

	if mode == "langevin":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss(reduction="none")
		elif loss =="hinge":
			criterion = lambda x,y: hinge_loss(x,y,reduction="none",relu=F.elu)
		elif loss == "Xent":
			criterion = nn.CrossEntropyLoss(reduction="none")

		langevin_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,device=device,dtype=dtype,**kwargs)


	if mode == "lee":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss(reduction="none")
		elif loss =="hinge":
			criterion = lambda x,y: hinge_loss(x,y,reduction="none")
		alpha=0.1
		steps=100
		training_lee(model, criterion, X_train,Y_train, optimizer,epochs,batch_size,alpha,steps,device=device,dtype=dtype,**kwargs )

	if mode == "max":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss(reduction="none")
		elif loss =="hinge":
			criterion = lambda x,y: hinge_loss(x,y,reduction="none")
		max_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,device=device,dtype=dtype,**kwargs)

	if mode == "adversarial":
		if loss =="bce":
			criterion = nn.BCEWithLogitsLoss()
		elif loss =="hinge":
			criterion = lambda x,y: hinge_loss(x,y,reduction="mean",relu=F.elu)
		elif loss == "Xent":
			criterion = nn.CrossEntropyLoss(reduction="mean")
		adversarial_training(model,X_train,Y_train,optimizer,criterion,epochs,batch_size,device=device,dtype=dtype,**kwargs)





def run_function(params,mode,epochs,loss,base,model,optimizer,attack_cls,kwargs_attack,device,dtype,X_train=None,Y_train=None,X_test=None,Y_test=None,**kwargs):

	"""
	Function that will be run by the genetic algorithm so as to maximise a criterion
		params={
	"ball_points_number":200,
	"radius":0.4,
	"final_temperature":0.0001,

	}
	"""
	if X_train is None or Y_train is None or X_test is None or Y_test is None:

		# X_train,Y_train,X_test,Y_test = select_base(base,dtype,device)
		X_train,Y_train,X_test,Y_test = load_generic_base(base,dtype,device)
	
	ball_points_number=int(params["ball_points_number"])
	radius=max(params["radius"],0.001)
	temperature_generator = decaying_temperature(1.,params["final_temperature"],1/epochs)

	train_macro(loss,mode,model,X_train,Y_train,optimizer,epochs,ball_points_number=ball_points_number,radius=radius, temperature_generator=temperature_generator,device=device,dtype=dtype,attack_cls=attack_cls,kwargs_attack=kwargs_attack,**kwargs)


def generate_targeted_adversaries(model,x,y,radius,alpha,steps):
	nb_class = int((torch.max(y)+1).item())
	selection = (y).unsqueeze(0).repeat(nb_class,*[1 for _ in y.shape])
	input_duplicated= x.unsqueeze(0).repeat(nb_class,*[1 for _ in x.shape])
	inpt_0 = input_duplicated.clone()
	faker = (1-y).unsqueeze(0).repeat(nb_class,*[1 for _ in y.shape])

	for step in range(steps):

		input_duplicated.requires_grad = True

		output = model(input_duplicated)

		loss = output[faker.long()]-output[selection.long()]
		loss = torch.sum(loss)
		loss.backward()
		with torch.no_grad():
			perturbation = input_duplicated -inpt_0 + alpha*input_duplicated.grad
			input_duplicated = torch.clamp(perturbation,-radius,radius) + inpt_0
	return input_duplicated


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



def plot_pcolor(X_test,Z_test,X_train,Z_train,model,N,title,name,device,dtype):
	"""
	Plotting the 2D landscape using testing, training data and the model. Oonly applicable on 2D models
	"""
	plt.figure(figsize=(20,10))
	xmin,ymin = torch.min(X_train,dim=0)[0].cpu().numpy()
	xmax,ymax = torch.max(X_train,dim=0)[0].cpu().numpy()
	x,y = np.meshgrid(np.linspace(xmin,xmax,N),
					np.linspace(ymin,ymax,N))
	with torch.no_grad():
		X_grid= torch.tensor(np.stack([x,y],axis=-1)).to(device=device,dtype=dtype)
		Z_pt = torch.sigmoid(model(X_grid).squeeze())
		Z_mesh = Z_pt.cpu().numpy()
	
	plt.contourf(x,y,Z_mesh,levels=[0.,0.001,0.01,0.1,0.2,0.3,0.4,0.45,0.50,0.55,0.6,0.7,0.8,0.9,0.99,0.999,1.],alpha=0.7)
	fig = plt.contour(x,y,Z_mesh,levels=[0.,0.001,0.01,0.1,0.2,0.3,0.4,0.45,0.50,0.55,0.6,0.7,0.8,0.9,0.99,0.999,1.],colors='k')
	plt.clabel(fig, fontsize=9, inline=1)
	#plt.scatter(X_train[:,0].cpu().numpy(),X_train[:,1].cpu().numpy(),c=Z_train.squeeze().cpu().numpy(),edgecolors='black',alpha=0.1)
	plt.scatter(X_test[:,0].cpu().numpy(),X_test[:,1].cpu().numpy(),c=Z_test.squeeze().cpu().numpy(),edgecolors='black')
	plt.title(title)
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)


	plt.savefig(name+".png")

def test_approximation_error(N,temperature,benchmark_function,optimum,min_value,max_value):
	device = torch.device("cuda")
	dtype = torch.double
	min_value = torch.tensor(min_value,device=device,dtype=dtype)
	max_value = torch.tensor(max_value,device=device,dtype=dtype)
	samples= (torch.rand(*(N,2),device=device,dtype=dtype))*(max_value-min_value) + min_value
	function_values = benchmark_function(samples)
	soft_sup = temperature*torch.logsumexp(function_values/temperature,dim=0)
	error = torch.abs(soft_sup - optimum)
	return error

# This evolution of the sampling is purely empirical it allows a small sampling at small perturbations
# and a important sampling at bigger perturbations
def n_exponential(epsilon):
	return int(616*(100**(epsilon/0.24))) #0.55



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d","--device",type=str,choices=["cpu","cuda"],help="Select the device on which the program should run")
	parser.add_argument("-i","--id",type=int,help="ID for the process")
	parser.add_argument("-t","--training",type=str,choices=["regular","robust","adversarial","langevin"],help="Type of training to perform")
	parser.add_argument("-s","--save",type=str,help="Folder where to save the model")
	parser.add_argument("-r","--radius",type=float,help="Robustness radius")
	args = parser.parse_args()

	save_file = args.save+"eps_"+ str(args.radius)+"_trial_"+str(args.id)+".pt"
	device = torch.device(args.device)
	dtype = torch.float


	activation = GeneralizedRelu.apply
	model = MLPModel(output_classes=12,non_linearity=activation)
	model.to(device=device,dtype=dtype)

	optimizer = optim.SGD(model.parameters(),lr=0.2)
	# optimizer = optim.Adam(model.parameters(),lr=0.01)
	# optimizer=Ruszczynski(model.parameters(), lr=0.01, mu=0.7)
	scheduler = optim.lr_scheduler.StepLR(optimizer,50000)
	## Arguments for the PGD algorithm for adversarial training
	kwargs_pgd = {
	"loss_fn":nn.CrossEntropyLoss(),
	"eps":args.radius,
	"nb_iter":40,
	"eps_iter":2.5*args.radius/40, # thumb rule
	"clip_min":-100000,
	"clip_max":1000000}

	# Parameters for the training. Not all parameters are used by all trainings methods
	kwargs={
	"mode":args.training,
	"epochs":1500,
	"p":np.inf,
	"device":device,
	"dtype":dtype,
	"model":model,
	"optimizer":optimizer,
	"scheduler":scheduler,
	"base":"avila",
	"loss":"Xent",
	"batch_size": 100,
	"attack_cls":LinfPGDAttackNew, # only for adversarial method
	"kwargs_attack":kwargs_pgd, # only for adversarial method
	"save_file":save_file,
	"gamma":1., # only for langvin method
	"delta":0.5, # only for langvin method
	"eps_iter":4*args.radius/50, # only for adversarial method
	"iterations":400, # only for adversarial method
	"criterion_langevin":nn.CrossEntropyLoss(), # only for langvin method
	}

	# Parameters for the log_sum_exp computation
	args_func={"ball_points_number":n_exponential(args.radius), # empirical rule, beware of ram usage
			"radius":args.radius,
			"final_temperature":0.0001}

	losses = run_function(args_func,**kwargs)

	torch.save(model.state_dict(),save_file)
	

