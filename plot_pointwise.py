import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from scipy.stats.mstats import mquantiles
import pickle
import matplotlib.ticker as mtick


def clean_data(data):
	data_out={}
	for eps in data:
		values = data[eps]
		data_out[eps] = values[values[:,0]>=0.12]
	return data_out

def clean_lip(data_lip,data_acc):
	data_out={}
	for eps in data_lip:
		values = data_acc[eps]
		data_out[eps] = data_lip[eps][values[:,0]>=0.12]

	return data_out

def merge_data(*args):
	eps = sorted(set(sum([list(arg.keys()) for arg in args],[])))
	output = {}
	for ep in eps:
		output[ep]=None
		for arg in args:
			if ep in arg:
				if output[ep] is None:
					output[ep]= arg[ep]
				else:
					output[ep]= np.concatenate((output[ep],arg[ep]),axis=0)
	return output




def extract_data(data,prob):
	x=[]
	quantiles_acc = dict(zip(prob,[[] for _ in prob]))
	quantiles_adv = dict(zip(prob,[[] for _ in prob]))
	quantiles_rob = dict(zip(prob,[[] for _ in prob]))
	for eps in data:
		x.append(eps)
		values = data[eps]
		Qs = mquantiles(values,prob=prob,axis=0)
		for i,pb in enumerate(prob):
			quantiles_acc[pb].append(Qs[i,0])
			quantiles_adv[pb].append(Qs[i,2])
			quantiles_rob[pb].append(Qs[i,1])
	return x,quantiles_acc,quantiles_adv,quantiles_rob


def extract_single_data(data,prob):
	x=[]
	quantiles= dict(zip(prob,[[] for _ in prob]))
	for eps in data:
		x.append(eps)
		values = data[eps]
		Qs = mquantiles(values,prob=prob)
		for i,pb in enumerate(prob):
			quantiles[pb].append(Qs[i])
	return x,quantiles

def get_percentage_of_failed_computations(data):
	x=[]
	y=[]
	for eps in data:
		x.append(eps)
		values = data[eps][:,0]
		y.append(np.mean((values>=0.65).astype(np.float32)))
	return x,y

			



def plot_filled_quantiles(x,qs,label,color):
	avg=qs[0.5]
	plt.plot(x,avg,label=label,color=color)
	for q in qs:
		if q==0.5:
			continue
		if q < 0.5:
			lower=qs[q]
			upper= avg
		if q > 0.5:
			lower=avg
			upper= qs[q]
		plt.fill_between(x,lower,upper,color=color,alpha=0.2)

def plot_new_format():
	path_regular = ["logs/avila_reg_elu_sgd.pk"]
	path_robust = ["logs/avila_rob_elu_sgd.pk"]
	path_adversarial = ["logs/avila_adv_elu_sgd.pk"]
	path_langevin = ["logs/avila_lang_elu_sgd3.pk"]
	data_regular={}
	data_robust={}
	data_adv={}
	data_lang={}
	prob = [0.1,0.5,0.9]
	for reg_path in path_regular:
		with open(reg_path, 'rb') as handle:
			data_regular = merge_data(pickle.load(handle),data_regular)
	for rob_path in path_robust:
		with open(rob_path, 'rb') as handle:
			data_robust = merge_data(pickle.load(handle),data_robust)
	for adv_path in path_adversarial:
		with open(adv_path,'rb') as handle:
			data_adv = merge_data(pickle.load(handle),data_adv)
	for lang_path in path_langevin:
		with open(lang_path,'rb') as handle:
			data_lang = merge_data(pickle.load(handle),data_lang)

	# x1,y1 = get_percentage_of_failed_computations(data_regular)
	# x2,y2 = get_percentage_of_failed_computations(data_robust)
	# x3,y3 = get_percentage_of_failed_computations(data_adv)
	# plt.figure(figsize=(20,10))
	# plt.plot(x1,y1,label="Vanilla",color='blue')
	# plt.plot(x2,y2,label="Robust",color='red')
	# plt.plot(x3,y3,label="Adversarial",color='green')
	# plt.legend()
	# plt.xlabel(r'$\epsilon$')
	# plt.ylabel("Percentage failed trainings")
	# plt.savefig("AcheivmentPercentMLP.png")
	# exit()
	# data_adv = clean_data(data_adv)

	path_lip_rob = "logs/lipschitz_avila_rob_elu_sgd.pk"
	with open(path_lip_rob,'rb') as handle:
		data_lip_rob = pickle.load(handle)



	path_lip_reg = "logs/lipschitz_avila_reg_elu_sgd.pk"
	with open(path_lip_reg,'rb') as handle:
		data_lip_reg = pickle.load(handle)
	

	path_lip_adv = "logs/lipschitz_avila_adv_elu_sgd.pk"
	with open(path_lip_adv,'rb') as handle:
		data_lip_adv = pickle.load(handle)

	path_lip_lang = "logs/lipschitz_avila_lang_elu_sgd.pk"
	with open(path_lip_lang,'rb') as handle:
		data_lip_lang = pickle.load(handle)
	

	x_regular,quantiles_acc,quantiles_adv,quantiles_rob = extract_data(data_regular,prob)
	x_robust,quantiles_acc_robust,quantiles_adv_robust,quantiles_rob_robust = extract_data(data_robust,prob)
	x_adv,quantiles_acc_adv,quantiles_adv_adv,quantiles_rob_adv = extract_data(data_adv,prob)
	x_lang,quantiles_acc_lang,quantiles_adv_lang,quantiles_rob_lang = extract_data(data_lang,prob)
	
	prob = [0.1,0.25,0.5,0.75,0.9]
	x_reg_lip,quantiles_reg_lip = extract_single_data(data_lip_reg,prob)
	x_rob_lip,quantiles_rob_lip = extract_single_data(data_lip_rob,prob)
	x_adv_lip,quantiles_adv_lip = extract_single_data(data_lip_adv,prob)
	x_lang_lip,quantiles_lang_lip = extract_single_data(data_lip_lang,prob)

	figsize=(8,4)

	fig = plt.figure(figsize=figsize)
	plot_filled_quantiles(x_regular,quantiles_acc,"Vanilla Model",color="blue")
	plot_filled_quantiles(x_robust,quantiles_acc_robust,"Robust Model",color="red")
	plot_filled_quantiles(x_adv,quantiles_acc_adv,"Adversarial Model",color="green")
	plot_filled_quantiles(x_lang,quantiles_acc_lang,"Langevin Model",color="purple")
	plt.legend()
	plt.xlabel(r'$\epsilon$')
	# plt.xlabel("Epochs")
	plt.ylabel("Accuracy Test Set")
	plt.savefig("AccuracyQuantilesAvilaLang.png")

	plt.figure(figsize=figsize)

	plot_filled_quantiles(x_regular,quantiles_adv,"Vanilla Model",color="blue")
	plot_filled_quantiles(x_robust,quantiles_adv_robust,"Robust Model",color="red")
	plot_filled_quantiles(x_adv,quantiles_adv_adv,"Adversarial Model",color="green")
	plot_filled_quantiles(x_lang,quantiles_adv_lang,"Langevin Model",color="purple")
	plt.legend()
	plt.xlabel(r'$\epsilon$')
	# plt.xlabel("Epochs")
	plt.ylabel("Accuracy Adversarial Set")
	plt.savefig("AccuracyAdvQuantilesAvilaLang.png")

	plt.figure(figsize=figsize)

	plot_filled_quantiles(x_regular,quantiles_rob,"Vanilla Model",color="blue")
	plot_filled_quantiles(x_robust,quantiles_rob_robust,"Robust Model",color="red")
	plot_filled_quantiles(x_adv,quantiles_rob_adv,"Adversarial Model",color="green")
	plot_filled_quantiles(x_lang,quantiles_rob_lang,"Langevin Model",color="purple")
	plt.legend()
	plt.xlabel(r'$\epsilon$')
	# plt.xlabel("Epochs")
	plt.ylabel("Accuracy on worst case set")
	plt.savefig("AccuracyRobQuantilesAvilaLang.png")

	plt.figure(figsize=figsize)

	plot_filled_quantiles(x_reg_lip,quantiles_reg_lip,"Vanilla Model",color="blue")
	plot_filled_quantiles(x_rob_lip,quantiles_rob_lip,"Robust Model",color="red")
	plot_filled_quantiles(x_adv_lip,quantiles_adv_lip,"Adversarial Model",color="green")
	plot_filled_quantiles(x_lang_lip,quantiles_lang_lip,"Langevin Model",color="purple")
	plt.legend()
	plt.yscale('log')
	plt.ylim(100,10000)
	fig.get_axes()[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
	plt.xlabel(r'$\epsilon$')
	# plt.xlabel("Epochs")
	plt.ylabel("Lipschitz constant")
	plt.savefig("LipsQuantilesAvilaLang.png")


def plot_test_langevin():
	gammas = [1.0] #[0.5, 1.0, 5.0, 10.0 ]
	deltas = [2.0]#[0.5, 1.0, 5.0, 10.0 ]
	epsiters = [250.0,500.0,1000.0,10000.0] #[1.0, 10.0, 50.0 ]
	save = "modelsTopaze/models/logging/"
	data = []
	for gamma in gammas:
		for delta in deltas:
			for epsiter in epsiters:
				save_file = (save+"iter_{0}_gamma_{1}_delta_{2}_epsiter_{3}.pt").format(str(100),str(gamma),str(delta),str(epsiter))
				with open(save_file,'rb') as handle:
					value = pickle.load(handle)
					# if epsiter==50.:
					print(value)
					data.append(value)
	print(max(data,key=lambda x: x["losses"][2][0]))



if __name__ == "__main__":
	plot_new_format()
	#plot_matrix_results()
	# plot_test_langevin()



	


	