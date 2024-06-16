import torch
import math
import numpy as np



class LinfPGDAttackNew():
	"""
	updating the advertorch classes that are outdated
	"""
	def __init__(self,predict,loss_fn,eps,nb_iter,eps_iter,rand_init=True,clip_min=0.,clip_max=1.,targeted=False):
		self.predict=predict
		self.loss_fn=loss_fn
		self.clip_max=clip_max
		self.clip_min=clip_min
		self.eps = eps
		self.nb_iter = nb_iter
		self.eps_iter = eps_iter
		self.rand_init = rand_init
		self.targeted = targeted

	def perturb(self,x,y=None):
		delta = torch.zeros_like(x)
		if self.rand_init:
			delta = delta.uniform_(-1,1)*self.eps_iter
		delta.requires_grad = True

		#this attack is multi step and the grad ascent is on the noise that we add
		for i in range(self.nb_iter):
			delta.requires_grad = True			
			outputs = self.predict(x + delta)
			loss = self.loss_fn(outputs, y)
			if self.targeted:
				loss = -loss
			loss.backward()

			grad_sign = delta.grad.detach().sign().clone()
			delta.grad.zero_()
			with torch.no_grad():
				delta = delta + self.eps_iter * grad_sign
				#making sure the the perturbation lies in the perturbation space
				delta = torch.clamp(delta, min=-self.eps, max=self.eps)
				#making sure that the perturbation lies in the input test
				delta = torch.clamp(delta+ x, min=self.clip_min, max=self.clip_max) -x
		return (x+delta).detach()

class LangevinDiffusion(LinfPGDAttackNew):
	"""
	Implementation of Langevin diffusion
	"""
	def perturb(self,x,y):
		delta = torch.zeros_like(x)
		trajectory = []
		if self.rand_init:
			delta = delta.uniform_(-1,1)*self.eps_iter
		delta.requires_grad = True

		#this attack is multi step and the grad ascent is on the noise that we add
		for i in range(self.nb_iter):
			delta.requires_grad = True			
			outputs = self.predict(x + delta)
			loss = self.loss_fn(outputs, y)
			if self.targeted:
				loss = -loss
			loss.backward()

			grad = delta.grad.detach().clone()
			delta.grad.zero_()
			with torch.no_grad():
				delta = delta + self.eps_iter * grad * 1000 + self.eps_iter*torch.randn(grad.shape)
				#making sure the the perturbation lies in the perturbation space
				delta = torch.clamp(delta, min=-self.eps, max=self.eps)
				#making sure that the perturbuation lies in the input test
				delta = torch.clamp(delta+ x, min=self.clip_min, max=self.clip_max) -x
				trajectory.append(delta.clone())
		return (x+torch.stack(trajectory,dim=0)).detach()

class LinfPGDAttackDecaying(LinfPGDAttackNew):
	"""
	Modification of the attack so as to include a monitoring of the loss and a decaying epsilon iter
	"""
	def perturb(self,x,y=None):
		self.monitored_loss = []
		x, y = self._verify_and_process_inputs(x, y)
		delta = torch.zeros_like(x)
		if self.rand_init:
			delta = delta.normal_()*self.eps_iter
		delta.requires_grad = True

		#this attack is multi step and the grad ascent is on the noise that we add
		for i in range(self.nb_iter):
			delta.requires_grad = True			
			outputs = self.predict(x + delta)
			loss = self.loss_fn(outputs, y)
			if self.targeted:
				loss = -loss
			loss.backward()
			self.monitored_loss.append(loss.item())

			grad_sign = delta.grad.detach().sign()
			with torch.no_grad():
				delta = delta + self.eps_iter/math.log(i+2) * grad_sign
				#making sure the the perturbation lies in the perturbation space
				delta = torch.clamp(delta, -self.eps, self.eps)
				#making sure that the perturbuation lies in the input test
				delta = torch.clamp(delta+ x, self.clip_min, self.clip_max) -x

		return (x+delta).detach()


def optimize_linear(grad, eps, norm=np.inf):
	"""
	Solves for the optimal input to a linear function under a norm constraint.
	Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
	:param grad: Tensor, shape (N, d_1, ...). Batch of gradients
	:param eps: float. Scalar specifying size of constraint region
	:param norm: np.inf, 1, or 2. Order of norm constraint.
	:returns: Tensor, shape (N, d_1, ...). Optimal perturbation
	"""

	red_ind = list(range(1, len(grad.size())))
	avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
	if norm == np.inf:
		# Take sign of gradient
		optimal_perturbation = torch.sign(grad)
	elif norm == 1:
		abs_grad = torch.abs(grad)
		sign = torch.sign(grad)
		red_ind = list(range(1, len(grad.size())))
		abs_grad = torch.abs(grad)
		ori_shape = [1] * len(grad.size())
		ori_shape[0] = grad.size(0)

		max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
		max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
		num_ties = max_mask
		for red_scalar in red_ind:
			num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
		optimal_perturbation = sign * max_mask / num_ties
		# TODO integrate below to a test file
		# check that the optimal perturbations have been correctly computed
		opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
		assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
	elif norm == 2:
		square = torch.max(avoid_zero_div, torch.sum(grad ** 2, red_ind, keepdim=True))
		optimal_perturbation = grad / torch.sqrt(square)
		# TODO integrate below to a test file
		# check that the optimal perturbations have been correctly computed
		opt_pert_norm = (
			optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
		)
		one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
			square > avoid_zero_div
		).to(torch.float)
		assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
	else:
		raise NotImplementedError(
			"Only L-inf, L1 and L2 norms are " "currently implemented."
		)

	# Scale perturbation to be the solution for the norm=eps rather than
	# norm=1 problem
	scaled_perturbation = eps * optimal_perturbation
	return scaled_perturbation

def fast_gradient_method(
	model_fn,
	x,
	eps,
	norm,
	clip_min=None,
	clip_max=None,
	y=None,
	targeted=False,
	sanity_checks=False,
	loss_fn=torch.nn.BCEWithLogitsLoss()
):
	"""
	PyTorch implementation of the Fast Gradient Method.
	:param model_fn: a callable that takes an input tensor and returns the model logits.
	:param x: input tensor.
	:param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
	:param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
	:param clip_min: (optional) float. Minimum float value for adversarial example components.
	:param clip_max: (optional) float. Maximum float value for adversarial example components.
	:param y: (optional) Tensor with true labels. If targeted is true, then provide the
			  target label. Otherwise, only provide this parameter if you'd like to use true
			  labels when crafting adversarial samples. Otherwise, model predictions are used
			  as labels to avoid the "label leaking" effect (explained in this paper:
			  https://arxiv.org/abs/1611.01236). Default is None.
	:param targeted: (optional) bool. Is the attack targeted or untargeted?
			  Untargeted, the default, will try to make the label incorrect.
			  Targeted will instead try to move in the direction of being more like y.
	:param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
			  memory or for unit tests that intentionally pass strange input)
	:return: a tensor for the adversarial example
	"""
	if norm not in [np.inf, 1, 2]:
		raise ValueError(
			"Norm order must be either np.inf, 1, or 2, got {} instead.".format(norm)
		)
	if eps < 0:
		raise ValueError(
			"eps must be greater than or equal to 0, got {} instead".format(eps)
		)
	if eps == 0:
		return x
	if clip_min is not None and clip_max is not None:
		if clip_min > clip_max:
			raise ValueError(
				"clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
					clip_min, clip_max
				)
			)

	asserts = []

	# If a data range was specified, check that the input was in that range
	if clip_min is not None:
		assert_ge = torch.all(
			torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
		)
		asserts.append(assert_ge)

	if clip_max is not None:
		assert_le = torch.all(
			torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
		)
		asserts.append(assert_le)

	# x needs to be a leaf variable, of floating point type and have requires_grad being True for
	# its grad to be computed and stored properly in a backward call
	x = x.clone().detach().to(dtype=x.dtype).requires_grad_(True)
	if y is None:
		# Using model predictions as ground truth to avoid label leaking
		_, y = torch.max(model_fn(x), 1)

	# Compute loss
	loss = loss_fn(model_fn(x), y)
	# If attack is targeted, minimize loss of target label rather than maximize loss of correct label
	if targeted:
		loss = -loss

	# Define gradient of loss wrt input
	loss.backward()
	optimal_perturbation = optimize_linear(x.grad, eps, norm)

	# Add perturbation to original example to obtain adversarial example
	adv_x = x + optimal_perturbation

	# If clipping is needed, reset all values outside of [clip_min, clip_max]
	if (clip_min is not None) or (clip_max is not None):
		if clip_min is None or clip_max is None:
			raise ValueError(
				"One of clip_min and clip_max is None but we don't currently support one-sided clipping"
			)
		adv_x = torch.clamp(adv_x, clip_min, clip_max)

	if sanity_checks:
		assert np.all(asserts)
	return adv_x

def clip_eta(eta, norm, eps):
	"""
	PyTorch implementation of the clip_eta in utils_tf.
	:param eta: Tensor
	:param norm: np.inf, 1, or 2
	:param eps: float
	"""
	if norm not in [np.inf, 1, 2]:
		raise ValueError("norm must be np.inf, 1, or 2.")

	avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
	reduc_ind = list(range(1, len(eta.size())))
	if norm == np.inf:
		eta = torch.clamp(eta, -eps, eps)
	else:
		if norm == 1:
			raise NotImplementedError("L1 clip is not implemented.")
			norm = torch.max(
				avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
			)
		elif norm == 2:
			norm = torch.sqrt(
				torch.max(
					avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
				)
			)
		factor = torch.min(
			torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
		)
		eta *= factor
	return eta

def projected_gradient_descent(
	model_fn,
	x,
	eps,
	eps_iter,
	nb_iter,
	norm,
	clip_min=None,
	clip_max=None,
	y=None,
	targeted=False,
	rand_init=True,
	rand_minmax=None,
	sanity_checks=True,
	loss_fn = torch.nn.BCEWithLogitsLoss()
):
	"""
	This class implements either the Basic Iterative Method
	(Kurakin et al. 2016) when rand_init is set to False. or the
	Madry et al. (2017) method if rand_init is set to True.
	Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
	Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
	:param model_fn: a callable that takes an input tensor and returns the model logits.
	:param x: input tensor.
	:param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
	:param eps_iter: step size for each attack iteration
	:param nb_iter: Number of attack iterations.
	:param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
	:param clip_min: (optional) float. Minimum float value for adversarial example components.
	:param clip_max: (optional) float. Maximum float value for adversarial example components.
	:param y: (optional) Tensor with true labels. If targeted is true, then provide the
			  target label. Otherwise, only provide this parameter if you'd like to use true
			  labels when crafting adversarial samples. Otherwise, model predictions are used
			  as labels to avoid the "label leaking" effect (explained in this paper:
			  https://arxiv.org/abs/1611.01236). Default is None.
	:param targeted: (optional) bool. Is the attack targeted or untargeted?
			  Untargeted, the default, will try to make the label incorrect.
			  Targeted will instead try to move in the direction of being more like y.
	:param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
	:param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
			  which the random perturbation on x was drawn. Effective only when rand_init is
			  True. Default equals to eps.
	:param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
			  memory or for unit tests that intentionally pass strange input)
	:return: a tensor for the adversarial example
	"""
	if norm == 1:
		raise NotImplementedError(
			"It's not clear that FGM is a good inner loop"
			" step for PGD when norm=1, because norm=1 FGM "
			" changes only one pixel at a time. We need "
			" to rigorously test a strong norm=1 PGD "
			"before enabling this feature."
		)
	if norm not in [np.inf, 2]:
		raise ValueError("Norm order must be either np.inf or 2.")
	if eps < 0:
		raise ValueError(
			"eps must be greater than or equal to 0, got {} instead".format(eps)
		)
	if eps == 0:
		return x
	if eps_iter < 0:
		raise ValueError(
			"eps_iter must be greater than or equal to 0, got {} instead".format(
				eps_iter
			)
		)
	if eps_iter == 0:
		return x

	assert eps_iter <= eps, (eps_iter, eps)
	if clip_min is not None and clip_max is not None:
		if clip_min > clip_max:
			raise ValueError(
				"clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
					clip_min, clip_max
				)
			)

	asserts = []

	# If a data range was specified, check that the input was in that range
	if clip_min is not None:
		assert_ge = torch.all(
			torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
		)
		asserts.append(assert_ge)

	if clip_max is not None:
		assert_le = torch.all(
			torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
		)
		asserts.append(assert_le)

	# Initialize loop variables
	if rand_init:
		if rand_minmax is None:
			rand_minmax = eps
		eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
	else:
		eta = torch.zeros_like(x)

	# Clip eta
	eta = clip_eta(eta, norm, eps)
	adv_x = x + eta
	if clip_min is not None or clip_max is not None:
		adv_x = torch.clamp(adv_x, clip_min, clip_max)

	if y is None:
		# Using model predictions as ground truth to avoid label leaking
		_, y = torch.max(model_fn(x), 1)

	i = 0
	while i < nb_iter:
		adv_x = fast_gradient_method(
			model_fn,
			adv_x,
			eps_iter,
			norm,
			clip_min=clip_min,
			clip_max=clip_max,
			y=y,
			targeted=targeted,
			loss_fn = loss_fn
		)

		# Clipping perturbation eta to norm norm ball
		eta = adv_x - x
		eta = clip_eta(eta, norm, eps)
		adv_x = x + eta

		# Redo the clipping.
		# FGM already did it, but subtracting and re-adding eta can add some
		# small numerical error.
		if clip_min is not None or clip_max is not None:
			adv_x = torch.clamp(adv_x, clip_min, clip_max)
		i += 1

	asserts.append(eps_iter <= eps)
	if norm == np.inf and clip_min is not None:
		# TODO necessary to cast clip_min and clip_max to x.dtype?
		asserts.append(eps + clip_min <= clip_max)

	if sanity_checks:
		assert np.all(asserts)
	return adv_x