'''
Define helper functions here.
'''
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.video.resnet import VideoResNet, BasicStem, Conv3DSimple
from torchvision.models.video.resnet import BasicBlock as BasicBlock3D
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

'''
Define custom models inheritting existing PyTorch pretrained model classes to reimplement forward functions
'''
class ResNet_Extended2D(ResNet):

	def __init__(self, pretrained):
		super(ResNet_Extended2D, self).__init__(BasicBlock, [2, 2, 2, 2])

		####### Find some way to build a 3D version ResNet and copy stacked weights of 2D ResNet to it.
		# self.load_state_dict(pretrained.state_dict())


	def forward(self, x):

		out_tmp = self.conv1(x)
		out_tmp = self.bn1(out_tmp)
		out_tmp = self.relu(out_tmp)
		out_tmp = self.maxpool(out_tmp)
		out_tmp = self.layer1(out_tmp)
		out_tmp = self.layer2(out_tmp)
		out_tmp = self.layer3(out_tmp)
		out = self.layer4(out_tmp)

		return out



class Custom_ResNet3D(VideoResNet):

	def __init__(self, pretrained):
		super(Custom_ResNet3D, self).__init__(BasicBlock3D, [Conv3DSimple, Conv3DSimple, Conv3DSimple, Conv3DSimple], [2, 2, 2, 2], BasicStem)

	def forward(self, x):

		# print ('input shape: ', x.shape)
		out_tmp = self.stem(x)
		# print ('intermediate shape: ', out_tmp.shape)
		out_tmp = self.layer1(out_tmp)
		# print ('intermediate shape: ', out_tmp.shape)
		out_tmp = self.layer2(out_tmp)
		# print ('intermediate shape: ', out_tmp.shape)
		out_tmp = self.layer3(out_tmp)
		# print ('intermediate shape: ', out_tmp.shape)
		out = self.layer4(out_tmp)
		# print ('output shape: ', out.shape)

		return out



'''
Define helper functions
'''
def eval_model(loader, prepro_mod, encoder, evals, eval_samps):

	# Collect the embeddings and labels
	embeddings = list()
	name2labels = {eval_dict['name']: list() for eval_dict in evals}

	samp_cnt = 0
	for batch_dict in loader:
		samp = batch_dict['samp']
		emb_tmp = prepro_mod(samp)
		emb = torch.flatten(encoder(emb_tmp), start_dim = 1)
		embeddings.append(emb)

		for evaluation in evals:
			name = evaluation['name']
			eval_type = evaluation['type']
			label = batch_dict[name]
			
			if eval_type in ['curr_value', 'pred_value']:
				name2labels[name].append(label)
			else:
				raise Exception ('Inavlid eval_type found.')

		samp_cnt += len(emb)
		if samp_cnt >= eval_samps:
			break
	
	embeddings = torch.cat(embeddings, dim = 0).detach().cpu().numpy()
	# print ('shape of embeddings: ', embeddings.shape)

	# train evlauating models and calculate metrics
	print ('\n\tEvaluation')

	for evaluation in evals:
		name = evaluation['name']
		eval_type = evaluation['type']
		model = evaluation['model']
		metric = evaluation['metric']

		if eval_type in ['curr_value', 'pred_value']:
			label = np.squeeze(torch.cat(name2labels[name], dim = 0).numpy(), axis = 1)

			scaler = StandardScaler()
			scaler.fit(embeddings)
			embeddings = scaler.transform(embeddings)

			svr = LinearSVR()
			print ('\t\ttraining SVR...')
			svr.fit(embeddings, label)
			pred = svr.predict(embeddings)

		else:
			raise Exception ('Inavlid eval_type found.')

		####### Still have to implement other metrics such as accuracy for binary classification (data imbalance issue)
		if metric == 'mse':
			mse = np.mean((pred - label) ** 2)
			print ('\t\t%s, mse: ' % name, mse)
		else:
			raise Exception ('Invalid metric found.')



def save_model(name, prepro_mod, encoder):

	prepro_mod_save_name = name + '_prepro_mod'
	encoder_save_name = name + '_encoder'

	torch.save(prepro_mod.state_dict(), prepro_mod_save_name)
	torch.save(encoder.state_dict(), encoder_save_name)
	print ('\t\tmodels saved as %s and %s.' % (prepro_mod_save_name, encoder_save_name))



def get_pretrained(name):

	if name == 'ResNet':
		raise Exception ('Extention from 2D ResNet to 3D is not implemented yet.')

		pretrained = models.resnet18(pretrained = True)
		model = ResNet_Extended2D(pretrained)

	if name == 'ResNet3D':
		pretrained = models.video.r3d_18(pretrained = True)
		model = Custom_ResNet3D(pretrained)

	else:
		raise Exception ('Invalid pretrained model name found.')

	return model


####### Not yet implemented
def check_params(branch2params, objs):
	'''
	The function checks if the index range of any learning objective is out of the range of model output shape, or if an output value is used by multiple learning objectives.
	'''
	print ('Parameter correctness checking is not implemented yet.')



def print_params():
	'''
	The function takes in an argument object and print out the parameters.
	'''
	####### Print parameters
	pass


def print_obj2sum_losses(obj2sum_losses, sum_ttl_loss, samp_cnt):

	samp_cnt = float(samp_cnt)

	print ('\n\ttraining losses')
	
	for obj_name, loss_dict in obj2sum_losses.items():
		loss = loss_dict['sum_loss'] / samp_cnt
		weighted_loss = loss_dict['sum_weighted_loss'] / samp_cnt

		print ('\t\tObjective %s:' % obj_name)
		print ('\t\t\traw loss: %f' % loss)
		print ('\t\t\tweighted loss: %f' % weighted_loss)

	print ('\n\t\ttotal loss: %f' % (sum_ttl_loss / samp_cnt))






