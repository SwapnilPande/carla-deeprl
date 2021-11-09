'''
Define the dataset class whose objects are taken by data loaders as argument.
'''
import torch
import json
import os
import cv2

import torch.nn.functional as F
from torch.utils.data import Dataset



class BEV_Dataset(Dataset):

	def __init__(self, data_dir, pix_class_num, stack_size, objs):
		'''
		Arguments
			objs: A list of learning objectives.
				It determines what information is provided by the dataset as samples and labels.
		'''
		
		self.pix_class_num = pix_class_num
		self.stack_size = stack_size
		self.objs = objs

		self.episodes = self.load_data(data_dir)
		self.episode_cnt = len(self.episodes)
		self.episode_lens = [len(episode) for episode in self.episodes]
		print ('episode_cnt: ', self.episode_cnt)
		print ('episode_lens: ', self.episode_lens)

		self.idx2frame_idxs, self.len = self.build_idx_map()


	def __len__(self):
		return self.len


	def __getitem__(self, idx):

		samp_dict = dict()

		episode_idx, frame_idx = self.idx2frame_idxs[idx]
		last_frame_idx = frame_idx + self.stack_size - 1
		last_frame = self.episodes[episode_idx][last_frame_idx]

		imgs = [self.episodes[episode_idx][idx]['img'] for idx in range(frame_idx, frame_idx + self.stack_size)]
		stack = torch.stack(imgs, dim = 0)
		samp_dict['samp'] = torch.moveaxis(F.one_hot(stack, num_classes = self.pix_class_num), 3, 0).float()

		# Put each label into samp_dict
		for obj_dict in self.objs:
			obj_name = obj_dict['name']
			obj_type = obj_dict['type']

			if obj_type == 'reconstruction':
				# No label to return as the input itself serves as the label.
				pass

			elif obj_type == 'pred_stack':
				pred_stack_size = obj_dict['pred_stack_size']
				pred_time_forward = obj_dict['pred_time_forward']

				pred_idx = last_frame_idx + pred_time_forward

				pred_imgs = [self.episodes[episode_idx][idx]['img'] for idx in range(pred_idx, pred_idx + pred_stack_size)]
				pred_stack = torch.stack(pred_imgs, dim = 0)
				samp_dict[obj_name] = torch.moveaxis(F.one_hot(pred_stack, num_classes = self.pix_class_num), 3, 0).float()

			elif obj_type == 'curr_value':
				value_key = obj_dict['value_key']
				value = last_frame[value_key]
				samp_dict[obj_name] = torch.tensor([value,])

			elif obj_type == 'pred_value':
				value_key = obj_dict['value_key']
				pred_time_forward = obj_dict['pred_time_forward']
				value = self.episodes[episode_idx][last_frame_idx + pred_time_forward][value_key]
				samp_dict[obj_name] = torch.tensor([value,])

			else:
				raise Exception ('Invalid objective type found.')

		return samp_dict


	def load_data(self, data_dir):
		
		episodes = list()

		for samp_folder in os.listdir(data_dir):
			if samp_folder.startswith('.'):
				continue

			samp_dir = os.path.join(data_dir, samp_folder)
			img_dir = os.path.join(samp_dir, 'topdown')
			attr_dir = os.path.join(samp_dir, 'measurements')

			frame_idx2attrs = dict()
			for img_name in os.listdir(img_dir):
				if img_name.startswith('.'):
					continue

				frame_idx = int(img_name.split('_')[1][:4])

				# Load the attribute file
				str_idx = str(frame_idx)
				attrs_path = os.path.join(attr_dir, '0' * (4 - len(str_idx)) + str_idx + '.json')

				with open(attrs_path) as f:
					attrs = json.load(f)

				img_path = os.path.join(img_dir, img_name)
				# take only the first dimension as all three of them are identical
				# Minus one to make the pixel class indices 0-based.
				img = torch.tensor(cv2.imread(img_path))[:, :, 0] - 1
				attrs['img'] = img.long()
				
				frame_idx2attrs[frame_idx] = attrs

			episodes.append(frame_idx2attrs)

		return episodes


	def build_idx_map(self):
		'''
		Build a mapping from a sample index to its corresponding episode and frame. objs is used to determine how many future frames have to be reserved for a sample to provide labels from future frames.
		'''
		fut_step_cnt = 0
		for obj_dict in self.objs:
			obj_type = obj_dict['type']
			
			if obj_type == 'pred_stack':
				pred_stack_size = obj_dict['pred_stack_size']
				pred_time_forward = obj_dict['pred_time_forward']
				obj_fut_step_cnt = pred_time_forward + pred_stack_size - 1
				fut_step_cnt = max(fut_step_cnt, obj_fut_step_cnt)
			
			elif obj_type == 'pred_value':
				pred_time_forward = obj_dict['pred_time_forward']
				fut_step_cnt = max(fut_step_cnt, pred_time_forward)


		idx2frame_idxs = dict()
		idx = 0
		for episode_idx in range(len(self.episodes)):
			episode = self.episodes[episode_idx]
			# Leave enough time steps for stack size and prediction of future frames.
			for frame_idx in range(len(episode) - fut_step_cnt - self.stack_size + 1):
				if frame_idx not in episode:
					raise Exception ('Missing frame index found in an episode.')
				idx2frame_idxs[idx] = (episode_idx, frame_idx)
				idx += 1

		return idx2frame_idxs, idx





