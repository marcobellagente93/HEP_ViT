from time import time

import torch
import torch.nn
import torch.optim
import numpy as np
import sys

import config as c

from model import *
from load_data import *
import data

model = VisualTransformer(dim=c.in_dim, image_size=c.img_size, patch_size=c.patch_size, num_classes=c.num_classes, depth=c.depth, heads=c.heads, mlp_dim=c.mlp_dim, channels=c.channels)

model.set_optimizer()
model = model.float()

print(model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in model.params_trainable]))

if c.dataset == 'top_tagging':
	train_loader, train_size, data_shape = Loader('train', c.batch_size, False)
	val_loader, val_size, data_shape     = Loader('val', c.batch_size, False)

N_epochs = c.n_epochs
t_start = time()
loss_mean = []

criterion = nn.CrossEntropyLoss()

print("\n" + "==="*30 + "\n")
print('Epoch\tBatch/Total \tTime \tTraining Loss \tValid Loss \tAccuracy \t\tLR')

for epoch in range(N_epochs):
	#for i, x in enumerate(train_loader):
	for i, (x, l) in enumerate(data.train_loader):

		x = x.float()

		if c.dataset == 'top_tagging':
			img = x[:,:1600].reshape(c.batch_size, c.channels, c.img_size, c.img_size)
			l = x[:,-2].type(torch.LongTensor)

		else:
			img = x

		output = model(img).float()

		loss = criterion(output, l)
	
		model.optimizer.zero_grad()
		loss.backward()
		model.optimizer.step()

		loss_mean.append(loss.item())

		if not i % c.show_interval:
			with torch.no_grad():
				acc = []
				loss_val = []

				for k, (x, l) in enumerate(data.test_loader):
				#for k, x in enumerate(val_loader):

					x = x.float()
					if c.dataset == 'top_tagging':
						img = x[:,:1600].reshape(c.batch_size, c.channels, c.img_size, c.img_size)
						l = x[:,-2].type(torch.LongTensor)
					else:
						img = x

					output_val = model(img).float()
					loss_val.append(criterion(output_val, l))

					pred, idx = torch.max(output_val, 1)

					acc.append((torch.sum(l == idx) / len(idx)).item())

			print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.4f\t\t%.2e' % (epoch,
															i, data.train_size/c.batch_size,
															(time() - t_start)/60.,
															np.mean(loss_mean),
															np.mean(loss_val),
															np.mean(acc),
															model.optimizer.param_groups[0]['lr'],
															), flush=True)
			loss_mean = []

	model.scheduler.step()

#Save last model
#torch.save(cinn.state_dict(), 'output/model.pt')
