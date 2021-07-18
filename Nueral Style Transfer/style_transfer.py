"""
In this Code I want to implement the Neural Style Transfer from this Paper https://arxiv.org/pdf/1508.06576.pdf

"""

import cv2
import pdb
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

def load_the_image(path):
	#Load the tensor, read it and apply transformations
	img = cv2.imread(path)
	img = cv2.resize(img,(256,256))
	trans = transforms.ToTensor()
	image_tensor = trans(img)
	return image_tensor

content_image = load_the_image('content_image.jpg')
style_image = load_the_image("style_image.jpg")
generated_image=content_image.clone().requires_grad_(True)


class VGG(nn.Module):
	"""
	Create a VGG Class to create a model object to train the model to Extract  feature maps out of the Images
	"""

	def __init__(self):

		super(VGG,self).__init__()

		#The paper points out to obtain feature maps from the following extracts conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1
		self.req_features_map_layers= ['0','5','10','19','28'] 

		#We will be needing only the CNN part of the VGG Network, so we will be taking only those layers
		self.model=models.vgg19(pretrained=True).features[:29]

	def forward(self,x):

		#initialize an array that wil hold the activations from the chosen layers
		features_maps=[]

		x = torch.reshape(x,(1,3,256,256))

		#Iterate over all the layers of the mode
		for layer_num,layer in enumerate(self.model):

			x=layer(x)


			if (str(layer_num) in self.req_features_map_layers):
				#Appending the features maps from the layers of our intrest
				features_maps.append(x)

		return features_maps



def loss(alpha,beta,content_features_maps,style_feature_maps, generated_feature_maps):
	"""
	Calculating the Loss = alpha*content loss + beta* style_loss
	alpha and beta are hyper-parameters
	"""
	style_loss = 0
	content_loss = 0

	for content_features ,style_features ,gen_features in zip(content_features_maps,style_feature_maps, generated_feature_maps):
		
		#Calculating the Content Loss
		content_loss = content_loss + torch.mean((gen_features - content_features)**2)

		#Calculating the Style Loss
		batch_size,channel,height,width=gen_features.shape

		#Graham Matrix Calculaiton

		gen_Graham_matrix = torch.mm(gen_features.view(channel,height*width),gen_features.view(channel,height*width).t())
		style_Graham_matrix = torch.mm(style_features.view(channel,height*width),style_features.view(channel,height*width).t())

		style_loss = style_loss + (1/(4*(channel**2)*((height*width)**2)))*torch.mean((gen_Graham_matrix - style_Graham_matrix)**2)

	total_loss = alpha*content_loss + beta*style_loss
	return total_loss


device=torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')

model=VGG().to(device).eval() 

#Model Params
num_epochs = 100
lr = 0.007
alpha = 8
beta = 70

#Update the Generated Image Pixels
optimizer=optim.Adam([generated_image],lr=lr)

for i in range (1,num_epochs):

	content_features = model(content_image)
	style_features = model(style_image)
	gen_features = model(generated_image)

	total_loss = loss(alpha, beta, gen_features, content_features, style_features)

	optimizer.zero_grad()

	total_loss.backward()

	optimizer.step()

	print(i)

	if i % 10 ==0 :

		print (total_loss)

		save_image(generated_image, '.epoch_{}.png'.format(i))


