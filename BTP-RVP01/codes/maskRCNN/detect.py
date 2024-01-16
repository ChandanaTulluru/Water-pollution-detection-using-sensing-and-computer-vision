# evaluate the mask rcnn model on the kangaroo dataset
from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import defusedxml.ElementTree

# class that defines and loads the kangaroo dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "algae")
		self.add_class("dataset",2,"Trash_on_water")
		# define data locations
		images_dir = dataset_dir + '/algae/'
		annotations_dir = dataset_dir + '/annotation/'
		# find all images
		count=0
		dict1={}
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[19:32]
			# if(image_id in discard_id):
			# 	continue
			# dict1[image_id]=count/
			count+=1
			# skip bad images
			#if image_id in ['00090']:
			#	continue
			# skip all images after 150 if we are building the train set
			if is_train and count >= 1001:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and count < 1001:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + filename[:-4]+ '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
			
 
	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = defusedxml.ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height
 
	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('algae'))
			class_ids.append(self.class_names.index('Trash_on_water'))
			# print(self.class_names.index('Trash_on_water'))
			# print(self.class_names.index('algae'))
		return masks, asarray(class_ids, dtype='int32')
 
	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	print("hey reached prediction config")
	NAME = "algae_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 2
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		plt.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		plt.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		plt.imshow(image)
		plt.title('Predicted')
		ax = plt.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = pt.Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	plt.show()

# load the train dataset
train_set = KangarooDataset()
train_set.load_dataset('./Dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = KangarooDataset()
test_set.load_dataset('./Dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
cfg = PredictionConfig()

model = MaskRCNN(mode='inference', model_dir='./mode/algae_cfg/', config=cfg)
# load model weights
model_path = './mode/algae_cfg/mask_rcnn_algae_cfg_0050.h5'
#model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
image = plt.imread('img1.jpg')

plt.imshow(image)
# convert pixel values (e.g. center)
scaled_image = mold_image(image, cfg)
# convert image into one sample
sample = expand_dims(scaled_image, 0)
# make prediction
yhat = model.detect(sample, verbose=0)[0]
ax = plt.gca()
# plot each box
for box in yhat['rois']:
	# get coordinates
	y1, x1, y2, x2 = box
	# calculate width and height of the box
	width, height = x2 - x1, y2 - y1
	# create the shape
	rect = pt.Rectangle((x1, y1), width, height, fill=False, color='red')
	# draw the box
	ax.add_patch(rect)
#plt.imshow(yhat)

# create config

# define the model
model = MaskRCNN(mode='inference', model_dir='./mode/algae_cfg/', config=cfg)
# load model weights
model_path = './mode/algae_cfg/mask_rcnn_algae_cfg_0050.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)
