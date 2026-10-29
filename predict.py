import torch
import torch.nn as nn
from torchvision import models, transforms
import json
from PIL import Image
import argparse

def load_model(weights_path, classes_path):

	with open(classes_path) as f:
		class_names = json.load(f)

	model = models.resnet18(weights=None)
	model.fc = nn.Linear(model.fc.in_features, len(class_names))
	model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
	model.eval()

	return model, class_names

#Mean: [0.485, 0.456, 0.406] (Red, Green, Blue)
#Standard Deviation (Std): [0.229, 0.224, 0.225] (Red, Green, Blue)

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
])


def preprocess(img):
	img = img.convert("RGB") # .convert will convert and return (won't modify the original image)
	transformed_tensor = transform(img)
	return transformed_tensor.unsqueeze(0)




def predict(model, class_names, x):
	with torch.inference_mode():
		logits = model(x)
		probs = torch.softmax(logits, dim=1)
		conf, idx = probs.max(dim=1) #.max will return highest value in each row and its idx
		return class_names[idx.item()], conf.item()




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Classify a EuroSAT image")
	parser.add_argument("image", help="path to the image file")
	parser.add_argument("--weights", default='models/best.pt', help='path to the checkpoint')
	parser.add_argument('--class_names', default="models/classes.json", help="path to the classes")

	args = parser.parse_args()

	model, class_names = load_model(args.weights, args.class_names)
	image = Image.open(args.image)
	preprocessed_tensor = preprocess(image)
	class_name, conf = predict(model, class_names, preprocessed_tensor)
	print(f"{class_name} ({conf * 100:.2f}%)")



























	# model, class_names = load_model('models/best.pt', 'models/classes.json')
	# image = Image.open("samples/Forest_9.jpg")
	# t = preprocess(image)
	# print(t.shape, type(t), t.ndim)
	# fake = Image.new("RGBA", (64, 64))
	# print(preprocess(fake).shape)
	# cls, conf = predict(model, class_names, t)
	# print(f"Class: {cls} and confidence: {conf:.2f}%")







