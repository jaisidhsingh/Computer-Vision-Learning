import torch
import numpy as np
import matplotlib.pyplot as plt


# change final output with channels = number of classes to a 2D array with pixels corresponding to classes
def class_wise_pixels(image):
	out = torch.argmax(image.squeeze(), dim=0).detach().cpu().numpy()
	return out

# to view segmented image, changing 2D array to rgb correspondinf to class wise colour coding
def segmented2RGB(reference, image_shape, nc=21):
	label_colors = np.array([(0, 0, 0), # 0=background
	           # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
	           (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
	           # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	           (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
	           # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
	           (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
	           # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
	           (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

	r = np.zeros(image_shape).astype(np.uint8)
	g = np.zeros(image_shape).astype(np.uint8)
	b = np.zeros(image_shape).astype(np.uint8)

	for i in range(0, nc):
		o = np.where(reference==i)
		o = zip(o[0], o[1])
		o = np.array(list(o))

		for item in o:
			r[item[0]][item[1]] = label_colors[i, 0]
			g[item[0]][item[1]] = label_colors[i, 1]
			b[item[0]][item[1]] = label_colors[i, 2]

	rgb = np.stack([r, g, b], axis=2)
	return rgb

# visualize output
image = torch.rand((1, 21, 224, 224))
out = class_wise_pixels(image)
rgb = segmented2RGB(out, (224, 224))

print(rgb)
print(rgb.shape)
print(np.unique(rgb))


plt.figure()
plt.imshow(rgb)
plt.axis("off")
plt.show()
