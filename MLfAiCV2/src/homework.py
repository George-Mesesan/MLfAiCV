# this script should be executed in the caffe directory
# the script is in large part based on the example from http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/classification.ipynb

import numpy as np
import math

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, '../python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load('../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

n = 1000
max_entropy = -math.log(1.0/n, 2)
print max_entropy

for i in range(1,9):
        image_path = '../examples/IMG_' + str(i) + '_square.jpg'
        print 'image path:', image_path
        input_image = caffe.io.load_image(image_path)

        prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
        print 'len:', len(prediction)
        print 'predicted class:', prediction.argmax()
        print 'predicted probability:', prediction.max()
        print prediction.sum()

        entropy = 0
        for j in range(0,len(prediction)):
                entropy = entropy + prediction[0][j] * math.log(prediction[0][j], 2)
        print 'entropy:', -entropy
        print 'normalized entropy:', (-entropy) / max_entropy