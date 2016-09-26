import numpy as np

from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import (VGG19,
                                      preprocess_input,
                                      decode_predictions)



def preprocess_image(img_path, nrows=224, ncols=224):
    '''
    Utility function to open, resize and format pictures into
    appropriate tensors.
    '''
    img = load_img(img_path, target_size=(nrows, ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


img_path = 'doggy.jpg'
x = preprocess_image(img_path)

model = VGG19(weights='imagenet')#, include_top=False)
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
