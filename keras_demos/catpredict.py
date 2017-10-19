import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

model = inception_v3.InceptionV3()

img = image.load_img("./cat/cat.png", target_size=(299,299))
input_image = image.img_to_array(img)

input_image /=255.0
input_image -= 0.5
input_image *=2.0

input_image = np.expand_dims(input_image, axis=0)

prediction = model.predict(input_image)

predicted_classes = inception_v3.decode_predictions(prediction, top=5)
# print(predicted_classes)

imagenetid, name, confidence = predicted_classes[0][0]
print("this is a {} with {:.4}% confidence!".format(name, confidence))


