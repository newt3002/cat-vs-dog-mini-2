from keras.models import model_from_json
import cv2
import numpy as np

# Loading the Model from Json File
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Loading the weights
loaded_model.load_weights("./model.h5")
print("Loaded model from disk")

# Compiling the model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# loading the test image
image = cv2.imread('./test1/1.jpg')
image = cv2.resize(image, (50,50))
image = image.reshape(1, 50, 50, 3)

cv2.imshow("Input Image", image)
if cv2.waitKey(0) & 0xFF == ord('x'):
    pass
cv2.destroyAllWindows()

# Predicting to which class the input image has been classified
result = loaded_model.predict_classes(image)
if(result[0][0] == 1):
    print("It is a Dog")
else:
    print("It is a Cat")