# Importing the required packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
model = Sequential()

# Convolution
model.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))

# Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
model.add(Flatten())


model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Loading the training Set
training_set = train_datagen.flow_from_directory('./train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
# Training the classifier
model.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 25,
                         validation_steps = 2000)

# Converting the Model to json
model_json = model.to_json()
with open("./model.json","w") as json_file:
    json_file.write(model_json)

# Saving the weights in a seperate file
model.save_weights("./model.h5")

print("Classifier trained Successfully!")