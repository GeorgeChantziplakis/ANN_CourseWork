#Creating the model
cnn_model = Sequential()
############################################################################################################################
# Keras Conv2D is a 2D Convolution Layer, 
# this layer creates a convolution kernel that is wind with layers input which helps produce a tensor of outputs.
cnn_model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (img_size,img_size,1)))

cnn_model.add(MaxPool2D(pool_size=(2,2)))

# Dropout is a technique where randomly selected neurons are ignored during training and is used for preventing overfitting
cnn_model.add(Dropout(0.25))
############################################################################################################################
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
cnn_model.add(Dropout(0.25))
############################################################################################################################
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
cnn_model.add(Dropout(0.3))
############################################################################################################################
cnn_model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
cnn_model.add(Dropout(0.3))

############################################################################################################################
cnn_model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
cnn_model.add(Dropout(0.3))

############################################################################################################################
cnn_model.add(Flatten()) # this converts our image into 1-dimensional for our algorithm to read it (input layer)
cnn_model.add(Dense(1024, activation = "relu")) # here are the neurons we are using for processing , relu is the activation function
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(4, activation = "softmax")) # here is the output layer, having 4 neurons because we have 4 classes, softmax is the activation function
#define optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#compile the model
cnn_model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# The number of epochs is the number of complete passes through the training dataset
# The batch size is a number of samples processed before the model is updated
datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=90,
        zoom_range = 0.1,
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=True,  
        vertical_flip=False)  # we do this for normalising our data 

batch_size=40
datagen.fit(X_train)
history = cnn_model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = 20, validation_data = (X_val,Y_val),
                              steps_per_epoch = X_train.shape[0] // batch_size) 
