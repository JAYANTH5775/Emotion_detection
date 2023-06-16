from keras.preprocessing.image import ImageDataGenerator
from keras.models  import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras import backend as K
import os



# img = load_img('path/to/image.jpg')



# train_data = 'data/train'
# test_data = 'data/test'

# Replace the above line with:
train_data = os.path.join(os.getcwd(), 'data', 'train')
test_data = os.path.join(os.getcwd(), 'data', 'test')

train_datagen  =  ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')



validation_datagen = ImageDataGenerator(
    rescale=1./255
)


train_generator = train_datagen.flow_from_directory(
    train_data, 
    color_mode='grayscale', 
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator  = validation_datagen.flow_from_directory(
    test_data, 
    color_mode='grayscale', 
    target_size=(48,48), 
    batch_size=32, 
    class_mode='categorical',
    shuffle=True)



class_labels = ['Angry', 'Disgust', 'Fear','Happy', 'Neutral','Sad','Suprise']

# img , label = train_generator.__next__()

img , label  = next(train_generator)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation ='relu', input_shape = (48,48,1)))
model.add(Conv2D(64, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(256, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(7,activation='softmax'))   #  since it is a categorical   we are using softmazx, we use the sigmoifd for the binary   classification 

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())



train_path = 'data/train'

test_path = 'data/test'

num_train_imgs = 0


for root , dirs , files in os.walk(train_path):
    num_train_imgs +=len(files)

num_test_imgs = 0 
for root ,dirs, files in os.walk(test_path): 
    num_test_imgs +=len(files)   #counting the number of the images 


print(num_test_imgs)    #Number of the test images
print(num_train_imgs)    #Number od the train images 

epochs = 30 
history = model.fit(
                    train_generator, 
                    steps_per_epoch=num_train_imgs//32,
                    epochs = epochs, 
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs//32)




model.save('model_file.h5')
# Plot the loss and accuracy curves for training and validation


