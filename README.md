
# Dog Breed Classifier

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.


    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](dog_app/output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def face_plot(img):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
    return image
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 
The percentage of human faces detected in the first 100 human images is 100% while the percentage of human faces detected in first 100 dog images is 12%. The dog images where the `face_detector` function misdetected a human faces are plotted. In some of the images, humans are also present. In some of the images false positives are identified.


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
```


```python
# testing the performance of face_detector
detected = {}
detected["human"] = []
detected["dog"] = []

for human_file in human_files_short:
    detected["human"].append(face_detector(human_file))

for dog_file in dog_files_short:
    detected["dog"].append(face_detector(dog_file))
```


```python
percentage = {"human":sum(detected["human"]), "dog":sum(detected["dog"])}
print("Percentage of human faces detected in first 100 human images: {}".format(percentage["human"]))
print("Percentage of human faces detected in first 100 dog images: {}".format(percentage["dog"]))
```

    Percentage of human faces detected in first 100 human images: 100
    Percentage of human faces detected in first 100 dog images: 11



```python
def show_images(images, cmap=None):
    cols = 2
    rows = (len(images)+1)//cols
    
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
```


```python
images = []
for idx, dog_file in enumerate(dog_files_short):
    if detected["dog"][idx]:
        image = cv2.cvtColor(cv2.imread(dog_file), cv2.COLOR_BGR2RGB)
        images.append(face_plot(image))
```

These are the pictures the `face_detector` function miscategorized dog images as humans.


```python
show_images(images)
```


![png](dog_app/output_15_0.png)


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__ It is a reasonable expectation if we need to detect only the human faces using OpenCV. However, the face detector from OpenCV can identify false positives like it has done with the dog images before. It is also to be noted that the face detector does not detect human limbs or presence of any humans.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
human_detection = []
for human_file in human_files[:200]:
    human_detection.append(face_detector(human_file))

print("Percentage of human faces detected in all human images (OpenCV): ", sum(human_detection)/len(human_detection)*100)
```

    Percentage of human faces detected in all human images (OpenCV):  99.5


---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ The percentage of dog faces detected in human images is 0 whereas the same for dog images is 100. This image detection algorithm is better than that of OpenCV's.


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.
dog_detected = {}
dog_detected["human_files"] = [dog_detector(human_file) for human_file in human_files_short]
dog_detected["dog_files"] = [dog_detector(dog_file) for dog_file in dog_files_short]

percentage = {}
percentage["human"] = sum(dog_detected["human_files"])/len(dog_detected["human_files"])*100
percentage["dog"] = sum(dog_detected["dog_files"])/len(dog_detected["dog_files"])*100
```


```python
print("Percentage of dog faces detected in first 100 human images: {}".format(percentage["human"]))
print("Percentage of dog faces detected in first 100 dog images: {}".format(percentage["dog"]))
```

    Percentage of dog faces detected in first 100 human images: 0.0
    Percentage of dog faces detected in first 100 dog images: 100.0


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:39<00:00, 169.34it/s]
    100%|██████████| 835/835 [00:04<00:00, 190.02it/s]
    100%|██████████| 836/836 [00:04<00:00, 191.13it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ I chose to modify the suggested architecture with two additional layers of nodes. A detailed description is given below.
* First layer: 8 nodes to identify the lines in the images
* Second layer: maxpooling layer
* Third layer: 16 nodes to identify shapes from the images
* Fourth layer: maxpooling layer
* Fifth layer: 32 nodes to identify structures in the images
* Sixth layer: maxpooling layer
* Seventh layer: 64 nodes to identify minor details like patterns in the images
* Eight layer: maxpooling layer
* Nineth layer: 128 nodes to identify major details of breeds
* Tenth layer: global average pooling
* Eleventh layer: dense layer with 133 nodes for each dog breed.

From many trials, I found this architecture working better for the classifier.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

architecture = [[8, "relu"], [16, "relu"], [32, "relu"], [64, "relu"], [128, "relu"], [133, "sigmoid"]]

### TODO: Define your architecture.
def model_architecture(architecture):
    model = Sequential()
    nodes, function = architecture[0]
    model.add(Conv2D(nodes, (2, 2), activation=function, input_shape=train_tensors.shape[1:]))
    for nodes, function in architecture[1:-1]:
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(nodes, (2, 2), activation=function))
    
    model.add(GlobalAveragePooling2D())
    nodes, function = architecture[-1]
    model.add(Dense(nodes, activation=function))
    return model

model = model_architecture(architecture)
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 223, 223, 8)       104       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 8)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 110, 110, 16)      528       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 55, 55, 16)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 54, 54, 32)        2080      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 27, 27, 32)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 26, 26, 64)        8256      
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 12, 12, 128)       32896     
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               17157     
    =================================================================
    Total params: 61,021.0
    Trainable params: 61,021.0
    Non-trainable params: 0.0
    _________________________________________________________________


### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 5

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.8843 - acc: 0.0084Epoch 00000: val_loss improved from inf to 4.87069, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 58s - loss: 4.8842 - acc: 0.0085 - val_loss: 4.8707 - val_acc: 0.0108
    Epoch 2/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.8604 - acc: 0.0119Epoch 00001: val_loss improved from 4.87069 to 4.83325, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 58s - loss: 4.8606 - acc: 0.0118 - val_loss: 4.8333 - val_acc: 0.0132
    Epoch 3/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.7853 - acc: 0.0173Epoch 00002: val_loss improved from 4.83325 to 4.71942, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 59s - loss: 4.7850 - acc: 0.0172 - val_loss: 4.7194 - val_acc: 0.0192
    Epoch 4/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.6885 - acc: 0.0261Epoch 00003: val_loss improved from 4.71942 to 4.66157, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 58s - loss: 4.6886 - acc: 0.0260 - val_loss: 4.6616 - val_acc: 0.0311
    Epoch 5/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.6073 - acc: 0.0329Epoch 00004: val_loss improved from 4.66157 to 4.65566, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 58s - loss: 4.6077 - acc: 0.0328 - val_loss: 4.6557 - val_acc: 0.0263





    <keras.callbacks.History at 0x7f0799f172e8>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 3.7081%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229.0
    Trainable params: 68,229.0
    Non-trainable params: 0.0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6660/6680 [============================>.] - ETA: 0s - loss: 12.2076 - acc: 0.1195Epoch 00000: val_loss improved from inf to 10.73136, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 12.2138 - acc: 0.1193 - val_loss: 10.7314 - val_acc: 0.2144
    Epoch 2/20
    6640/6680 [============================>.] - ETA: 0s - loss: 10.1224 - acc: 0.2819Epoch 00001: val_loss improved from 10.73136 to 10.14909, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 10.1259 - acc: 0.2819 - val_loss: 10.1491 - val_acc: 0.2671
    Epoch 3/20
    6640/6680 [============================>.] - ETA: 0s - loss: 9.6934 - acc: 0.3367Epoch 00002: val_loss improved from 10.14909 to 10.10717, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.7049 - acc: 0.3361 - val_loss: 10.1072 - val_acc: 0.2862
    Epoch 4/20
    6560/6680 [============================>.] - ETA: 0s - loss: 9.4233 - acc: 0.3633Epoch 00003: val_loss improved from 10.10717 to 9.86264, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.4064 - acc: 0.3639 - val_loss: 9.8626 - val_acc: 0.3054
    Epoch 5/20
    6620/6680 [============================>.] - ETA: 0s - loss: 9.1544 - acc: 0.3905Epoch 00004: val_loss improved from 9.86264 to 9.64324, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 9.1617 - acc: 0.3901 - val_loss: 9.6432 - val_acc: 0.3138
    Epoch 6/20
    6580/6680 [============================>.] - ETA: 0s - loss: 8.9615 - acc: 0.4091Epoch 00005: val_loss improved from 9.64324 to 9.54229, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.9654 - acc: 0.4093 - val_loss: 9.5423 - val_acc: 0.3317
    Epoch 7/20
    6580/6680 [============================>.] - ETA: 0s - loss: 8.8623 - acc: 0.4191Epoch 00006: val_loss improved from 9.54229 to 9.43721, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.8523 - acc: 0.4196 - val_loss: 9.4372 - val_acc: 0.3437
    Epoch 8/20
    6600/6680 [============================>.] - ETA: 0s - loss: 8.5879 - acc: 0.4348Epoch 00007: val_loss improved from 9.43721 to 9.14776, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.5785 - acc: 0.4356 - val_loss: 9.1478 - val_acc: 0.3581
    Epoch 9/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.3802 - acc: 0.4560Epoch 00008: val_loss improved from 9.14776 to 9.08632, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.3861 - acc: 0.4557 - val_loss: 9.0863 - val_acc: 0.3677
    Epoch 10/20
    6660/6680 [============================>.] - ETA: 0s - loss: 8.3220 - acc: 0.4664Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 8.3315 - acc: 0.4656 - val_loss: 9.0953 - val_acc: 0.3629
    Epoch 11/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.2578 - acc: 0.4699Epoch 00010: val_loss improved from 9.08632 to 9.07694, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.2555 - acc: 0.4701 - val_loss: 9.0769 - val_acc: 0.3653
    Epoch 12/20
    6620/6680 [============================>.] - ETA: 0s - loss: 8.1503 - acc: 0.4793Epoch 00011: val_loss improved from 9.07694 to 8.85040, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.1416 - acc: 0.4795 - val_loss: 8.8504 - val_acc: 0.3737
    Epoch 13/20
    6620/6680 [============================>.] - ETA: 0s - loss: 8.0229 - acc: 0.4881Epoch 00012: val_loss improved from 8.85040 to 8.83570, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 8.0354 - acc: 0.4874 - val_loss: 8.8357 - val_acc: 0.3725
    Epoch 14/20
    6600/6680 [============================>.] - ETA: 0s - loss: 7.9667 - acc: 0.4935Epoch 00013: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 7.9754 - acc: 0.4930 - val_loss: 8.9441 - val_acc: 0.3737
    Epoch 15/20
    6560/6680 [============================>.] - ETA: 0s - loss: 7.9276 - acc: 0.5009Epoch 00014: val_loss improved from 8.83570 to 8.75215, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 7.9381 - acc: 0.5000 - val_loss: 8.7522 - val_acc: 0.3892
    Epoch 16/20
    6540/6680 [============================>.] - ETA: 0s - loss: 7.8411 - acc: 0.5009Epoch 00015: val_loss improved from 8.75215 to 8.66477, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 7.8664 - acc: 0.4993 - val_loss: 8.6648 - val_acc: 0.3844
    Epoch 17/20
    6620/6680 [============================>.] - ETA: 0s - loss: 7.7838 - acc: 0.5071Epoch 00016: val_loss improved from 8.66477 to 8.63707, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 7.7761 - acc: 0.5073 - val_loss: 8.6371 - val_acc: 0.3892
    Epoch 18/20
    6560/6680 [============================>.] - ETA: 0s - loss: 7.7312 - acc: 0.5123Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 7.7445 - acc: 0.5115 - val_loss: 8.6490 - val_acc: 0.3964
    Epoch 19/20
    6520/6680 [============================>.] - ETA: 0s - loss: 7.6664 - acc: 0.5143Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 7.6730 - acc: 0.5135 - val_loss: 8.6394 - val_acc: 0.3820
    Epoch 20/20
    6520/6680 [============================>.] - ETA: 0s - loss: 7.5353 - acc: 0.5224Epoch 00019: val_loss improved from 8.63707 to 8.57799, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s - loss: 7.5528 - acc: 0.5214 - val_loss: 8.5780 - val_acc: 0.4024





    <keras.callbacks.History at 0x7f0752e86e80>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 41.8660%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ Along with the pretrained model, an additional dense layer with 150 nodes is added as 150 is close to the number of breeds and this will learn additional features of the dog breeds. Since there is a chance of overfitting the model, a dropout layer with 20% chance is also added. This will ensure that the model generalises the data.

Transfer learning is better for deep learning porjects like this since it utilizes the previously trained neural networks and adapting for classifying dog breeds.  The earlier approaches are not as successful as using transfer learning since the whole network has to be trained with lots of additional data. This takes up lots of computations and also trial and error method done on top the architectures will take time to review. Instead the transfer learning uses set of previously trained model.


```python
### TODO: Define your architecture.
VGG19_breed_classifier = Sequential()
VGG19_breed_classifier.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
VGG19_breed_classifier.add(Dense(150, activation='relu'))
VGG19_breed_classifier.add(Dropout(0.2))
VGG19_breed_classifier.add(Dense(133, activation='softmax'))

VGG19_breed_classifier.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 150)               76950     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 150)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 133)               20083     
    =================================================================
    Total params: 97,033.0
    Trainable params: 97,033.0
    Non-trainable params: 0.0
    _________________________________________________________________


### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
VGG19_breed_classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', verbose=1, save_best_only=True)

VGG19_breed_classifier.fit(train_VGG19, train_targets, validation_data=(valid_VGG19, valid_targets), epochs=20,
                      batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6640/6680 [============================>.] - ETA: 0s - loss: 3.9758 - acc: 0.2072Epoch 00000: val_loss improved from inf to 1.89493, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 3.9683 - acc: 0.2079 - val_loss: 1.8949 - val_acc: 0.5066
    Epoch 2/20
    6520/6680 [============================>.] - ETA: 0s - loss: 1.7441 - acc: 0.5396Epoch 00001: val_loss improved from 1.89493 to 1.34091, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 1.7416 - acc: 0.5409 - val_loss: 1.3409 - val_acc: 0.6216
    Epoch 3/20
    6580/6680 [============================>.] - ETA: 0s - loss: 1.2132 - acc: 0.6594Epoch 00002: val_loss improved from 1.34091 to 1.16505, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 1.2076 - acc: 0.6605 - val_loss: 1.1651 - val_acc: 0.6647
    Epoch 4/20
    6540/6680 [============================>.] - ETA: 0s - loss: 0.9677 - acc: 0.7272Epoch 00003: val_loss improved from 1.16505 to 1.10856, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 0.9678 - acc: 0.7281 - val_loss: 1.1086 - val_acc: 0.7102
    Epoch 5/20
    6580/6680 [============================>.] - ETA: 0s - loss: 0.8102 - acc: 0.7669Epoch 00004: val_loss improved from 1.10856 to 1.08725, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 0.8158 - acc: 0.7660 - val_loss: 1.0873 - val_acc: 0.7102
    Epoch 6/20
    6600/6680 [============================>.] - ETA: 0s - loss: 0.6908 - acc: 0.7956Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.6890 - acc: 0.7960 - val_loss: 1.1418 - val_acc: 0.7198
    Epoch 7/20
    6500/6680 [============================>.] - ETA: 0s - loss: 0.6218 - acc: 0.8168Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.6181 - acc: 0.8175 - val_loss: 1.0873 - val_acc: 0.7341
    Epoch 8/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.5487 - acc: 0.8362Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.5557 - acc: 0.8343 - val_loss: 1.1695 - val_acc: 0.7317
    Epoch 9/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.4834 - acc: 0.8552Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.4848 - acc: 0.8545 - val_loss: 1.1248 - val_acc: 0.7329
    Epoch 10/20
    6660/6680 [============================>.] - ETA: 0s - loss: 0.4413 - acc: 0.8709Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.4415 - acc: 0.8705 - val_loss: 1.1963 - val_acc: 0.7210
    Epoch 11/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.4089 - acc: 0.8781Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.4092 - acc: 0.8781 - val_loss: 1.2500 - val_acc: 0.7401
    Epoch 12/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.4040 - acc: 0.8822Epoch 00011: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.4035 - acc: 0.8820 - val_loss: 1.3367 - val_acc: 0.7413
    Epoch 13/20
    6620/6680 [============================>.] - ETA: 0s - loss: 0.3553 - acc: 0.8959Epoch 00012: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.3543 - acc: 0.8964 - val_loss: 1.1523 - val_acc: 0.7533
    Epoch 14/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.3464 - acc: 0.9003Epoch 00013: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.3455 - acc: 0.9004 - val_loss: 1.3769 - val_acc: 0.7533
    Epoch 15/20
    6620/6680 [============================>.] - ETA: 0s - loss: 0.3334 - acc: 0.9077Epoch 00014: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.3333 - acc: 0.9076 - val_loss: 1.3246 - val_acc: 0.7533
    Epoch 16/20
    6580/6680 [============================>.] - ETA: 0s - loss: 0.3134 - acc: 0.9099Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.3131 - acc: 0.9105 - val_loss: 1.4032 - val_acc: 0.7365
    Epoch 17/20
    6580/6680 [============================>.] - ETA: 0s - loss: 0.2915 - acc: 0.9155Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.2947 - acc: 0.9147 - val_loss: 1.4338 - val_acc: 0.7509
    Epoch 18/20
    6600/6680 [============================>.] - ETA: 0s - loss: 0.2910 - acc: 0.9191Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.2899 - acc: 0.9193 - val_loss: 1.3771 - val_acc: 0.7665
    Epoch 19/20
    6560/6680 [============================>.] - ETA: 0s - loss: 0.2722 - acc: 0.9235Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.2734 - acc: 0.9234 - val_loss: 1.5218 - val_acc: 0.7461
    Epoch 20/20
    6640/6680 [============================>.] - ETA: 0s - loss: 0.2662 - acc: 0.9276Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 2s - loss: 0.2672 - acc: 0.9272 - val_loss: 1.5333 - val_acc: 0.7521





    <keras.callbacks.History at 0x7f063c3ea5c0>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
VGG19_breed_classifier.load_weights('saved_models/weights.best.VGG19.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.
VGG19_breed_prediction = [np.argmax(VGG19_breed_classifier.predict(np.expand_dims(bt_features, axis=0)))
                          for bt_features in test_VGG19]

test_accuracy = 100*np.sum(np.asarray(VGG19_breed_prediction) == np.argmax(test_targets, axis=1))/len(VGG19_breed_prediction)
    
print("Test accuracy is {0:0.4f}".format(test_accuracy))
```

    Test accuracy is 72.2488


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def breed_predictor(image_path):
    # bottleneck features extraction
    bottleneck_feature = extract_VGG19(path_to_tensor(image_path))
    # prediction
    predicted = VGG19_breed_classifier.predict(bottleneck_feature)
    # returning prediction
    return dog_names[np.argmax(predicted)]
    
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def image_detector(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.show()
    if face_detector(image_path):
        print("The person in this image looks like a {}.".format(breed_predictor(image_path)))
    elif not face_detector(image_path):
        print("This dog looks like a {}.".format(breed_predictor(image_path)))
    else:
        print("Looks like it is difficult to predict!")
```


```python
def detector_images(images_path):
    for image_path in images_path:
        image_detector(image_path)
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ The dog breed classifier is able to classify the dog breeds very accurately except for few images where the images are themselves not very clear. The output is better than what I expected. The algorithm can be improved by adding some of the features below.
* Detection of number of human and dog faces in a photo.
* Classify the dog breeds in a given picture.
* Training of the algorithm with more images and also increasing the number of iterations for learning.
* Providing confidence value for the prediction to the user.


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
%matplotlib inline
images = glob("images/*jpg")
detector_images(images)
```


![png](dog_app/output_73_0.png)


    This dog looks like a French_bulldog.



![png](dog_app/output_73_2.png)


    This dog looks like a Labrador_retriever.



![png](dog_app/output_73_4.png)


    This dog looks like a Labrador_retriever.



![png](dog_app/output_73_6.png)


    The person in this image looks like a Afghan_hound.



![png](dog_app/output_73_8.png)


    This dog looks like a Curly-coated_retriever.



![png](dog_app/output_73_10.png)


    This dog looks like a Brittany.



![png](dog_app/output_73_12.png)


    This dog looks like a Irish_red_and_white_setter.



![png](dog_app/output_73_14.png)


    The person in this image looks like a Dogue_de_bordeaux.



![png](dog_app/output_73_16.png)


    This dog looks like a Chesapeake_bay_retriever.



![png](dog_app/output_73_18.png)


    The person in this image looks like a Basenji.



![png](dog_app/output_73_20.png)


    This dog looks like a Irish_water_spaniel.



![png](dog_app/output_73_22.png)


    This dog looks like a Golden_retriever.

