from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

#img_path = "C:/Users/Daithi/Documents/Coding/flowers/102flowers/jpg/image_00001.jpg"
def preprocessing(img_path, new_size):
    image = Image.open(img_path)  
    image = image.resize(new_size)
    image_array = np.array(image)
    image.close()
    image_array = greyscale(image_array)
    image_array = edge_detection(image_array)
    return image_array

""" def preprocessing(img_path, new_size):

    image = Image.open(img_path)
    image_array = np.array(image)
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/0original.jpg', image_array)
    image.close()

    image_array = greyscale(image_array)
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/1greyscale.jpg', image_array, cmap="gray")
    edge_array = edge_detection(image_array)
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/2edge.jpg', edge_array, cmap="gray")

    composite = image_array + edge_array
    #composite = normalise(composite) 
    composite = np.clip(image_array, 0, 255)#prevent values going over 255, under 0
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/3composite.jpg', composite, cmap="gray")

    scaled_image = scale(composite, new_size) #this saves and loads an image which is kinda crap, also calls greyscale again :(
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/4scaled.jpg', scaled_image, cmap="gray")

    edges_again = edge_detection(scaled_image)
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/5second_edge.jpg', edges_again, cmap="gray")

    final = scaled_image + edges_again #again prevent overflows
    #final = normalise(final)
    final = np.clip(final, 0, 255)
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/6final.jpg', final, cmap="gray")
   
    plt.imshow(final, cmap='gray')
    plt.show()
    return final  """

def normalise255(image_array):
    image_array = image_array/255
    return image_array

def normalise(image_array):
    max_value = np.max(image_array)
    image_array = image_array/max_value 
    return image_array
    
def greyscale(image_array):
    """ for i in range(new_size[0]):
        for j in range(new_size[1]):
            R,G,B = image.getpixel((i,j))
            normalized_image[i,j] = (0.299 * R + 0.587 * G + 0.114 * B) / 255 """
    greyscale_image = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
    return greyscale_image

def edge_detection(image_array):
    #kernel for finding feature map
    kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    convulsed_array = np.zeros(image_array.shape)
    #new_size = image_array.shape[0] + 2, image_array.shape[1] + 2
    new_size = tuple(np.array(image_array.shape) + 2)
    bordered_array = np.zeros(new_size)

    #add border to image
    """ for x in range(1, image_array.shape[0]):
        for y in range(1, image_array.shape[1]):
            bordered_array[x+1,y+1] = image_array[x,y] """
    bordered_array[1:-1,1:-1] = image_array

    #kernel convolution
    for x in range(1, image_array.shape[0]):
        for y in range(1, image_array.shape[1]):

            image_section = np.array([[bordered_array[x-1,y-1], bordered_array[x,y-1], bordered_array[x+1,y-1]],
                                      [bordered_array[x-1,y], bordered_array[x,y], bordered_array[x+1,y]],
                                      [bordered_array[x-1,y+1], bordered_array[x,y+1], bordered_array[x+1,y+1]]])

            convulsed_array[x,y] = np.sum(image_section * kernel, axis=None) 
    
    return convulsed_array

    
def scale(image_array, new_size):
    plt.imsave('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/temp.jpg', image_array, cmap="gray")
    image = Image.open('C:/Users/Daithi/Documents/Coding/flowers/preprocess_tests/temp.jpg')
    image = image.resize(new_size)
    scaled_image = np.array(image)

    scaled_image = greyscale(scaled_image)

    image.close()
    return scaled_image
 

def preprocess_files(folder_path, input_size, new_size):
    file_list = os.listdir(folder_path)
    num_files = len(file_list)
    
    x_train = [[] for _ in range(num_files)]
    for i in range(num_files):
        x_train[i] = [0]*input_size

    h = 0
    for filename in os.listdir(folder_path):
        # Check if the file is an image file
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            #flatten first
            x_train[h] = (preprocessing(image_path, new_size).flatten())/255
            h += 1
            print(h)

    print("Preprocessing Complete!")

    # convert x_train to a numpy array
    x_train = np.array(x_train)

    # save x_train to a file
    np.save('x_train_small.npy', x_train)