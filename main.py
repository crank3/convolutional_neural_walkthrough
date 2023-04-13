

# This runs Python 3.10 (Latest Version as of August 29, 2022)
# To install PIL, put 'pip install Pillow' in the terminal
# To install webdriver_manager, put 'pip install webdriver-manager' in the terminal
# To install cv2 (OpenCV), search 'opencv-python' in package manager

import random
import torch
import torch.nn as nn
import torch.nn.functional as nnfunc
from PIL import Image, ImageOps, ImageFont, ImageDraw
import torchvision.transforms as transforms
import os
import cv2






# ################################## Making the Neural Network ##################################


# We define class "Net" which inherits from nn.Module (neural network in pytorch)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # Input 1 image, output 6 images. 5x5 convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)  # Input 6 images, output 16 images. 5x5 convolution kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimensions after all convolutions and maxpools, 16 from previous outputs
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # This is the method we will call to feed data to the neural network
        x = nnfunc.max_pool2d(nnfunc.relu(self.conv1(x)), 2)  # Max pooling over a (2, 2) window. This cuts the width and height in half.
        x = nnfunc.max_pool2d(nnfunc.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flattens x into 1 dimension
        x = nnfunc.relu(self.fc1(x))  # Gets RELU of all activations of fc1
        x = self.dropout(x)
        x = nnfunc.relu(self.fc2(x))  # Gets RELU of all activations of fc2
        x = self.dropout(x)
        x = nnfunc.sigmoid(self.fc3(x))  # Gets Sigmoid of all activations of fc3
        return x

network = Net()  # network is instance of class Net, which we defined above



# Whether or not to load the previous network
load_network = input(f'\n{"+"*80}\nUse Downloaded Network? (y/n): ')
if load_network == "y":
    network.load_state_dict(torch.load('saved_network/network_meat'))
    network.eval()










# ################################## Preparing to Train Network ##################################

#  This function cleans input_image to feed to our network.
def clean_image(input_image):
    input_image = input_image.resize((32, 32))  # Resizes the image to 32x32 pixels
    # input_image = input_image.convert('RGB')  # Makes sure the image is in RGB format. Not needed if grayscaled
    input_image = ImageOps.grayscale(input_image)  # Grayscales the image

    # Transform Image to Tensor.
    # We want it to be a tensor of format []
    input_image_tensor = transforms.ToTensor()(input_image)
    input_image_tensor = [input_image_tensor.tolist()]
    input_image_tensor = torch.tensor(input_image_tensor)
    return input_image_tensor



# Iterates over files in folder "hotdog_pictures", and makes list of all pictures.
hotdog_images = []
for filename in os.listdir("hotdog_pictures"):
    f = os.path.join("hotdog_pictures", filename)
    if os.path.isfile(f):
        image = Image.open(f)
        image_tensor = clean_image(image)
        hotdog_images.append(image_tensor)

# Iterates over files in folder "hotdog_pictures", and makes list of all pictures.
nothotdog_images = []
for filename in os.listdir("nothotdog_pictures"):
    f = os.path.join("nothotdog_pictures", filename)
    if os.path.isfile(f):
        image = Image.open(f)
        image_tensor = clean_image(image)
        nothotdog_images.append(image_tensor)




# ################################## Training Loop ##################################
# Big loop that feeds the network pictures and then preforms backpropagation

loss_func = nn.L1Loss()  # Defining loss function (a metric of the network's accuracy)
learning_rate = 0.001  # The rate at which the network learns (keep this super low)


training_loops = int(input(f'\n{"+"*80}\nTraining Loops (even integer): '))  # keep to an even number

if training_loops != 0:

    losses = []
    for i in range(training_loops):
        print(f'\n{"_"*80}\nLoop {i}:\n')


        # Forward propagation
        file_num = random.randint(0, 1)  # 1 = hotdog, 0 = nothotdog
        picture = 0
        if file_num == 1:
            pic_num = random.randint(0, len(hotdog_images)-1)
            picture = hotdog_images[pic_num]
        else:
            pic_num = random.randint(0, len(nothotdog_images)-1)
            picture = nothotdog_images[pic_num]
        print(f'\nFile Num (1 = hotdog, 0 = nothotdog): {file_num}')
        print(f'\nPicture Num: {pic_num}')
        output = network(picture)
        print(f'\nOutput:\n{output[0][0]}')


        # Calculate Loss
        target = torch.tensor(file_num)
        target = target.view(1, -1)
        print(f'\nTarget:\n{target[0][0]}')

        loss = loss_func(output, target)
        print(f'\nLoss: {loss}')
        losses.append(float(loss))


        # Back propagation
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\n\n{"="*60}\n{"="*60}\n')




    # Printing Loss

    recent_percent = 10  # most recent % of losses to look at
    recent_loss = 0  # loss within the last <recent_percent>% of
    total_loss = 0
    for loss_i in losses:
        total_loss += loss_i
        if losses.index(loss_i) == (training_loops / 2):
            print(f'\n1st Half Average Loss = {total_loss / (training_loops/2)}')
            total_loss = 0
        elif losses.index(loss_i) == (training_loops-1):
            print(f'\n2nd Half Average Loss = {total_loss / (training_loops/2)}')
        if losses.index(loss_i) >= (training_loops-(training_loops*(1.0/recent_percent))):
            recent_loss += loss_i
    recent_loss = recent_loss/(training_loops*(1/recent_percent))
    print(f'\nRecent (last {recent_percent}%) Average Loss = {recent_loss}')

    total_loss = 0



    # Save Network
    saving = input(f'\n{"+"*80}\nSave Network? (y/n): ')
    if saving == "y":
        torch.save(network.state_dict(), "saved_network/network_meat")










# ################################## Testing the Network (with webcam) ##################################

testing = input(f'\n{"+"*80}\nTest Network? (y/n): ')
if testing == "y":
    testing = True
else:
    testing = False



ding_threshold = 0.999998  # 0.9999999  0.9999998

# Video Loop
camera = cv2.VideoCapture(0)

while testing:
    try:
        frame = camera.read()

        display_frame = frame

        frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)  # frame is the object to use for the network
        frame = Image.fromarray(frame)



        # Feeding to Network

        fres = frame.size  # Webcam Image Resolution = (640, 480)

        subdiv = 4  # 2 works
        jumpx = fres[0]/(subdiv*4)
        jumpy = fres[1]/(subdiv*4)

        for iy in range(0, fres[1] - int(fres[1] / subdiv)):
            for ix in range(0, fres[0] - int(fres[0]/subdiv)):
                if ((ix % jumpx) == 0) and ((iy % jumpy) == 0):
                    subframe = frame.crop((ix, iy, ix + int(fres[0]/subdiv), iy + int(fres[1]/subdiv)))
                    network_frame = clean_image(subframe)  # frame cleaned to put into network
                    output = network(network_frame)
                    if output[0][0] > ding_threshold:  # If network detects enough hotdog
                        print('DING')
                        # subframe.show()
                        for iiy in range(iy, iy + int(fres[1] / subdiv)):
                            for iix in range(ix, ix + int(fres[0] / subdiv)):
                                if ((iix % 2) == 0) and ((iiy % 2) == 0):
                                    display_frame[1][iiy][iix] = [0, 255, 255]  # [B, G, R]
                    # print(f'\nOutput:\n{output[0][0]}')

                    # print(f'top-left = ({ix}, {iy})\n\n')
                    # subframe.show()
                    # cv2.waitKey(500)


        network_frame = clean_image(frame)  # frame cleaned to put into network
        output = network(network_frame)
        # print(f'\nOutput:\n{output[0][0]}')



        if output[0][0] > ding_threshold:
            print('DING !!!!!!!!!!!!!!!!!!')
        else:
            print('.')






        cv2.imshow('Live_Video', display_frame[1])  # display_frame[1] is the object to use for this function

        if cv2.waitKey(1) >= 0:
            break
    except:
        print('oops!')


camera.release()
cv2.destroyAllWindows()
