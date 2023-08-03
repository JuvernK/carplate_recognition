#%%
import numpy as np
from PIL import Image
import cv2
import random
import glob

class Neural_Network:
    def __init__(self,  outputSize = 20):
        # The width and height of the image
        self.width = 50
        self.height = 60
        self.inputSize = self.width * self.height
        self.outputSize = outputSize
        self.hiddenSize =  200 # 40 # round(self.inputSize * 4/3)

        # self initialisation on the weight and bias ######################## for test pur poses
        # self.input_data = np.array([[0.2], [0.5]])
        # self.w_ji = np.array([[0.1, 0.2], [0.3,0.4]])    # input to hidden
        # self.w_kj = np.array([[0.5, 0.6], [0.7,0.8]])   
        # self.bias_j = np.array([[0.2], [0.2]])    # cha
        # self.bias_k = np.array([[0.4], [0.4]])    # cha   
        # self.target = np.array([[0.2],[0.8]])


    def sigmoid(self, z):
        """
        Activation Function for forward propagation
        """    
        return 1/(1+np.exp(-z))

    def read_file(self, infile, target_value):
        """
        Initialise:
            1) Input neuron 
            2) Target data 
        of the neural network

        infile : File that contains files of the selected alphabets and number in required naming format
        target_value : Expected output of the neural network in string, exp: "1", "7" and "10"(which refers to B)...etc
        """
        # Reading of Input File, and Target File.
        input_image = Image.open(infile).convert("L") 
        
        # Resize the image with width = 32 and height = 49
        input_image = input_image.resize((self.width, self.height))
        
        # Convert the image to array
        input_data = np.array(input_image)

        # Flatten the array to 1D, and normalize the data between 0 and 1. 
        self.input_data = (input_data.flatten() / 255)

        # Make each input data into an array as each pixel is a neuron
        self.input_data =  self.input_data.reshape((self.width * self.height), 1)  

        # Initialize the target array set the only the target position to 1
        self.target = np.zeros((self.outputSize, 1),  dtype=int) 
        self.target[target_value] = 1

    def Weight_Initialization(self, reload = False):
        """
        Initialise the weights and bias of the neural network 
        """
        
        if reload:
            self.w_ji = np.load('w_ji.npy')
            self.w_kj = np.load('w_kj.npy')
            self.bias_j = np.load('bias_j.npy')
            self.bias_k = np.load('bias_k.npy')
    
        # Initializing of the Weights with random float number between -0.5 to 0.5.
        else:
            np.random.seed(1)
            self.w_ji= np.random.uniform(-0.5, 0.5, size=(self.hiddenSize, self.inputSize))    # j = input to hidden
            self.w_kj = np.random.uniform(-0.5, 0.5, size=(self.outputSize, self.hiddenSize))  # k = hidden to output
            self.bias_j = np.random.uniform(0, 1, size=(self.hiddenSize, 1))
            self.bias_k = np.random.uniform(0, 1, size=(self.outputSize, 1))

    def Forward_Input_Hidden(self):
        """
        Forward Propagation from Input -> Hidden Layer, i represents the neuron number at Input Layer.
        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"): 
        1) NetJ = ∑ni=0 (W[j][i] * input[i])  , where n = number of input neuron
        2) OutJ = 1 / (1 + e^( -NetJ[j]+biasJ[j] ) )
        """
        self.NetJ = np.dot(self.w_ji, self.input_data)    
        self.OutJ = self.sigmoid(np.add(self.NetJ, self.bias_j))
     
    def Forward_Hidden_Output(self):
        """
        Forward Propagation from Hidden -> Output Layer, j represents the neuron number at K Layer.
        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"):
        1) NetK = ∑ni=0 (W[j][i] * OutJ[i])
        2) OutK = 1 / (1 + e^( -NetK[j]+biasK[j] ) )
        """
        self.NetK = np.dot(self.w_kj, self.OutJ)
        self.OutK = self.sigmoid(np.add(self.NetK, self.bias_k))

    def Error_Correction(self):
        """
        Error Correction
        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"):
        1) E = 1/2 * ∑ni=0 (Target[i] - OutputK[i])^2
        """
        # print("Each error is: \n", 0.5 * np.square(self.target - self.OutK))
        self.total_error = 0.5 * np.sum(np.square(self.target - self.OutK))


    def Weight_Bias_Correction_Output(self):
        """
        Correction of Weights and Bias between Hidden and Output Layer.
        delta_E_WK = delta_E_outK * delta_outK_netK * delta_netK_wK

        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"):
        1) delta_Wk[j][i] = OutK[j] * (1 - Target[j]) * OutK[j] * (1 - OutK[j] ) * OutJ[i]
        2) delta_biasK[j] = (OutK[j] - Target[j]) * OutK[j] * (1 - OutK[j])
        """

        # delta_𝑊𝐾 𝑗 [ 𝑖 ] = 𝑂𝑢𝑡𝐾[ 𝑗 ] − 𝑇𝑎𝑟𝑔𝑒𝑡[ 𝑗 ] ∗ 𝑂𝑢𝑡𝐾[ 𝑗 ] 1 − 𝑂𝑢𝑡𝐾[ 𝑗 ] * 𝑂𝑢𝑡𝐽[ 𝑖 ]
        # delta_𝑏𝑖𝑎𝑠𝐾[ 𝑗 ] = 𝑂𝑢𝑡𝐾[ 𝑗 ] − 𝑇𝑎𝑟𝑔𝑒𝑡[ 𝑗 ] ∗ 𝑂𝑢𝑡𝐾[ 𝑗 ] 1 − 𝑂𝑢𝑡𝐾[ 𝑗 
        # Calculate the weight correction terms
        self.delta_WKj =  np.outer((self.OutK - self.target) * (self.OutK * (1-self.OutK) ), self.OutJ)
        self.delta_biasKj = (self.OutK - self.target) * (self.OutK *( 1 - self.OutK)) 
    
    def Weight_Bias_Correction_Hidden(self):
        """
        Correction of Weights and Bias between Hidden and Output Layer.
        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"):
        1) delta_Wk[j][i] = OutK[j] * (1 - Target[j]) * OutK[j] * (1 - OutK[j] ) * OutJ[i]
        2) delta_biasK[j] = (OutK[j] - Target[j]) * OutK[j] * (1 - OutK[j])
        """
        # Calculate the weight correction terms
        self.delta_WJi = np.outer((self.OutJ * (1-self.OutJ) * np.dot(self.w_kj.T, (self.OutK - self.target) * (self.OutK * (1-self.OutK) ))), self.input_data)
        # Calculate the bias correction terms
        self.delta_biasJi = (self.OutJ * (1-self.OutJ) * np.dot(self.w_kj.T, (self.OutK - self.target) * (self.OutK * (1-self.OutK) )))

    
    def Weight_Bias_Update(self, learning_rate):
        """
        Update the weights and bias by multiplying the learning rate with the delta weights and bias
        Formula used (obtained from lecture notes "FIT3181 Week 5 part 1"):
        1) W[j][i] = W[j][i] - learning_rate * delta_Wk[j][i]
        2) biasK[j] = biasK[j] - learning_rate * delta_biasK[j]
        3) W[j][i] = W[j][i] - learning_rate * delta_Wk[j][i]
        4) biasK[j] = biasK[j] - learning_rate * delta_biasK[j]
        """

        self.w_kj -= learning_rate * self.delta_WKj
        self.bias_k -= learning_rate * self.delta_biasKj

        self.w_ji -= learning_rate * self.delta_WJi
        self.bias_j -= learning_rate * self.delta_biasJi


    def Check_for_End(self, tolerance_error = 0.02):
        """
        Check if the error is less than the tolerance error
        """
        self.Error_Correction()
        if self.total_error < tolerance_error:
            return True
        else:
            return False
        
    def Saving_Weights_Bias(self):
        """
        Save the weights and bias into a numpy file to be loaded later
        """
        np.save('w_ji.npy', self.w_ji)
        np.save('w_kj.npy', self.w_kj)
        np.save('bias_j.npy', self.bias_j)
        np.save('bias_k.npy', self.bias_k)

    def training(self, folder):
        """
        Training of the Neural Network
        """
        # Randomly iterate through all files in the folder from 0 to 19
        # Read the training folder, example: "train_data/8\\001.png"
        random.seed(7)
        # List all image file paths
        img_path = []
        for folders in range(20):
            paths = f"{folder}/{folders}/"  # Folders = the folder from 0 to 19
        #     for num in range(8):
        #         img_path.extend(glob.glob(paths + "00" + str(num) + "*.png"))  # Adjust the file extension if needed
        # random.shuffle(img_path)
            img_path.extend(glob.glob(paths + "*.png"))
        random.shuffle(img_path)
        # print(img_path)
        
        # Initialize the weights and bias
        self.Weight_Initialization()

        # Training of the Neural Network
        for i in range(200):
            print("Current epoch is: ", i)
            for img in img_path:
                
                # Identify the target
                if len(img) == 20:
                    target_value = int(img[11]) # train_data/0 # How to capture the two value 12 12 13
                elif len(img) == 21:
                    target_value = int(img[11] + img[12]) # train_data/0 # How to capture the two value 12 12 13
            
                # Read the image file and convert it to array
                self.read_file(img, target_value)

                # Forward Propagation
                self.Forward_Input_Hidden()
                self.Forward_Hidden_Output()

                # Error Correction
                self.Error_Correction()

                # Weight and Bias Correction
                self.Weight_Bias_Correction_Output()
                self.Weight_Bias_Correction_Hidden()

                # Weight and Bias Update
                self.Weight_Bias_Update(learning_rate= 0.5)

            # Check if the error is within the tolerance
            if self.Check_for_End():
                print("The total error is: ", self.total_error)
                print("The total iteration is: ", i)
                break

        # Save the weights and bias
        self.Saving_Weights_Bias()

    def testing(self, folder):
        """
        Testing of the Neural Network
        """
        # List all image file paths
        img_path = []
        for folders in range(20):
            paths = f"{folder}/{folders}/"
            img_path.extend(glob.glob(paths + "*.png"))
        random.shuffle(img_path)
        print(img_path)
        
        # Check the accuracy of the Neural Network, there are 40 images in total
        correct = 0
        for img in img_path:
            # Identify the target
            print("Test img is: \n", img)
            if len(img) == 19:
                target_value = int(img[10]) 
            elif len(img) == 20:
                target_value = int(img[10] + img[11]) 
        
            # Read the image file and convert it to array
            self.read_file(img, target_value)

            # Initialize the weights and bias
            self.Weight_Initialization(reload=True)

            # Forward Propagation
            self.Forward_Input_Hidden()
            self.Forward_Hidden_Output()

            print("The predicted is: ", np.argmax(self.OutK), "The target is: ", np.argmax(self.target))

            # Check if the output is correct
            if np.argmax(self.OutK) == np.argmax(self.target):
                correct += 1
                
        print("The accuracy is: ", correct/len(img_path))

    def test_cropped_plates(self, folder):
        """
        Test the accuracy of the cropped plates
        """
        img_path = []
        no_of_carplate = 10
        for folders in range(1, no_of_carplate + 1):
            paths = r"{}/Carplate_{}/".format(folder, folders)
            img_path.extend(glob.glob(paths + "*.jpg"))
        
        # Check the accuracy of the Neural Network, there are 69 images in total
        correct = 0
        iter= 0 
        for img in img_path:
            # Identify the target
            print("Test img is: \n", img)
            iter +=1

            if iter < 64: # 1.jpg to 
                if len(img) == 36:  # 10.jpg
                    target_value = int(img[30:32])
                elif len(img) == 37: # 8_1.jpg
                    target_value = int(img[30])
                elif len(img) == 38: # 10_1
                    target_value = int(img[30:32])
                else:
                    target_value = int(img[30])

            else:
                if len(img) == 37:  # 10.jpg
                    target_value = int(img[31:33])
                elif len(img) == 38: # 8_1.jpg
                    target_value = int(img[31])
                elif len(img) == 39: # 10_1
                    target_value = int(img[31:32])
                else:
                    target_value = int(img[31])


            # Read the image file and convert it to array
            self.read_file(img, target_value)

            # Initialize the weights and bias
            self.Weight_Initialization(reload=True)

            # Forward Propagation
            self.Forward_Input_Hidden()
            self.Forward_Hidden_Output()

            print("The predicted car's character is: ", np.argmax(self.OutK), "The target is: ", np.argmax(self.target))

            # Check if the output is correct
            if np.argmax(self.OutK) == np.argmax(self.target):
                correct += 1

        print("The accuracy is: ", correct/len(img_path))




ann = Neural_Network()
ann.training("train_data")
ann.testing("test_data")    # Accuracy is 40%

# Uncomment the code below to test the cropped plates
    # Before running, make sure the naming of each image in the folder is according to the naming format for the program to know the target
# ann.test_cropped_plates("Cropped_Characters") # Accuracy is 49%, width = 50, height = 60, hiddenSize = 200

