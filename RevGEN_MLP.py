import numpy as np
from numpy.random import default_rng
from math import exp, log



class Linear_layer:
    def __init__(self, x, activation_function="sigmoid"):
        self.input = x
        self.input_shape = self.input.shape[0]
        self.weights = np.random.uniform(-1, 1, (self.input_shape, self.input_shape))
        self.bias = np.random.uniform(-1, 1, (self.input_shape, 1))
        self.determinant = 0
        self.y          = np.zeros((self.input_shape, 1))
        self.z          = np.zeros((self.input_shape, 1))
        self.x_reversed = np.zeros((self.input_shape, 1))
        self.reverse_constant = 0
        self.activation_function = activation_function

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-1 * input))

    def sigmoid_rev(self, input):
        return -np.log((1 - input) / input)

    def leaky_ReLu(self, input):
        return np.where(input >= 0, input, 0.01 * input)

    def leaky_ReLu_rev(self, input):
        return np.where(input >= 0, input, input / 0.01)

    def softmax(self, input):
        input_stable = input - np.max(input)
        exp_values = np.exp(input_stable)
        exp_sum = np.sum(exp_values)
        return exp_values / exp_sum, np.log(exp_sum) + np.max(input)

    def softmax_rev(self, input):
        return (log(input) + self.reverse_constant)

    def forward(self, input):
        self.input = input
        self.z = np.matmul(self.weights, self.input) + self.bias
        if self.activation_function == "sigmoid":
            self.y = self.sigmoid(self.z)
        elif self.activation_function == "leaky_ReLu":
            self.y = self.leaky_ReLu(self.z)
        else:
            self.y = self.z
        return self.y

    def reverse(self,input):

        self.y = input

        self.determinant = np.linalg.det(self.weights)
        if self.activation_function == "sigmoid":
            self.x_reversed = self.sigmoid_rev(self.y)

            if self.determinant != 0:
                diff = self.x_reversed - self.bias
                weight_inverse = np.linalg.inv(self.weights)
                self.x_reversed = np.dot(weight_inverse, diff)
                return self.x_reversed
            else:
                print("Not invertible")           

        elif self.activation_function == "leaky_ReLu":
            self.x_reversed = self.leaky_ReLu_rev(self.y)
            
            if self.determinant != 0:
                diff = self.x_reversed - self.bias
                weight_inverse = np.linalg.inv(self.weights)
                self.x_reversed = np.dot(weight_inverse, diff)
                return self.x_reversed
            else:
                print("Not invertible")       


        else:
            self.x_reversed = self.y


class Output_layer:
    def __init__(self, x, output_shape, activation_function="sigmoid"):
        self.input = x
        self.input_shape = self.input.shape[0]
        self.output_shape = output_shape
        self.weights = np.random.uniform(-1, 1, (self.output_shape, self.input_shape))
        self.shape_diff = self.input_shape - self.output_shape
        self.bias = np.random.uniform(-1, 1, (self.output_shape, 1))

        # Modifications to Weight matrix for the pass through latent variables not used for softmax
        if self.input_shape > self.output_shape:
            appended_matrix = np.zeros((self.shape_diff,self.input_shape))
            appended_bias = np.zeros((self.shape_diff,1))
            for i in range(self.shape_diff):
                appended_matrix[i][self.input_shape-i-1] = 1
            self.weights = np.vstack((self.weights,appended_matrix))
            self.bias = np.vstack((self.bias,appended_bias))

        
        self.determinant = 0
        self.y          = np.zeros((self.input_shape, 1))
        self.z          = np.zeros((self.input_shape, 1))
        self.x_reversed = np.zeros((self.input_shape, 1))
        self.reverse_constant = 0
        self.activation_function = activation_function

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-1 * input))

    def sigmoid_rev(self, input):
        return -np.log((1 - input) / input)

    def softmax(self, input):
        input_stable = input - np.max(input)
        exp_values = np.exp(input_stable)
        exp_sum = np.sum(exp_values)
        return exp_values / exp_sum, np.log(exp_sum) + np.max(input)

    def softmax_rev(self, input):
        return (np.log(input) + self.reverse_constant)

    def forward(self, input):
        self.input = input
        self.z = np.matmul(self.weights, self.input) + self.bias
        if self.activation_function == "sigmoid":
            self.y = self.sigmoid(self.z)
        elif self.activation_function == "softmax":
            self.y[0:self.output_shape], self.reverse_constant = self.softmax(self.z[0:self.output_shape])
            self.y[self.output_shape:] = self.z[self.output_shape:]
        return self.y

    def reverse(self,input):
        self.y = input
        self.determinant = np.linalg.det(self.weights)
        if self.activation_function == "sigmoid":
            self.x_reversed[0:self.output_shape] = self.sigmoid_rev(self.y[0:self.output_shape])
            self.x_reversed[self.output_shape:] = self.y[self.output_shape:]

            if self.determinant != 0:
                diff = self.x_reversed - self.bias

                weight_inverse = np.linalg.inv(self.weights)
                self.x_reversed = np.dot(weight_inverse, diff)
                return self.x_reversed
            else:
                print("Not invertible")           

        elif self.activation_function == "softmax":
            self.x_reversed[0:self.output_shape] = self.softmax_rev(self.y[0:self.output_shape])
            self.x_reversed[self.output_shape:] = self.y[self.output_shape:]
            
            if self.determinant != 0:
                diff = self.x_reversed - self.bias
                #print(diff.shape)
                #print(self.x_reversed.shape)
                #print(self.bias.shape)
                weight_inverse = np.linalg.inv(self.weights)
                self.x_reversed = np.dot(weight_inverse, diff)
                return self.x_reversed
            else:
                print("Not invertible")       
        else:
            self.x_reversed = self.y



class RevGEN_MLP():
    def __init__(self, n_layers, x, y_actual, epochs, loss_function):
        self.n_layers = n_layers
        self.input = x
        self.input_shape = x.shape[0]
        self.layer_list = []
        self.y_actual = y_actual
        self.latent = 0
        self.y_pred = 0
        self.lr = 0.002  
        self.num_epochs = epochs
        self.loss_function = loss_function

        temp_input = self.input
        for i in range(self.n_layers):
            linear_layer = Linear_layer(temp_input, activation_function="leaky_ReLu")
            output = linear_layer.forward(temp_input)
            self.layer_list.append(linear_layer)
            temp_input = output

        output_layer = Output_layer(temp_input, activation_function="softmax", output_shape= self.y_actual.shape[0])
        self.y_pred = output_layer.forward(temp_input)
        self.layer_list.append(output_layer)

        self.weight_derivative_list = []
        self.bias_derivative_list = []

    def loss_fn(self,input, target):
        self.input = input
        self.y_actual = target
        if self.loss_function == "MSE":
            num_elements = self.y_pred.shape[0]
            loss = (1 / num_elements) * ((self.y_actual - self.y_pred) ** 2).sum()
        elif self.loss_function == "binary_crossentropy":
            loss = - (self.y_actual * np.log(self.y_pred) + (1 - self.y_actual) * np.log(1 - self.y_pred))
        elif self.loss_function == "cross_entropy":
            e = 0.00001 
            loss = -np.sum(self.y_actual * np.log(self.y_pred[0:self.y_actual.shape[0]])+e)

        
        return np.mean(loss)
    

    def loss_derivative(self):
        if self.loss_function == "MSE":
            num_elements = self.y_pred.shape[0]
            return ((-2 / num_elements) * (self.y_actual - self.y_pred))
        elif self.loss_function == "binary_crossentropy":
            return (-(self.y_actual / self.y_pred) + ((1 - self.y_actual) / (1 - self.y_pred)))
        elif self.loss_function == "cross_entropy":
            return self.y_pred[0:self.y_actual.shape[0]] - self.y_actual

    def sigmoid(self, input):
        return 1 / (1 + np.exp(-1 * input))
    def sigmoid_rev(self, input):
        return -np.log((1 - input) / input)
    def sigmoid_derivative(self, input):
        sig = self.sigmoid(input)
        return sig * (1 - sig)

    def leaky_ReLu(self, input):
        return np.where(input > 0, input, 0.01 * input)
    def leaky_ReLu_rev(self, input):
        return np.where(input >= 0, input, input / 0.01)
    def leaky_ReLu_derivative(self, input):
        return np.where(input > 0, 1.0, 0.01)

    def softmax(self, input):
        input_stable = input - np.max(input)
        exp_values = np.exp(input_stable)
        exp_sum = np.sum(exp_values)
        return exp_values / exp_sum
    
    def softmax_rev(self, input):
        return (np.log(input) + self.reverse_constant)
    
    def softmax_derivative(self, softmax_output, true_label):
        return softmax_output - true_label

    def forward(self, input):
        self.input = input
        current_input = input
        for layer in self.layer_list:
            current_input = layer.forward(current_input)
            if layer == self.layer_list[-2]:
                self.latent = current_input
        self.y_pred = current_input
        return self.y_pred

    def backward(self):
        self.weight_derivative_list = []
        self.bias_derivative_list = []
        y = self.forward(self.input)
        d_error = self.loss_derivative()

        for i in reversed(range(len(self.layer_list))):
            if self.layer_list[i].activation_function == "sigmoid":
                da_dz = self.sigmoid_derivative(self.sigmoid_rev(y)) 
            elif self.layer_list[i].activation_function == "leaky_ReLu":
                da_dz = self.leaky_ReLu_derivative(self.leaky_ReLu_rev(y)) 
            elif self.layer_list[i].activation_function == "softmax":
                da_dz = self.softmax_derivative(self.layer_list[i].y[0:self.y_actual.shape[0]], self.y_actual)

            dz_dw = self.layer_list[i].reverse(y)

            y = self.layer_list[i].reverse(y)

            if self.layer_list[i].activation_function == "softmax":
                derivative_vector = d_error
            else:
                derivative_vector = np.multiply(d_error, da_dz)
            dW = np.dot(derivative_vector, dz_dw.transpose())
            db = derivative_vector

            self.weight_derivative_list.insert(0, dW)
            self.bias_derivative_list.insert(0, db)
            if self.layer_list[i].activation_function == "softmax":
                d_error = np.dot(self.layer_list[i].weights[0:self.y_actual.shape[0]].transpose(), derivative_vector)
            else:
                d_error = np.dot(self.layer_list[i].weights.transpose(), derivative_vector)

    def train(self, input, target):
        self.input = input
        self.y_actual = target

        self.backward()

        for i in range(len(self.layer_list)):
            if self.layer_list[i].activation_function == "softmax":
                self.layer_list[i].weights[0:self.y_actual.shape[0]] -= self.lr * self.weight_derivative_list[i]
                self.layer_list[i].bias[0:self.y_actual.shape[0]] -= self.lr * self.bias_derivative_list[i]
            else:
                self.layer_list[i].weights -= self.lr * self.weight_derivative_list[i]
                self.layer_list[i].bias -= self.lr * self.bias_derivative_list[i]                

        self.forward(self.input)

    def reverse(self,input):
        x_rev = input
        for i in reversed(range(len(self.layer_list))):
            x_rev = self.layer_list[i].reverse(x_rev)
        return x_rev



