#implementation of multilayer perceptron to simulate XOR using BP

import numpy as np
from matplotlib import pyplot

def sigmoid(x):
	return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
	return x*(1-x)
#XOR input
inputs=np.array([[0,0],[0,1],[1,0],[1,1]])
output=np.array([[0],[1],[1],[0]])

#learning rate be 0.01
epoch=50000
lr=0.1
input_layer,hidden_layer,output_layer=2,2,1
er=[]
#Random initialization of weight and Bias
hidden_weight=np.random.random(size=(input_layer,hidden_layer))
hidden_bias=np.random.random(size=(1,hidden_layer))
output_weight=np.random.random(size=(hidden_layer,output_layer))
output_bias=np.random.random(size=(1,output_layer))

print("Initial hidden weights: ",end='')
print(*hidden_weight)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weight)
print("Initial output biases: ",end='')
print(*output_bias)

#training algorithm
for _ in range(epoch):
	#feedforward propagation 
	hidden_layer_activation=np.dot(inputs,hidden_weight)
	hidden_layer_activation += hidden_bias
	hidden_layer_output =sigmoid(hidden_layer_activation)

	output_layer_activation=np.dot(hidden_layer_output,output_weight)
	output_layer_activation += output_bias
	predicted_output=sigmoid(output_layer_activation)

	#backpropagation 
	error=output - predicted_output
	er.append(error.mean())
	#print("for", _ ,"error :",error)
	derivative_predicted_output=error*derivative_sigmoid(predicted_output)

	error_hidden_layer=derivative_predicted_output.dot(output_weight.T)
	derivative_hidden_layer=error_hidden_layer*derivative_sigmoid(hidden_layer_output)

	#updating weight and bias
	output_weight +=hidden_layer_output.T.dot(derivative_predicted_output)*lr
	output_bias += np.sum(derivative_predicted_output,axis=0,keepdims=True)*lr
	hidden_weight += inputs.T.dot(derivative_hidden_layer)*lr
	hidden_bias += np.sum(derivative_hidden_layer,axis=0,keepdims=True)*lr

#final 
print("Final hidden weights: ",end='')
print(*hidden_weight)
print("Final hidden biases: ",end='')
print(*hidden_bias)
print("Final output weights: ",end='')
print(*output_weight)
print("Final output biases: ",end='')
print(*output_bias)
print("error :",error)
print("output :",predicted_output)

