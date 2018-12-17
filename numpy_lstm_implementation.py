'''
Implementation of LSTM using numpy	

'''
'''

#Important please read:
Why  LSTM is best for all of our use cases (along with steps to implement)

https://docs.google.com/document/d/1pCacDrNdJPV0dOiuA2sFUTUNT1tJ_CO33dlaHNWnT3k/edit?usp=sharing

'''

'''
Below is the theoretical formulation for LSTM model using basic numpy array manipulation
'''
import numpy as np
x = np.array([[1., 1., 1.]])
c = 0.1 * np.asarray([[0, 1]])
h = 0.1 * np.asarray([[2, 3]])
#(Refer Image)Since we have 4 gates, we need weight array of 4 * num_units (2).
#We have proj_size 5 because LSTM uses previous output. If we have input_size 3, we concatenate previous h(size 2) then we have 5 proj_size. The value of num_units is size of c and h.
#print(h)
num_units = 2
args = np.concatenate((x,h), axis=1)
#print(args)
out_size = 4 * num_units
#get shape of column i.e. get number of columns obtained by concatenation of x and h
proj_size = args.shape[-1]
#print(out_size)
#print(proj_size)
#forming an array of 0.5s according to rows and columns with reference from out_size and proj_size 
weights = np.ones([proj_size, out_size]) * 0.5
#print(weights)
#Taking ot products (using matmul as it allows scalar multiplication)
out = np.matmul(args, weights)
#print(out)	
#a = np.arange(2*2*4).reshape((2,2,4))
#print(a)
#b = np.arange(2*2*4).reshape((2,4,2))
#print(b)
#c = np.matmul(a,b)
#print(c)
bias = np.ones([out_size]) * 0.5
#print(bias)
concat = out + bias
#print(concat)
# we have 4 gates for LSTM
i, j, f, o = np.split(concat, 4, 1)
#print(i)
#print(j)
#print(f)
#print(o)
g = np.tanh(j)
#print(g)
#Real formulation starts from here::::
#creating the forget gate by sigmoid function i.e 1/(1+e^-x)
def sigmoid_array(x):
	return 1/(1+np.exp(-x))
forget_bias = 1.0
sigmoid_f = sigmoid_array(f + forget_bias)
#print(sigmoid_f)
#print(sigmoid_array(i))
#print(c)
# calculate C
new_c = c * sigmoid_f + sigmoid_array(i) * g
#print(new_c)
#calculate h
#print(new_c)
#tanh = ((e^x - e^-x))/((e^x + e^-x))
d = np.tanh(new_c)
#print(d)
#final ouput:
new_h = np.tanh(new_c) * sigmoid_array(o)
 
print(new_h)
print(new_c)
''''
Refer the sample results if we implement this using Tensorflow,keras on Nvidia Graphic card using abstractive summarization:
Review(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
Summary(1): great coffee
Review(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!
Summary(2): omg gross gross
Review(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
Summary(3): love it
''''

