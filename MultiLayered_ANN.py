import numpy as np
import math

#----------------------------------------
# Step 1:- Input Layer 
#----------------------------------------

x1 = 2.0
x2 = 3.0
print("-----------------------------------------------")
print("Step 1:- Input Layer")
print("-----------------------------------------------")

print("Input Features : X (These are the features)")
print(f"x1 = {x1}")
print(f"x2 = {x2}")

#----------------------------------------
# Step 2:- Hidden Layer 
#----------------------------------------
print("-----------------------------------------------")
print("Step 2:- Hidden Layer ( 2 Neurons)")
print("-----------------------------------------------")

print("Hidden Neuron 1")

w11 = 0.5
w12 = -0.2
b1 = 0.1

print("Weights :")
print(f"w11 : {w11}")
print(f"w12 : {w12}")

print("Bias :")
print(f"b : {b1}")

print("Weighted Sum :")
print("z1 = (x1 * w11) + (x2 * w12) + b1")

z1 = (x1 * w11) + (x2 * w12) + b1

print("Weighted Sum(z1) :",z1)

h1 = max(0,z1)

print("Output of hidden neuron1 :", h1) 

print("-----------------------------------------------")

print("Hidden Neuron 2")

w21 = 0.8
w22 = 0.4
b2 = -0.1

print("Weights :")
print(f"w21 : {w21}")
print(f"w22 : {w22}")

print("Bias :")
print(f"b : {b2}")

print("Weighted Sum :")
print("z2 = (x1 * w21) + (x2 * w22) + b2")

z2 = (x1 * w21) + (x2 * w22) + b2

print("Weighted Sum(z2) :",z2)

h2 = max(0,z2)

print("Output of hidden neuron1 :", h2) 

#----------------------------------------
# Step 3:- Output Layer 
#----------------------------------------
print("-----------------------------------------------")
print("Step 3:- Output Layer")
print("-----------------------------------------------")

w_out1 = 1.0
w_out2 = -1.5
b_out = 0.2

print("Weights :")
print(f"w_out1 : {w_out1}")
print(f"w_out2 : {w_out2}")

print("Bias :")
print(f"b_out : {b_out}")

z_out = (h1 * w_out1) + (h2 * w_out2) + b_out

print("Weighted Sum :", z_out)

#Sigmoid
z = 1 / (1 + math.exp(-z_out))

print("-----------------------------------------------")
print("------------Neural Network Summary------------")
print("-----------------------------------------------")

print("Input Layer")
print(f"x1 : {x1}")
print(f"x2 : {x2}")
print(" ")

print("Hidden Layer")
print(f"h1 : {h1}")
print(f"h2 : {h2}")
print(" ")

print("Output Layer")
print(f"z : {z}")

print("-----------------------------------------------")
print("Prediction of Neural Network")
print("-----------------------------------------------")

if(z >= 0.5):
    print("Predicted as Positive class")

else:
    print("Predicted as Negative class")
print(" ")