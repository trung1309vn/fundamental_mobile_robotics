"""_NOTE_
For this part, we do not have any template. You may need some python library to read the csv file.
you have 10,000 random numbers.
obtain the range of the given numbers. 
divide the range into "n" intervals. For instance 50.
for each interval, count the number of random numbers which are between that interval. 
For each interval, you have a number showing the abundance of the random numbers withing that interval. 
Normalize the distribution of the random varialbe --the 50 numbers you obtianed-- by dividing them to the 1,000.
draw the bar chart. (x is the middle of the interval and y is the normalized number you derived.)
"""
import numpy as np
data = np.asarray(np.genfromtxt('discrete_random_variable.csv', delimiter=','))
range = (np.min(data), np.max(data))


