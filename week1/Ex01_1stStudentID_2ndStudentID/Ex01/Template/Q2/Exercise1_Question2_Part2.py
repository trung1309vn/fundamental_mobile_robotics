"""_NOTE_
For this part, we do not have any template. You may need some python library to read the csv file.

you have 10,000 random numbers.

obtain the range of the given numbers. 

divide the range into "n" intervals. For instance 50.

for each interval, count the number of random numbers which are between that interval. 

For each interval, you have a number showing the abundance of the random numbers withing that interval. 

Normalize the distribution of the random varialbe--the 50 numbers you obtianed--by dividing them to the 1,000.

draw the bar chart. (x is the middle of the interval and y is the normalized number you derived.)

"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("discrete_random_variable.csv", delimiter=",")
n = 50
x_lin = np.linspace(np.min(data), np.max(data)+0.0000001, n+1)
interval = (np.max(data) - np.min(data)) / n
mid_interval_lists = []
norm_lists = []

# For each interval
for i in range(n):
    # Get number from an interval
    values = data[(data >= x_lin[i]) & (data < x_lin[i+1])]
    # Get number of element in current interval
    num_val = len(values)

    # Get middle interval
    mid_interval_lists.append(x_lin[i]+interval/2)
    norm_lists.append(num_val/len(data))

print(np.sum(norm_lists))
plt.bar(mid_interval_lists, norm_lists)
plt.xlabel('x')
plt.ylabel('Normalized Count')
plt.title('Distribution of Random Numbers')
plt.show()

