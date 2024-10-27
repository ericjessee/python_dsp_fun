import numpy as np
import matplotlib.pyplot as plt

# Triangle kernel function
def triangle_kernel(t, width=1.0):
    if abs(t) <= width:
        return 1 - abs(t) / width
    else:
        return 0.0

# Rectangle function generator
def rectangle_data(width, length, dt=1.0):
    """
    Generates a rectangular function data centered at 0.

    Parameters:
        width (float): Width of the rectangle (total time the function is 1).
        length (int): Total number of data points.
        dt (float): Time step between points.
    
    Returns:
        numpy array: Data points representing a rectangular function.
    """
    data = np.zeros(length)
    center = length // 2
    half_width = int((width / dt) / 2)
    data[center - half_width:center + half_width + 1] = 1
    return data

def finite_convolution_continuous(kernel_func, g, dt=1.0):
    g = np.array(g)
    n = len(g)
    conv_result = np.zeros(n)
    
    # Calculate the normalization factor by integrating the kernel over its width
    kernel_sum = sum(kernel_func(t) for t in np.arange(0, n * dt, dt))
    
    for i in range(n):
        for j in range(n):
            time = (i - j) * dt
            if time >= 0:
                conv_result[i] += (kernel_func(time) / kernel_sum) * g[j]
    
    return conv_result


# Scatter plot function
def scatter_plot(data1, data2, labels=('Data Set 1', 'Data Set 2'), colors=('blue', 'orange')):
    x1, y1 = data1
    x2, y2 = data2
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x1, y1, c=colors[0], label=labels[0], alpha=0.7)
    plt.scatter(x2, y2, c=colors[1], label=labels[1], alpha=0.7)
    
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.title('Scatter Plot of Data Before and After Convolution')
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Test function
def test_convolution_with_triangle_kernel():
    length = 500       # Length of the data array (number of data points)
    width = 100        # Width of the rectangle function in time units
    dt = 1.0           # Time step

    # Generate the rectangular data points
    rect_data = rectangle_data(width, length, dt)
    indices = np.arange(length)

    # Define kernel function with fixed width
    kernel_width = 100
    conv_result = finite_convolution_continuous(lambda t: triangle_kernel(t, width=kernel_width), rect_data, dt=dt)
    
    # Prepare data for scatter plot
    data_before = (indices, rect_data)
    data_after = (indices, conv_result)
    
    # Plot data before and after convolution
    scatter_plot(data_before, data_after, labels=('Before Convolution', 'After Convolution'))

def plot_kernel(kernel_func, time_range=(-10, 10), dt=0.1, width=1.0):
    """
    Plots the specified kernel function over a given time range.

    Parameters:
        kernel_func (function): The kernel function to plot.
        time_range (tuple): The range of time values (start, end) to plot.
        dt (float): Time step for sampling the kernel function.
        width (float): Width parameter for the kernel function.
    """
    # Generate time values
    t_values = np.arange(time_range[0], time_range[1], dt)
    # Evaluate the kernel function at each time value
    kernel_values = [kernel_func(t, width=width) for t in t_values]
    
    # Plot the kernel function
    plt.figure(figsize=(8, 6))
    plt.plot(t_values, kernel_values, label=f'Triangle Kernel (width={width})', color='purple')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Plot of the Triangle Kernel Function')
    plt.legend()
    plt.grid(True)
    
    # Show plot
    plt.show()

# Call the test function
#plot_kernel(triangle_kernel, time_range=(-5, 5), dt=0.1, width=2.0)
test_convolution_with_triangle_kernel()
