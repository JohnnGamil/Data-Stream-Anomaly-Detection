import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the data stream simulation function based on the changes
def data_stream_simulation(size=1000, base_level=10, noise_level=0.7, anomaly_chance=0.005, spike_intensity=(5, 10)):
    
    """
    param size: Total number of data points to simulate.
    param base_level: Base value of the data stream (constant or slightly fluctuating).
    param noise_level: Standard deviation of Gaussian noise added to the data.
    param anomaly_chance: Probability of an anomaly occurring at each step.
    param spike_intensity: Range of intensities for anomaly spikes (min, max).
    """
    
    data = []
    for i in range(size):
        # Generate base value with Gaussian noise
        base_value = base_level + np.random.normal(0, noise_level)
        
        # Inject anomaly (spike) with a small probability
        if random.random() < anomaly_chance:
            value = base_value + random.uniform(*spike_intensity)
        else:
            value = base_value
        data.append(value)
    return data

def visualize_stream(stream, thresh):
    
    """
    Real-time visualization of the data stream with anomaly detection.
    :param stream: Data stream generator.
    :param thresh: Threshold multiplier for anomaly detection.
    """
    
    data = []
    anomalies = []
    plt.ion()       
    fig, ax = plt.subplots() 

    # Check for window close event
    try:
        for value in tqdm(stream):
            # Stop loop if window is closed
            if not plt.fignum_exists(fig.number):
                print("Plot closed, stopping visualization.")
                break

            data.append(value)
            is_anomaly = (value > np.array(data).mean() + np.array(data).std() + thresh) or (value < np.array(data).mean() - np.array(data).std() - thresh)

            if is_anomaly:
                anomalies.append(len(data) - 1)

            ax.clear()
            ax.plot(data, label="Data Stream")

            if anomalies:
                ax.scatter(anomalies, [data[i] for i in anomalies], color='red', label="Anomalies", zorder=5)
            plt.legend()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("Interrupted, stopping visualization.")
    
    plt.ioff()
    plt.show()

# Simulate the data stream and run visualization
stream = data_stream_simulation(size=1000)
visualize_stream(stream, 2)
