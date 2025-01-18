import numpy as np
import matplotlib.pyplot as plt

def generate_interference_pattern(freq1, freq2, size=500, amplitude=1):
    """
    Generate an interference pattern for two frequencies.

    Args:
        freq1 (float): Frequency of the first wave.
        freq2 (float): Frequency of the second wave.
        size (int): Size of the grid for visualization.
        amplitude (float): Amplitude of the waves.

    Returns:
        None. Displays the interference pattern.
    """
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    X, Y = np.meshgrid(x, y)

    # Calculate two waves
    wave1 = amplitude * np.sin(freq1 * X + freq1 * Y)
    wave2 = amplitude * np.sin(freq2 * X + freq2 * Y)

    # Combine the waves to create the interference pattern
    interference = wave1 + wave2

    # Visualize the pattern
    plt.figure(figsize=(8, 8))
    plt.imshow(interference, cmap='viridis', extent=(0, 2 * np.pi, 0, 2 * np.pi))
    plt.colorbar(label="Amplitude")
    plt.title(f"Interference Pattern: Frequencies {freq1} Hz and {freq2} Hz")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.axis("off")
    plt.show()
    
    

def generate_circular_interference_pattern(freq1, freq2, resolution=500):
    """
    Generates a 2D circular interference pattern from two frequencies.
    
    Args:
        freq1: Frequency of the first wave.
        freq2: Frequency of the second wave.
        resolution: Resolution of the circular grid (higher is more detailed).
    """
    # Create a polar coordinate grid
    theta = np.linspace(0, 2 * np.pi, resolution)
    r = np.linspace(0, 1, resolution)
    r, theta = np.meshgrid(r, theta)
    
    # Convert polar coordinates to Cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Generate interference pattern in polar coordinates
    pattern = np.sin(2 * np.pi * freq1 * r) + np.sin(2 * np.pi * freq2 * r)
    
    # Plot the interference pattern in circular form
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(x, y, pattern, shading='auto', cmap='viridis')
    plt.axis('off')  # Remove axes for a clean look
    plt.gca().set_aspect('equal')  # Ensure the circle is not distorted
    plt.title("Circular Interference Pattern", fontsize=14)
    plt.show()    
    
    

def generate_chladni_pattern(m, n, plate_size=(1, 1), resolution=500):
    """
    Generates and visualises a Chladni pattern based on mode numbers (m, n).

    Args:
        m (int): Number of half-wavelengths along the x-axis.
        n (int): Number of half-wavelengths along the y-axis.
        plate_size (tuple): Size of the plate (width, height).
        resolution (int): Resolution of the grid for the simulation.
    """
    # Define plate dimensions
    width, height = plate_size

    # Create a grid of points
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    x, y = np.meshgrid(x, y)

    # Generate standing wave patterns
    z = np.cos(m * np.pi * x / width) * np.cos(n * np.pi * y / height)

    # Visualise the pattern
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, z, levels=100, cmap="binary")  # Use a binary colour map for simplicity
    plt.axis("off")
    plt.title(f"Chladni Pattern (m={m}, n={n})", fontsize=14)
    plt.show()

def generate_complex_chladni(frequencies, plate_size=(1, 1), resolution=500):
    """
    Generates and visualises a Chladni-like pattern based on a complex waveform.

    Args:
        frequencies (list): List of frequencies to combine into the pattern.
        plate_size (tuple): Size of the plate (width, height).
        resolution (int): Resolution of the grid for the simulation.
    """
    if not frequencies:
        raise ValueError("The frequencies list must not be empty.")

    # Define plate dimensions
    width, height = plate_size

    # Create a grid of points
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    x, y = np.meshgrid(x, y)

    # Initialise the pattern
    z = np.zeros_like(x)

    # Add contributions from all frequencies
    for freq in frequencies:
        z += np.cos(freq * np.pi * x / width) * np.cos(freq * np.pi * y / height)

    # Normalise the resulting pattern
    z /= len(frequencies)

    # Visualise the pattern
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, z, levels=100, cmap="binary")  # Use binary colormap for simplicity
    plt.axis("off")
    plt.title(f"Complex Chladni Pattern (frequencies={frequencies})", fontsize=14)
    plt.show()
    


def generate_enhanced_chladni(frequencies, plate_size=(1, 1), resolution=500):
    """
    Generates and visualises a Chladni pattern with enhanced complexity.

    Args:
        frequencies (list): List of frequencies (or mode numbers) to combine.
        plate_size (tuple): Size of the plate (width, height).
        resolution (int): Resolution of the grid for the simulation.
    """
    # Define plate dimensions
    width, height = plate_size

    # Create a grid of points
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    x, y = np.meshgrid(x, y)

    # Initialise the pattern
    z = np.zeros_like(x)

    # Simulate mode shapes using sinusoidal functions
    for i, freq in enumerate(frequencies):
        mode_x = freq % 5 + 1  # Mode number for x-direction
        mode_y = freq // 5 + 1  # Mode number for y-direction
        z += np.sin(mode_x * np.pi * x / width) * np.sin(mode_y * np.pi * y / height)

    # Add some randomisation for natural effects
    z += 0.1 * np.random.randn(*z.shape)

    # Normalise and enhance the result
    z = np.sin(z)  # Non-linear transformation to create sharper features

    # Visualise the pattern
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, z, levels=100, cmap="binary")
    plt.axis("off")
    plt.title(f"Enhanced Chladni Pattern (frequencies={frequencies})", fontsize=14)
    plt.show()

# Example Usage
generate_enhanced_chladni(frequencies=[2, 5, 50, 103], plate_size=(1, 1), resolution=1000)
   
    


#generate_complex_chladni(frequencies=[2,5,7,9,23], plate_size=(1, 1), resolution=1000)

#f1 = float(input("please enter 1 frequency:\n"))
#f2 = float(input("please enter 2 frequency:\n"))
#n = float(input("please enter n:\n"))
#generate_interference_pattern(freq1=f1, freq2=f2, size=500)
#generate_circular_interference_pattern(freq1=f1, freq2=f2, resolution=500)
#generate_chladni_pattern(m=f1, n=3, plate_size=(1, 1), resolution=1000)
