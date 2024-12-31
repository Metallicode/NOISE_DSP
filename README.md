# NOISE_DSP
a collection of various noise generation algorithms for your pleasure

### Overview
This project is a Python-based terminal application for generating various types of noise algorithms, commonly used in audio Digital Signal Processing (DSP). The application allows users to choose from a diverse set of noise types and outputs them as `.wav` files. Each algorithm has unique characteristics and applications, ranging from audio synthesis to testing and procedural generation.

### Features
The application supports the following noise algorithms:

1. **White Noise**:
   - Random values where all frequencies have equal power.
   - Commonly used for audio testing and masking.

2. **Pink Noise**:
   - Power decreases with increasing frequency (1/f spectrum).
   - Mimics natural sounds like rainfall or heartbeats.

3. **Brown Noise**:
   - Power decreases more steeply with frequency than pink noise.
   - Simulates random walks and produces a "darker" sound.

4. **Blue Noise**:
   - Power increases with frequency.
   - Suitable for dithering and anti-aliasing.

5. **Violet Noise**:
   - Power increases steeply with frequency.
   - Rarely used but useful for scientific applications.

6. **Gaussian Noise**:
   - Values follow a Gaussian distribution.
   - Used in simulations and statistical modeling.

7. **Uniform Random Noise**:
   - Random values are uniformly distributed.
   - Simple and effective for creating randomness.

8. **Perlin Noise**:
   - Smooth, gradient-based noise.
   - Frequently used for procedural textures and sound modulations.

9. **Simplex Noise**:
   - A computationally efficient alternative to Perlin noise.
   - Produces smoother, artifact-free noise.

10. **Fractal Noise**:
    - Combines multiple octaves of noise for complexity.
    - Used in creating realistic textures and soundscapes.

11. **Granular Noise**:
    - Constructs noise from small overlapping "grains."
    - Produces textured, dynamic noise patterns.

12. **Random Walk Noise**:
    - Values evolve through random steps.
    - Produces a "wandering" sound, ideal for dynamic modulations.

13. **Chaotic Noise**:
    - Generated using chaotic systems like the logistic map.
    - Pseudo-random with deterministic properties.

14. **Markov Chain-Based Noise**:
    - Random values governed by a Markov process.
    - Useful for controlled generative audio.

15. **Tonal Noise**:
    - Combines random noise with periodic tones.
    - Creates hybrid textures with tonal characteristics.

16. **Stochastic Resonance Noise**:
    - Adds noise to weak signals to enhance them.
    - Mimics phenomena observed in biological systems.

17. **Cellular Automata Noise**:
    - Noise based on cellular automata rules.
    - Produces evolving patterns for dynamic soundscapes.

### How to Use
1. **Run the Application**:
   - Use the terminal to run the script: `python noise_generator.py`.

2. **Select an Algorithm**:
   - Choose an algorithm from the displayed menu by entering its corresponding number.

3. **Set Parameters**:
   - Enter the desired length of the audio (in seconds).
   - Specify the output file name (e.g., `output.wav`).

4. **Generate and Save**:
   - The application generates the noise and saves it as a `.wav` file in the specified location.

### Requirements
- Python 3.7 or later
- Required Python packages:
  - `numpy`
  - `scipy`
  - `noise`

Install the dependencies using:
```bash
pip install numpy scipy noise
```

### Applications
- **Audio Testing**: Validate sound systems and filters with different types of noise.
- **Sound Design**: Create textures, ambiances, and effects.
- **Procedural Generation**: Use noise in games, simulations, and animations.
- **Scientific Research**: Simulate natural phenomena or random systems.

### Customization
Many algorithms allow parameter customization (e.g., frequency, persistence, octaves). Edit these parameters in the source code or extend the program to accept additional user inputs.


