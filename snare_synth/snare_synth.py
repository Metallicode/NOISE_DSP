import numpy as np
import wave
from noise_algorithms import *



def apply_attack_decay(signal, sample_rate, attack, decay):
    """
    Apply an Attack-Decay envelope with exponential decay to the signal.
    Parameters:
    - signal: Input signal (numpy array).
    - sample_rate: Sample rate in Hz.
    - attack: Attack duration in seconds.
    - decay: Decay duration in seconds.
    """
    num_samples = len(signal)
    attack_samples = int(sample_rate * attack)
    decay_samples = int(sample_rate * decay)

    if attack_samples + decay_samples > num_samples:
        raise ValueError("Attack and decay durations exceed signal length.")

    # Create Attack-Decay envelope
    envelope = np.zeros(num_samples)

    # Linear Attack phase
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Exponential Decay phase
    decay_x = np.linspace(0, 1, decay_samples)
    envelope[attack_samples:attack_samples + decay_samples] = np.exp(-5 * decay_x)

    # Apply envelope to signal
    return signal * envelope


def generate_snare(sample_rate=44100, length=1.0, algorithm=white_noise, attack=0.01, decay=0.2, choice=1):
    """
    Generate a snare drum sample using Attack-Decay envelope.
    Parameters:
    - sample_rate: Sample rate in Hz.
    - length: Length of the sample in seconds.
    - algorithm: Noise algorithm to use.
    - attack: Attack duration in seconds.
    - decay: Decay duration in seconds.
    """
    # Generate noise using the selected algorithm
    if choice in ["8", "9", "10"]:  # These require additional parameters
        signal = algorithm(length, octaves=3)  # Customize the parameters as needed
    elif choice == "11":  # Granular Noise requires additional parameters
        signal = algorithm(length, grain_size=0.1, overlap=0.4)
    elif choice == "12":  # Random Walk Noise requires step size
        signal = algorithm(length, step_size=0.01)
    elif choice == "13":  # Chaotic Noise requires parameters r and x0
        signal = algorithm(length, r=3.95, x0=0.6)
    elif choice == "14":  # Markov Chain-Based Noise requires states and transition matrix
        signal = algorithm(length, states=[-1, 0, 1])
    elif choice == "15":  # Tonal Noise requires frequency and noise level
        signal = algorithm(length, frequency=440, noise_level=0.2)
    elif choice == "16":  # Stochastic Resonance Noise requires weak signal frequency and noise level
        signal = algorithm(length, weak_signal_frequency=2, noise_level=0.2)
    elif choice == "17":  # Cellular Automata Noise requires grid size and rule
        signal = algorithm(length, grid_size=100, rule=30)
    else:
        signal = algorithm(length)
 

    # Apply Attack-Decay envelope
    signal = apply_attack_decay(signal, sample_rate, attack, decay)

    # Normalize to [-1, 1]
    signal = signal / max(np.abs(signal))
    return signal


def save_to_wav(filename, signal, sample_rate=44100):
    """Save the generated sample to a .wav file."""
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        signal = (signal * 32767).astype(np.int16)  # Scale to 16-bit range
        wav_file.writeframes(signal.tobytes())

def main():
    # Select noise algorithm and parameters
    algorithms = {
        "1": ("White Noise", white_noise),
        "2": ("Pink Noise", pink_noise),
        "3": ("Brown Noise", brown_noise),
        "4": ("Blue Noise", blue_noise),
        "5": ("Violet Noise", violet_noise),
        "6": ("Gaussian Noise", gaussian_noise),
        "7": ("Uniform Random Noise", uniform_random_noise),
        "8": ("Perlin Noise", perlin_noise),
        "9": ("Simplex Noise", simplex_noise),
        "10": ("Fractal Noise", fractal_noise),
        "11": ("Granular Noise", granular_noise),
        "12": ("Random Walk Noise", random_walk_noise),
        "13": ("Chaotic Noise", chaotic_noise),
        "14": ("Markov Chain-Based Noise", markov_chain_noise),
        "15": ("Tonal Noise", tonal_noise),
        "16": ("Stochastic Resonance Noise", stochastic_resonance_noise),
        "17": ("Cellular Automata Noise", cellular_automata_noise),
    }

    print("Snare Drum Synth")
    print("=================")
    for key, (name, _) in algorithms.items():
        print(f"{key}. {name}")
    choice = input("Select a noise algorithm: ")

    if choice not in algorithms:
        print("Invalid choice!")
        return

    length = float(input("Enter length of the sample (in seconds): "))
    attack = float(input("Enter attack duration (seconds): "))
    decay = float(input("Enter decay duration (seconds): "))
    filename = input("Enter output file name (e.g., snare.wav): ")

    _, generator = algorithms[choice]
    print("Generating snare sample...")
    signal = generate_snare(
        sample_rate=44100,
        length=length,
        algorithm=generator,
        attack=attack,
        decay=decay,
        choice=choice
    )
    save_to_wav(filename, signal)
    print(f"Snare sample saved to {filename}")
    
if __name__ == "__main__":
    main()

