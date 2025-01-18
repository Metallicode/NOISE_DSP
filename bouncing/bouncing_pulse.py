import numpy as np
import wave


    
def generate_bouncing_pulse2(length, sample_rate=44100, initial_interval=0.5, pulse_duration=0.01, bounces=50):
    """
    Generate a bouncing pulse effect with evenly distributed bounces.
    Parameters:
    - length: Length of the output in seconds.
    - sample_rate: Sample rate in Hz.
    - initial_interval: Initial interval between bounces in seconds.
    - pulse_duration: Duration of each pulse in seconds.
    - bounces: Total number of bounces to simulate.
    Returns:
    - signal: Numpy array representing the bouncing pulse effect.
    """
    total_samples = int(length * sample_rate)
    signal = np.zeros(total_samples)

    # Calculate the interval for each bounce
    intervals = np.linspace(initial_interval, initial_interval / (bounces * 0.5), bounces)

    current_position = 0

    for i, interval in enumerate(intervals):
        # Calculate the start and end of the current pulse
        start = int(current_position * sample_rate)
        end = start + int(pulse_duration * sample_rate)

        if end >= total_samples:
            break

        # Generate a pulse (simple square wave for now)
        signal[start:end] = 1.0

        # Update position
        current_position += interval

    # Normalize signal
    signal = signal / max(np.abs(signal))
    return signal

def generate_bouncing_pulse(length, sample_rate=44100, initial_interval=0.5, pulse_duration=0.01):
    """
    Generate a bouncing pulse effect that lasts for the full length.
    Parameters:
    - length: Length of the output in seconds.
    - sample_rate: Sample rate in Hz.
    - initial_interval: Initial interval between bounces in seconds.
    - pulse_duration: Duration of each pulse in seconds.
    Returns:
    - signal: Numpy array representing the bouncing pulse effect.
    """
    total_samples = int(length * sample_rate)
    signal = np.zeros(total_samples)

    # Calculate the time positions for each bounce
    bounces = 2000  # Number of bounces to simulate
    intervals = np.logspace(np.log10(initial_interval), np.log10(initial_interval / 10), bounces)

    # Normalize intervals to ensure the sequence lasts the full duration
    total_time = sum(intervals)
    scaling_factor = length / total_time
    intervals *= scaling_factor

    current_position = 0

    for interval in intervals:
        # Calculate the start and end of the current pulse
        start = int(current_position * sample_rate)
        end = start + int(pulse_duration * sample_rate)

        if end >= total_samples:
            break

        # Generate a pulse (simple square wave for now)
        signal[start:end] = 1.0

        # Update position
        current_position += interval

    # Normalize signal
    signal = signal / max(np.abs(signal))
    return signal

def save_to_wav(filename, signal, sample_rate=44100):
    """Save the signal as a .wav file."""
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        signal = (signal * 32767).astype(np.int16)  # Scale to 16-bit range
        wav_file.writeframes(signal.tobytes())

def main():
    print("Bouncing Pulse Generator")
    print("=========================")
    length = float(input("Enter the length of the output (seconds): "))
    initial_interval = float(input("Enter the initial interval between bounces (seconds): "))
    #decay_factor = float(input("Enter the decay factor (must be < 1): "))
    pulse_duration = float(input("Enter the duration of each pulse (seconds): "))
    filename = input("Enter output file name (e.g., bouncing_pulse.wav): ")

    # Generate the bouncing pulse
    print("Generating bouncing pulse...")
    signal = generate_bouncing_pulse(
        length=length,
        sample_rate=44100,
        initial_interval=initial_interval,
        #decay_factor=decay_factor,
        pulse_duration=pulse_duration
    )

    # Save to .wav file
    save_to_wav(filename, signal)
    print(f"Bouncing pulse saved to {filename}")

if __name__ == "__main__":
    main()

