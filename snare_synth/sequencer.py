import numpy as np
import wave
import random
from snare_synth import generate_snare, save_to_wav

def generate_sequence(bpm, length, pattern, sample_rate=44100, snare_algorithms=None):
    """
    Generate a sequence of snares with random parameters on each step.
    Parameters:
    - bpm: Beats per minute of the sequence.
    - length: Length of the output in seconds.
    - pattern: String pattern (e.g., "000101001") where '1' triggers a snare and '0' is silence.
    - sample_rate: Sample rate in Hz.
    - snare_algorithms: List of noise algorithms to randomly select from.
    Returns:
    - signal: The generated sequence as a numpy array.
    """
    if snare_algorithms is None:
        from noise_algorithms import white_noise, pink_noise, brown_noise
        snare_algorithms = [white_noise, pink_noise, brown_noise]

    # Calculate duration of one beat in seconds
    beat_duration = 60 / bpm
    num_beats = int(length / beat_duration)

    # Repeat the pattern to fill the sequence
    extended_pattern = (pattern * (num_beats // len(pattern) + 1))[:num_beats]

    # Initialize an empty signal
    total_samples = int(length * sample_rate)
    signal = np.zeros(total_samples)

    # Generate sequence with random snare parameters
    for i, char in enumerate(extended_pattern):
        if char == "1":
            # Calculate maximum allowable duration for attack and decay
            max_duration = beat_duration

            # Ensure attack and decay fit within the beat duration
            random_attack = random.uniform(0.005, max(0.005, max_duration * 0.4))  # Attack up to 40% of beat
            random_decay = random.uniform(0.005, max(0.005, max_duration - random_attack))  # Remaining time for decay

            # Randomize noise algorithm
            random_algorithm = random.choice(snare_algorithms)

            # Generate random snare
            snare = generate_snare(
                sample_rate=sample_rate,
                length=beat_duration,
                algorithm=random_algorithm,
                attack=random_attack,
                decay=random_decay
            )

            # Place snare in the sequence
            start = int(i * beat_duration * sample_rate)
            end = start + len(snare)
            signal[start:end] += snare[: min(len(signal[start:end]), len(snare))]

    # Normalize the final signal to [-1, 1]
    signal = signal / max(np.abs(signal))
    return signal



def main():
    print("Randomized Snare Sequencer")
    print("===========================")
    bpm = int(input("Enter BPM: "))
    length = float(input("Enter length of the sequence (seconds): "))
    pattern = input("Enter pattern (e.g., '000101001'): ")
    filename = input("Enter output file name (e.g., randomized_sequence.wav): ")

    # Generate the sequence
    print("Generating sequence with randomized snares...")
    signal = generate_sequence(bpm, length, pattern)

    # Save to .wav file
    save_to_wav(filename, signal)
    print(f"Sequence saved to {filename}")

if __name__ == "__main__":
    main()

