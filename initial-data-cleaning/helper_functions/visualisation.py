from matplotlib import pyplot as plt

def plot_scans(freq, ampl, wire, res_freq=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq, ampl, color='blue', linewidth=1.5)

    if res_freq is not None:
        ax.axvline(x=res_freq, color='red', linestyle='--', label=f'Resonant Frequency: {res_freq:.2f} Hz')
        ax.legend()

    ax.set_title(f'Frequency vs Amplitude for wire: {wire}', fontsize=16)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
