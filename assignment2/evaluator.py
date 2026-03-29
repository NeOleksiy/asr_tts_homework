import pandas as pd
import torchaudio
import jiwer
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from wav2vec2decoder import Wav2Vec2Decoder


def load_samples(file_path):
    df = pd.read_csv(file_path)
    return list(zip(df['path'], df['text']))

samples_dict = {
    "earnings22": load_samples("data/earnings22_test/manifest.csv"),
    "librispeech": load_samples("data/librispeech_test_other/manifest.csv")
}

temperatures = [0.5, 0.7, 1.0, 2, 3]
methods = ["greedy", "beam", "beam_lm", "beam_lm_rescore"]
metrics = ["wer", "cer"]


plot_data = {m: {met: {s: [] for s in samples_dict} for met in metrics} for m in methods}

decoder = Wav2Vec2Decoder()


print("Starting evaluation...")
for temp in tqdm(temperatures):
    print(f"  Testing temperature: {temp}")
    decoder.temperature = temp
    
    for s_name, data in samples_dict.items():
        temp_metrics = {m: {met: [] for met in metrics} for m in methods}
        
        for audio_path, ref in tqdm(data):
            try:
                audio, sr = torchaudio.load(audio_path)
                for m in methods:
                    hyp = decoder.decode(audio, method=m)
                    temp_metrics[m]["wer"].append(jiwer.wer(ref, hyp))
                    temp_metrics[m]["cer"].append(jiwer.cer(ref, hyp))
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        for m in methods:
            for met in metrics:
                avg_val = sum(temp_metrics[m][met]) / len(temp_metrics[m][met]) if temp_metrics[m][met] else 0
                plot_data[m][met][s_name].append(avg_val)

print("Generating and saving plots...")
os.makedirs("plots", exist_ok=True)

for m in methods:
    for met in metrics:
        plt.figure(figsize=(8, 5))
        
        for s_name in samples_dict:
            plt.plot(temperatures, plot_data[m][met][s_name], marker='o', label=s_name)
        
        plt.title(f"Method: {m.upper()} | Metric: {met.upper()}")
        plt.xlabel("Temperature")
        plt.ylabel(met.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        filename = f"plots/{m}_{met}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  Saved: {filename}")

print("\nDone! All plots are in the 'plots/' folder.")