import argparse
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
import torch

try:
    ort.preload_dlls() 
except Exception:
    pass

class CMGANInference:
    def __init__(self, onnx_path: str, sr: int = 16000, n_fft: int = 400, hop: int = 100):

        available_providers = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.window = torch.hamming_window(self.n_fft)
        
        self.max_chunk_samples = 4 * self.sr 
        
    @staticmethod
    def power_compress(x):
        spec = torch.complex(x[..., 0], x[..., 1])
        mag = torch.abs(spec) ** 0.3
        phase = torch.angle(spec)
        return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)

    @staticmethod
    def power_uncompress(real, imag):
        spec = torch.complex(real, imag)
        mag = torch.abs(spec) ** (1.0 / 0.3)
        phase = torch.angle(spec)
        return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Processes a single small chunk of audio."""
        noisy_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
        
        c = torch.sqrt(noisy_tensor.size(-1) / (torch.sum((noisy_tensor ** 2.0), dim=-1, keepdim=True) + 1e-8))
        noisy_norm = noisy_tensor * c

        noisy_spec = torch.stft(noisy_norm, self.n_fft, self.hop, window=self.window, onesided=True, return_complex=True)
        noisy_spec = torch.view_as_real(noisy_spec)
        noisy_spec = self.power_compress(noisy_spec)
        
        noisy_spec_np = noisy_spec.permute(0, 3, 2, 1).numpy()

        est_real, est_imag = self.session.run(None, {self.input_name: noisy_spec_np})

        est_real = torch.from_numpy(est_real)
        est_imag = torch.from_numpy(est_imag)

        est_spec_uncompress = self.power_uncompress(est_real, est_imag).squeeze(1)
        est_spec_complex = torch.complex(est_spec_uncompress[..., 0], est_spec_uncompress[..., 1])

        est_audio = torch.istft(est_spec_complex.transpose(1, 2), self.n_fft, self.hop, window=self.window, onesided=True)
        return (est_audio / c).squeeze(0).numpy()

    def process_file(self, input_path: str, output_path: str):
        audio, orig_sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sr)

        total_samples = audio.shape[0]
        enhanced_chunks = []
        
        print(f"Processing {total_samples / self.sr:.2f} seconds of audio in chunks...")
        
        for start in range(0, total_samples, self.max_chunk_samples):
            end = min(start + self.max_chunk_samples, total_samples)
            chunk = audio[start:end]
            
            if len(chunk) < self.n_fft:
                continue
                
            enhanced_chunk = self._process_chunk(chunk)
            enhanced_chunks.append(enhanced_chunk)

        enhanced_audio = np.concatenate(enhanced_chunks, axis=-1)

        peaks = np.max(np.abs(enhanced_audio))
        if peaks > 1.0:
            enhanced_audio /= peaks

        sf.write(output_path, enhanced_audio, self.sr)
        print(f"Successfully enhanced and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Audio Enhancement using ONNX")
    parser.add_argument("-m", "--model", required=True, help="Path to your .onnx model")
    parser.add_argument("-i", "--input", required=True, help="Path to noisy input audio")
    parser.add_argument("-o", "--output", required=True, help="Path to save enhanced audio")
    args = parser.parse_args()

    enhancer = CMGANInference(onnx_path=args.model)
    enhancer.process_file(args.input, args.output)
