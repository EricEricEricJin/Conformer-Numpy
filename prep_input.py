
import librosa
import kaldi_native_fbank as knf
import numpy as np

"""
Compute FilterBank
similar to spectrum but been processed
"""
def compute_feat(filename):
    sample_rate = 16000
    samples, _ = librosa.load(filename, sr=sample_rate)
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80

    online_fbank = knf.OnlineFbank(opts)
    online_fbank.accept_waveform(sample_rate, (samples * 32768).tolist())
    online_fbank.input_finished()

    features = np.stack(
        [online_fbank.get_frame(i) for i in range(online_fbank.num_frames_ready)]
    )
    assert features.data.contiguous is True
    assert features.dtype == np.float32, features.dtype
    mean = features.mean(axis=0, keepdims=True)
    stddev = features.std(axis=0, keepdims=True)
    features = (features - mean) / (stddev + 1e-5)
    return features

def load_tokens(filename):
    ans = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            sym, idx = line.strip().split()
            ans[int(idx)] = sym
    return ans


if __name__ == "__main__":
    feat = compute_feat("test_wavs/0.wav")
    print(feat.shape)
    