import os
import torch
import torchaudio
import argparse
import look2hear.models
from thop import profile

def load_audio(file_path):
    audio, samplerate = torchaudio.load(file_path)
    return audio.unsqueeze(0).cuda()  # [1, 1, samples]

def save_audio(file_path, audio, samplerate=44100):
    audio = audio.squeeze(0).cpu()
    torchaudio.save(file_path, audio, samplerate)

def main(input_wav, output_wav):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model = look2hear.models.BaseModel.from_pretrain("/kaggle/working/Model", sr=44100, win=20, feature_dim=256, layer=6).cuda()
    test_data = load_audio(input_wav)
    
    print("Measuring now")
    
    dummy_input = torch.randn_like(test_data)
    
    macs, params = profile(model, inputs=(dummy_input,))
    gmacs = macs / 1e9
    
    print(f"Parameters: {params:,}")
    print(f"MACs: {macs:,}")
    print(f"GMACs: {gmacs:.3f}")
    
    # Run inference
    with torch.no_grad():
        out = model(test_data)
    save_audio(output_wav, out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--in_wav", type=str, required=True, help="Path to input wav file")
    parser.add_argument("--out_wav", type=str, required=True, help="Path to output wav file")
    args = parser.parse_args()

    main(args.in_wav, args.out_wav)
