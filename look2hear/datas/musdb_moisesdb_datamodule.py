import os
import h5py
import numpy as np
from typing import Any, Tuple
import torch
import random
from pytorch_lightning import LightningDataModule
import torchaudio
from torchaudio.functional import apply_codec
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Optional, Tuple

def compute_mch_rms_dB(mch_wav, fs=16000, energy_thresh=-50):
    """Return the wav RMS calculated only in the active portions"""
    mean_square = max(1e-20, torch.mean(mch_wav ** 2))
    return 10 * np.log10(mean_square)

def match2(x, d):
    assert x.dim()==2, x.shape
    assert d.dim()==2, d.shape
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[:,0:minlen], d[:,0:minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd*Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:,0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(),dim=-1).tolist()
    return tau

def codec_simu(wav, sr=16000, options={'bitrate':'random','compression':'random', 'complexity':'random', 'vbr':'random'}):

    if options['bitrate'] == 'random':
        options['bitrate'] = random.choice([24000, 32000, 48000, 64000, 96000, 128000])
    compression = int(options['bitrate']//1000)
    param = {'format': "mp3", "compression": compression}
    wav_encdec = apply_codec(wav, sr, **param)
    if wav_encdec.shape[-1] >= wav.shape[-1]:
        wav_encdec = wav_encdec[...,:wav.shape[-1]]
    else:
        wav_encdec = torch.cat([wav_encdec, wav[..., wav_encdec.shape[-1]:]], -1)
    tau = match2(wav, wav_encdec) 
    wav_encdec = torch.roll(wav_encdec, -tau[0], -1)

    return wav_encdec

def get_wav_files(root_dir):
    wav_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.wav'):
                if "musdb18hq" in dirpath and "mixture" not in filename:
                    wav_files.append(os.path.join(dirpath, filename))
                elif "moisesdb" in dirpath:
                    wav_files.append(os.path.join(dirpath, filename))
    return wav_files

class MusdbMoisesdbDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        codec_type: str,
        codec_options: dict,
        sr: int = 16000,
        segments: int = 10,
        num_stems: int = 4,
        snr_range: Tuple[int, int] = (-10, 10),
        num_samples: int = 1000,
        # New arguments
        split: str = "train",
        seed: int | None = None,
    ) -> None:
        
        self.data_dir = data_dir
        self.codec_type = codec_type
        self.codec_options = codec_options
        self.segments = int(segments * sr)
        self.sr = sr
        self.num_stems = num_stems
        self.snr_range = snr_range
        self.num_samples = num_samples
        self.split = split.lower()
        self.seed = seed
        # dedicated RNG for reproducibility
        self._rng = random.Random(seed)
        
        self.instruments = [
            "bass", 
            # "bowed_strings", 
            "drums", 
            # "guitar",
            "other", 
            # "other_keys", 
            # "other_plucked", 
            # "percussion", 
            # "piano", 
            "vocals", 
            # "wind"
        ]

        # Pre-index available h5 files per instrument (and mixture) so that splits are deterministic.
        # We assume directory layout: data_dir/<stem>/*.hdf5 and data_dir/mixture/*.hdf5
        self._files_by_instrument: dict[str, list[str]] = {}
        for stem in self.instruments + ["mixture"]:
            stem_dir = os.path.join(self.data_dir, stem)
            if not os.path.isdir(stem_dir):
                self._files_by_instrument[stem] = []
                continue
            files = [f for f in os.listdir(stem_dir) if f.endswith('.hdf5')]
            files.sort()  # deterministic ordering
            if not files:
                self._files_by_instrument[stem] = []
                continue
            # Simple 80/10/10 split
            n = len(files)
            train_end = max(1, int(0.8 * n))
            val_end = max(train_end + 1, int(0.9 * n)) if n >= 3 else n
            if self.split in ("train", "training"):
                subset = files[:train_end]
            elif self.split in ("val", "valid", "validation"):
                subset = files[train_end:val_end]
            elif self.split in ("test", "eval", "evaluation"):
                subset = files[val_end:]
            else:
                # Unknown split label => use all
                subset = files
            # Fallback if subset empty (e.g., tiny dataset)
            if not subset:
                subset = files
            self._files_by_instrument[stem] = subset

    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._rng.random() > 0.5:
            select_stems = self._rng.randint(1, self.num_stems)
            select_stems = [self._rng.choice(self.instruments) for _ in range(select_stems)]
            ori_wav = []
            for stem in select_stems:
                candidates = self._files_by_instrument.get(stem, [])
                if not candidates:  # dynamic fallback
                    candidates = [f for f in os.listdir(os.path.join(self.data_dir, stem)) if f.endswith('.hdf5')]
                h5path = self._rng.choice(candidates)
                datas = h5py.File(os.path.join(self.data_dir, stem, h5path), 'r')['data']
                random_index = self._rng.randint(0, datas.shape[0]-1)
                music_wav = torch.FloatTensor(datas[random_index])
                start = self._rng.randint(0, music_wav.shape[-1] - self.segments)
                music_wav = music_wav[:, start:start+self.segments]
                
                rescale_snr = self._rng.randint(self.snr_range[0], self.snr_range[1])
                music_wav = music_wav * np.sqrt(10**(rescale_snr/10))
                ori_wav.append(music_wav)
            ori_wav = torch.stack(ori_wav).sum(0)
        else:
            candidates = self._files_by_instrument.get("mixture", [])
            if not candidates:
                candidates = [f for f in os.listdir(os.path.join(self.data_dir, "mixture")) if f.endswith('.hdf5')]
            h5path = self._rng.choice(candidates)
            datas = h5py.File(os.path.join(self.data_dir, "mixture", h5path), 'r')['data']
            random_index = self._rng.randint(0, datas.shape[0]-1)
            music_wav = torch.FloatTensor(datas[random_index])
            start = self._rng.randint(0, music_wav.shape[-1] - self.segments)
            ori_wav = music_wav[:, start:start+self.segments]
        
        codec_wav = codec_simu(ori_wav, sr=self.sr, options=self.codec_options)
        
        max_scale = max(ori_wav.abs().max(), codec_wav.abs().max())
        
        if max_scale > 0:
            ori_wav = ori_wav / max_scale
            codec_wav = codec_wav / max_scale
            
        return ori_wav, codec_wav
    

class MusdbMoisesdbEval(Dataset):
    def __init__(
        self,
        data_dir: str
    ) -> None:
        self.data_path = os.listdir(data_dir)
        self.data_path = [os.path.join(data_dir, i) for i in self.data_path]
        
    def __len__(self) -> int:
        return len(self.data_path)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_wav = torchaudio.load(self.data_path[idx]+"/ori_wav.wav")[0]
        codec_wav = torchaudio.load(self.data_path[idx]+"/codec_wav.wav")[0]
        
        return ori_wav, codec_wav, self.data_path[idx]
    
class MusdbMoisesdbDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        eval_dir: str,
        codec_type: str,
        codec_options: dict,
        sr: int = 16000,
        segments: int = 10,
        num_stems: int = 4,
        snr_range: Tuple[int, int] = (-10, 10),
        num_samples: int = 1000,
        batch_size: int = 32,
        num_workers: int = 4,
        # New: allow using training data as validation when no eval set is provided
        val_from_train: bool = True,
        val_num_samples: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = MusdbMoisesdbDataset(
                data_dir=self.hparams.train_dir,
                codec_type=self.hparams.codec_type,
                codec_options=self.hparams.codec_options,
                sr=self.hparams.sr,
                segments=self.hparams.segments,
                num_stems=self.hparams.num_stems,
                snr_range=self.hparams.snr_range,
                num_samples=self.hparams.num_samples,
                split="train",   # <---- important
                seed=42,
            )

            # Determine if we have a valid eval directory
            eval_dir = getattr(self.hparams, "eval_dir", None)
            has_eval = False
            if isinstance(eval_dir, str) and os.path.isdir(eval_dir):
                try:
                    # any() short-circuits on first entry
                    has_eval = any(os.scandir(eval_dir))
                except Exception:
                    has_eval = False

            if has_eval:
                self.data_val = MusdbMoisesdbEval(
                    data_dir=eval_dir
                )
            else:
                # Fallback: build a small validation set from the training data
                if getattr(self.hparams, "val_from_train", True):
                    self.data_val = MusdbMoisesdbDataset(
                        data_dir=self.hparams.train_dir,
                        codec_type=self.hparams.codec_type,
                        codec_options=self.hparams.codec_options,
                        sr=self.hparams.sr,
                        segments=self.hparams.segments,
                        num_stems=self.hparams.num_stems,
                        snr_range=self.hparams.snr_range,
                        num_samples=getattr(self.hparams, "val_num_samples", 16),
                        split="val",   # <---- important
                        seed=43,
                    )
                else:
                    # As a last resort, reuse the full training dataset (not recommended but ensures validation runs)
                    self.data_val = self.data_train
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
