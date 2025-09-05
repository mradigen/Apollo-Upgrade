import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
import h5py
from multiprocessing import Pool, cpu_count


def VAD(x, sr=44100, win=441, session_length=6):
    x_len = x.shape[0] // sr * sr
    x = x[:x_len]
    x_p = x.reshape(-1, win, 2)
    x_pow = np.sum(np.power(x_p, 2), (1, 2)).reshape(-1,)
    x_valid_pow = x_pow[x_pow > 1e-3]

    if len(x_valid_pow) == 0:
        return []

    threshold = np.quantile(x_valid_pow, 0.15)
    valid_segment = []
    num_session = (x_len - sr * session_length) // (sr * session_length // 2)

    for s in range(num_session):
        this_segment = x[s * sr * session_length // 2:
                         s * sr * session_length // 2 + sr * session_length, :]
        this_segment_pow = this_segment.reshape(-1, win, 2)
        this_segment_pow = np.sum(np.power(this_segment_pow, 2), (1,2)).reshape(-1,)
        pow_ratio = np.sum(this_segment_pow > threshold) / len(this_segment_pow)
        if pow_ratio > 0.5:
            valid_segment.append(this_segment)

    return valid_segment


def process_track(args):
    """Process a single track into HDF5 shards."""
    track, files, output_dir, sr, session_length, max_samples_per_file = args

    track_dir = os.path.join(output_dir, track)
    os.makedirs(track_dir, exist_ok=True)

    cnt = len(os.listdir(track_dir)) + 1
    buffer = []

    # Inner progress bar (files inside a track)
    for file in tqdm(files, desc=f"{track}", position=1, leave=False):
        x, sr = sf.read(file)
        x = VAD(x, sr=sr, session_length=session_length)
        if len(x) == 0:
            continue

        for seg in x:
            buffer.append(seg.T)  # shape (2, T)

            if len(buffer) >= max_samples_per_file:
                dataset_name = os.path.join(track_dir, f"sample{cnt}.hdf5")
                with h5py.File(dataset_name, 'w') as dataset:
                    dataset.create_dataset(
                        'data',
                        data=np.asarray(buffer, dtype=np.float32),
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=4
                    )
                buffer = []
                cnt += 1

    # flush leftovers
    if buffer:
        dataset_name = os.path.join(track_dir, f"sample{cnt}.hdf5")
        with h5py.File(dataset_name, 'w') as dataset:
            dataset.create_dataset(
                'data',
                data=np.asarray(buffer, dtype=np.float32),
                dtype=np.float32,
                compression="gzip",
                compression_opts=4
            )


def musdb_preprocess(train_root_dir, test_root_dir, output_dir,
                     sr=44100, session_length=6, max_samples_per_file=100):
    all_data_dir = {}
    for root, _, files in os.walk(train_root_dir):
        for file in files:
            if file.endswith(".wav"):
                all_data_dir.setdefault(file.split(".")[0], []).append(os.path.join(root, file))
    for root, _, files in os.walk(test_root_dir):
        for file in files:
            if file.endswith(".wav"):
                all_data_dir.setdefault(file.split(".")[0], []).append(os.path.join(root, file))

    print(f"Found {len(all_data_dir)} tracks.")

    args_list = [
        (track, files, output_dir, sr, session_length, max_samples_per_file)
        for track, files in all_data_dir.items()
    ]

    n_workers = cpu_count()  # Kaggle usually gives 2–4 CPUs
    print(f"Using {n_workers} CPU workers...")

    # Outer progress bar (tracks)
    with Pool(n_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_track, args_list),
                      total=len(args_list), desc="Tracks", position=0):
            pass


if __name__ == "__main__":
    train_root_dir = "/tmp/prism/musdb18-hq/train"
    test_root_dir = "/tmp/prism/musdb18-hq/test"
    output_dir = "/tmp/prism/Apollo-Upgrade/hdf5_datas_new"
    musdb_preprocess(train_root_dir, test_root_dir, output_dir,
                     sr=44100, session_length=6, max_samples_per_file=100)
