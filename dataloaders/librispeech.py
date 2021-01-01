import torchaudio
import torch
import torch.nn as nn
import argparse


from torch.utils.data import Dataset, DataLoader, random_split


import pytorch_lightning as pl
from torchaudio.transforms import TimeMasking


class LibriSpeechDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, download_path, train_url, test_url):
        self.batch_size = batch_size
        self.download_path = download_path
        self.train_url = train_url
        self.test_url = test_url

        self.train_transforms = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        )

        self.val_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128)

        self.test_transforms = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=128)

        # TODO
        # text(lable) transform

    def prepare_date(self):
        # Download LIBRISPEECH
        torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url, download=True)

        torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url, download=True)

    def setup(self, stage=None):
        full_dataset = torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url)

        self.test_dataset = torchaudio.datasets.LIBRISPEECH(
            self.download_path, url=self.train_url)

        print(full_dataset)

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        print(train_size, test_size)

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, test_size], generator=torch.Generator())

        return full_dataset

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def main():

    pl.seed_everything(777)
    parser = argparse.ArgumentParser(description="LIBRISPEECH")
    parser.add_argument('--train-url', type=str, default="train-clean-100")
    parser.add_argument('--test-url', type=str, default="test-clean")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--download-path', type=str,
                        default="./data/librispeech")
    args = parser.parse_args()

    dm = LibriSpeechDataModule(args.batch_size, args.download_path, args.train_url,
                               args.test_url)
    dm.setup()

    train = dm.train_dataloader()

    print(train)

    for a in train:
        print(a)
        break


if __name__ == "__main__":
    main()
