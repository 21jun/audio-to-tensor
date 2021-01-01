from matplotlib.pyplot import specgram
from torch.utils import data
import torchaudio
import torch
import torch.nn as nn
import argparse
from sklearn.model_selection import train_test_split

import json

import numpy as np
import scipy.signal
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchaudio import transforms
from torchaudio.transforms import TimeMasking

from pathlib import Path


def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform


class KoSpeechDataSet(Dataset):
    def __init__(self, data_list, data_root, transform) -> None:
        """
        data_list : json
        """
        super().__init__()
        self.data_list = data_list
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        wav_path = Path(self.data_root)/Path(self.data_list[index]['wav'])
        text = self.data_list[index]['text']
        waveform = load_audio(wav_path)
        # transform
        specgram = self.transform(waveform)
        return {"wave": waveform,
                "spectrogram": specgram,
                "text": text}


class KoSpeechDataModule(pl.LightningDataModule):

    def __init__(self, data_root="../data/KoSpeech/sub1", data_list_url="../data/KoSpeech/KsponSpeech.json", batch_size=8):
        self.batch_size = batch_size
        self.data_root = data_root
        self.data_list_url = data_list_url

        self.data_list = []
        with open(self.data_list_url, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)

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

        # TODO: text(lable) transform

    def prepare_date(self):
        pass

    def setup(self, stage=None):

        self.data_list

        train_data_list, val_data_list = train_test_split(
            self.data_list, test_size=0.3)

        #TODO: test_data_list

        self.train_dataset = KoSpeechDataSet(
            train_data_list, self.data_root, self.train_transforms)
        self.val_dataset = KoSpeechDataSet(
            val_data_list, self.data_root, self.val_transforms)

    def collate_fn(self, batch):
        waves = [x['wave'] for x in batch]
        spectrograms = [x['spectrogram'] for x in batch]
        texts = [x['text'] for x in batch]
        return waves, spectrograms, texts

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)


def main():

    pl.seed_everything(777)
    parser = argparse.ArgumentParser(description="LIBRISPEECH")
    parser.add_argument('--train-url', type=str, default="train-clean-100")
    parser.add_argument('--test-url', type=str, default="test-clean")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--download-path', type=str,
                        default="./data/librispeech")
    args = parser.parse_args()

    dm = KoSpeechDataModule(args.batch_size)
    dm.setup()

    train = dm.train_dataloader()

    print(train)

    for a in train:
        print(a)
        break


if __name__ == "__main__":
    main()
