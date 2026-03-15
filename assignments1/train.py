import os
import json
import sys
import time
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from melbanks import LogMelFilterBanks


class BinarySpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root: str, subset: str = 'training'):
        os.makedirs(root, exist_ok=True)
        super().__init__(root, download=True, subset=subset)

        self.classes = ['yes', 'no']
        self.class_to_idx = {'yes': 0, 'no': 1}

        self.filtered_indices = [
            idx for idx, path in enumerate(self._walker)
            if os.path.basename(os.path.dirname(path)) in self.classes
        ]

        print(f"[{subset}] загружено {len(self.filtered_indices)} примеров")

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        waveform, sr, label, _, _ = super().__getitem__(self.filtered_indices[idx])
        return waveform.squeeze(0), self.class_to_idx[label]


class PadCollate:
    def __init__(self, max_length: int = 16000):
        self.max_length = max_length

    def __call__(self, batch):
        waveforms, labels = zip(*batch)
        padded = []
        for w in waveforms:
            if w.size(0) < self.max_length:
                w = torch.nn.functional.pad(w, (0, self.max_length - w.size(0)))
            else:
                w = w[:self.max_length]
            padded.append(w)
        return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


class SpeechCommandsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 64, max_length: int = 16000):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.collate_fn = PadCollate(max_length=max_length)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = BinarySpeechCommands(self.data_dir, 'training')
        self.val_dataset = BinarySpeechCommands(self.data_dir, 'validation')
        self.test_dataset = BinarySpeechCommands(self.data_dir, 'testing')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

class SimpleCNN(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.logmel = LogMelFilterBanks(n_mels=n_mels)
        def safe_groups(ch, g):
            return g if ch % g == 0 else 1

        g1 = safe_groups(n_mels, groups)
        g2 = safe_groups(32, groups)
        g3 = safe_groups(64, groups)

        self.conv1 = self._conv_block(n_mels, 32, g1, dropout)
        self.conv2 = self._conv_block(32, 64, g2, dropout)
        self.conv3 = self._conv_block(64, 128, g3, dropout)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 2)
        )

        self._init_weights()

    def _conv_block(self, in_ch, out_ch, groups, dropout):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.logmel(x)      # (batch, n_mels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_len: int = 16000) -> int:
        total = 0
        n_fft = 512
        total += 5 * n_fft * input_len
        total += (input_len // 2) * self.logmel.n_mels
        total += (input_len // 2) * self.logmel.n_mels

        length = input_len // 2
        conv_layers = [
            (self.logmel.n_mels, 32, self.conv1[0].groups),
            (32, 64, self.conv2[0].groups),
            (64, 128, self.conv3[0].groups)
        ]
        for in_ch, out_ch, g in conv_layers:
            total += 2 * length * in_ch * out_ch * 3 // g
            length //= 2

        total += 128
        total += 2 * 128 * 2
        return total


class SimpleCNNLightning(pl.LightningModule):
    def __init__(self, n_mels: int = 80, groups: int = 1, dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = SimpleCNN(n_mels=n_mels, groups=groups, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()

        print(f"Создана модель: параметров = {self.model.count_parameters():,}, "
              f"FLOPs ≈ {self.model.count_flops():,}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return acc

    def configure_optimizers(self):
        return optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


def plot_results(results: List[Dict[str, Any]], output_dir: str):
    output_dir = "results"
    results.sort(key=lambda x: (x['n_mels'], x['groups']))

    exp_nmels = [r for r in results if r['groups'] == 1]
    if exp_nmels:
        plt.figure()
        n_mels_vals = [r['n_mels'] for r in exp_nmels]
        acc_vals = [r['test_accuracy'] for r in exp_nmels]
        plt.plot(n_mels_vals, acc_vals, marker='o')
        plt.xlabel('Number of Mel filters')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs n_mels (groups=1)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'n_mels_vs_accuracy.png'), dpi=150)
        plt.close()

    exp_groups = [r for r in results if r['n_mels'] == 80]
    if exp_groups:
        plt.figure()
        groups_vals = [r['groups'] for r in exp_groups]
        acc_vals = [r['test_accuracy'] for r in exp_groups]
        plt.plot(groups_vals, acc_vals, marker='s')
        plt.xlabel('Groups')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs groups (n_mels=80)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'groups_vs_accuracy.png'), dpi=150)
        plt.close()

    if exp_groups:
        plt.figure()
        groups_vals = [r['groups'] for r in exp_groups]
        time_vals = [r['avg_epoch_time'] for r in exp_groups]
        plt.plot(groups_vals, time_vals, marker='^', color='green')
        plt.xlabel('Groups')
        plt.ylabel('Avg Epoch Time (s)')
        plt.title('Training Time per Epoch vs groups (n_mels=80)')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'groups_vs_time.png'), dpi=150)
        plt.close()

    if exp_groups:
        fig, ax1 = plt.subplots()
        groups_vals = [r['groups'] for r in exp_groups]
        params_vals = [r['parameters'] / 1e3 for r in exp_groups]
        flops_vals = [r['flops'] / 1e6 for r in exp_groups]

        color = 'tab:red'
        ax1.set_xlabel('Groups')
        ax1.set_ylabel('Parameters (thousands)', color=color)
        ax1.plot(groups_vals, params_vals, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('FLOPs (millions)', color=color)
        ax2.plot(groups_vals, flops_vals, marker='s', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Parameters and FLOPs vs groups (n_mels=80)')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'groups_vs_params_flops.png'), dpi=150)
        plt.close()

    print(f"Графики сохранены в {output_dir}")


def main():
    DATA_DIR = './speech_commands'
    OUTPUT_DIR = './lightning_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pl.seed_everything(42, workers=True)

    experiments = [
        {'n_mels': n, 'groups': 1, 'epochs': 5, 'batch_size': 64}
        for n in [20, 40, 80]
    ] + [
        {'n_mels': 80, 'groups': g, 'epochs': 5, 'batch_size': 64}
        for g in [2, 4, 8, 16]
    ]

    all_results = []

    for exp in experiments:
        print("\n" + "="*70)
        print(f"Запуск эксперимента: n_mels={exp['n_mels']}, groups={exp['groups']}")
        print("="*70)

        dm = SpeechCommandsDataModule(
            data_dir=DATA_DIR,
            batch_size=exp['batch_size'],
            max_length=16000
        )

        model = SimpleCNNLightning(
            n_mels=exp['n_mels'],
            groups=exp['groups'],
            lr=1e-3,
            weight_decay=1e-4
        )

        logger = CSVLogger(save_dir=OUTPUT_DIR, name=f"exp_m{exp['n_mels']}_g{exp['groups']}")

        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            mode='max',
            save_top_k=1,
            filename='best-{epoch:02d}-{val_acc:.4f}'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            patience=3,
            mode='max'
        )

        trainer = Trainer(
            max_epochs=exp['epochs'],
            accelerator='auto',
            devices=1,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            deterministic=True,
            log_every_n_steps=10
        )

        start_time = time.time()
        trainer.fit(model, datamodule=dm)
        total_time = time.time() - start_time
        actual_epochs = trainer.current_epoch + 1
        avg_epoch_time = total_time / actual_epochs

        test_result = trainer.test(model, datamodule=dm, ckpt_path='best')
        test_acc = test_result[0]['test_acc']

        result = {
            'n_mels': exp['n_mels'],
            'groups': exp['groups'],
            'epochs': exp['epochs'],
            'actual_epochs': actual_epochs,
            'batch_size': exp['batch_size'],
            'parameters': model.model.count_parameters(),
            'flops': model.model.count_flops(),
            'test_accuracy': test_acc,
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
        }
        all_results.append(result)

        print(f"✓ Результат: test_acc={test_acc:.4f}, params={result['parameters']:,}, "
              f"avg_epoch_time={avg_epoch_time:.2f}с")

    with open(os.path.join(OUTPUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    plot_results(all_results, OUTPUT_DIR)

    print("\n" + "="*70)
    print("СВОДКА ПО ЭКСПЕРИМЕНТАМ")
    print("="*70)
    for r in all_results:
        print(f"n_mels={r['n_mels']:2d}, groups={r['groups']:2d} → "
              f"test_acc={r['test_accuracy']:.4f}, params={r['parameters']:8,}, "
              f"flops={r['flops']:10,}, time/epoch={r['avg_epoch_time']:.2f}с")


if __name__ == '__main__':
    main()