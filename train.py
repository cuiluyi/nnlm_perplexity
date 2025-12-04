from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from omegaconf import OmegaConf

from NNLM import FFNModel, RNNModel, SelfAttentionNNLM
from data.get_loader import get_loader, word_to_ix, ix_to_word, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_NNLM(
    config: dict,
    model: nn.Module,
    train_loader: DataLoader,
):
    model.train()
    n_iterations = len(train_loader)

    # prepare model, loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    # training loop
    for epoch in range(config.epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # Use the last time step's output for classification
            loss = criterion(outputs[:, -1, :], labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config.logging_steps == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}"
                )


def main():
    parser = ArgumentParser(description="Train Word Vectors")
    parser.add_argument(
        "--config",
        type=str,
        default="config/demo.yaml",
        help="Path to the training configuration YAML file",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = FFNModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
        config.context_size,
    ).to(device)

    train_loader = get_loader(
        data_path="data/train.txt",
        context_size=config.context_size,
        batch_size=config.batch_size,
    )
    train_NNLM(config, model, train_loader)
