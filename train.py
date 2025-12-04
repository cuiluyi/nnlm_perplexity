from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from omegaconf import OmegaConf

from NNLM import FFNModel, RNNModel, SelfAttentionNNLM
from data.get_loader import get_loader, word_to_ix, ix_to_word, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_NNLM(config: dict, model: nn.Module, train_loader: DataLoader, save_dir: str | Path):
    model.train()
    n_iterations = len(train_loader)

    # prepare model, loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    # training loop
    for epoch in range(config.epochs):
        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % config.logging_steps == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{config.epochs}], Step [{i+1}/{n_iterations}], Loss: {loss.item():.4f}"
                )
        # save model checkpoint
        torch.save(model.state_dict(), save_dir / f"epoch{epoch+1}.pth")


def main():
    parser = ArgumentParser(description="Train Word Vectors")
    parser.add_argument(
        "--recipe",
        type=str,
        default="recipes/demo.yaml",
        help="Path to the training configuration YAML file",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.recipe)

    train_loader = get_loader(
        data_path="data/train.txt",
        context_size=config.context_size,
        batch_size=config.batch_size,
    )

    # FFNModel
    model = FFNModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
        config.context_size,
    ).to(device)
    save_dir = Path("ckpts/FFNModel/")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_NNLM(config, model, train_loader, save_dir)

    # RNNModel
    model = RNNModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
    ).to(device)
    save_dir = Path("ckpts/RNNModel/")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_NNLM(config, model, train_loader, save_dir)

    # SelfAttentionNNLM
    model = SelfAttentionNNLM(
        vocab_size,
        config.embed_size,
        config.hidden_size,
        config.context_size,
    ).to(device)
    save_dir = Path("ckpts/SelfAttentionNNLM/")
    save_dir.mkdir(parents=True, exist_ok=True)
    train_NNLM(config, model, train_loader, save_dir)


if __name__ == "__main__":
    main()