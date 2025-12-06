from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
from omegaconf import OmegaConf

from NNLM import FFNModel, RNNModel, SelfAttentionNNLM
from data.get_loader import get_loader, word_to_ix, ix_to_word, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_NNLM(
    model: nn.Module,
    test_loader: DataLoader,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Use the last time step's output for classification
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    logger.info(f"Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")


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

    # Use the test set for evaluation and disable shuffling for deterministic perplexity
    test_loader = get_loader(
        data_path="data/test.txt",
        context_size=config.context_size,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # FFNModel
    logger.info("Testing FFNModel...")
    model = FFNModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
        config.context_size,
    ).to(device)
    for i in range(config.epochs):
        model_path = Path(f"ckpts/FFNModel/epoch{i+1}.pth")
        model.load_state_dict(torch.load(model_path))
        test_NNLM(model, test_loader)

    # RNNModel
    logger.info("Testing RNNModel...")
    model = RNNModel(
        vocab_size,
        config.embed_size,
        config.hidden_size,
    ).to(device)
    for i in range(config.epochs):
        model_path = Path(f"ckpts/RNNModel/epoch{i+1}.pth")
        model.load_state_dict(torch.load(model_path))
        test_NNLM(model, test_loader)

    # SelfAttentionNNLM
    logger.info("Testing SelfAttentionNNLM...")
    model = SelfAttentionNNLM(
        vocab_size,
        config.embed_size,
        config.hidden_size,
    ).to(device)
    for i in range(config.epochs):
        model_path = Path(f"ckpts/SelfAttentionNNLM/epoch{i+1}.pth")
        model.load_state_dict(torch.load(model_path))
        test_NNLM(model, test_loader)

if __name__ == "__main__":
    main()