# Neural Network Language Model (NNLM) Comparison

This project compares the **perplexity performance** of three neural network language models:

- **Feedforward Neural Network (FFN)**
- **Recurrent Neural Network (RNN)**
- **Self-Attention Model** (Transformer-like architecture)

All models were trained and evaluated on the **same Chinese news dataset**, using consistent preprocessing, training, and testing pipelines.  
The goal is to analyze the strengths and weaknesses of different architectures in language modeling tasks.

---

## 📂 Dataset & Preprocessing

**Data Source:**
- `data/news.2017.zh.shuffled.deduped`

**Splitting:**
- Last 2000 lines → **Test set** (`data/test.txt`)
- Remaining lines → **Training set** (`data/train.txt`)

**Tokenization & Vocabulary:**
- Tool: [jieba](https://github.com/fxsjy/jieba)
- Low-frequency words filtered (`MIN_FREQ = 3`)
- Vocabulary includes all tokenized results (Chinese characters, punctuation, numbers, English, etc.)
- Saved mappings:  
  - `vocab` – word list after frequency filtering
  - `word_to_ix` – word → index mapping
  - `ix_to_word` – index → word mapping
  - `vocab_size` – for embedding and output layers

---

## 🛠 Training Setup

**Configuration (`demo.yaml`):**
```yaml
embed_size: 128
hidden_size: 256  # Self-Attention FFN middle layer default: 2048
context_size: 4
epochs: 30
learning_rate: 0.001
batch_size: 256
```

**Training process (`train.py`):**
- Function: `train_NNLM(config, model, train_loader, save_dir)`
- Iterate through all batches each epoch, record training loss.
- Save checkpoints: `ckpts/<ModelName>/epoch{N}.pth`
- Training logs saved to `logs/train_*.log`

**Testing (`test.py`):**
- Load test set
- For each model:
  - Evaluate all 30 epochs' checkpoints
  - Compute:
    - Average cross-entropy loss → `avg_loss`
    - Perplexity → `perplexity = exp(avg_loss)`
  - Output: `Test Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}`
- Test logs saved to `logs/test_*.log`

---

## 📖 Model Architectures

### Feedforward Neural Network (FFN)
- Input → Embedding → Flatten → Linear layers → Output layer

### Recurrent Neural Network (RNN)
- Embedding → `nn.RNN` → Output layer

### Self-Attention NNLM
- Embedding: `nn.Embedding(vocab_size, embed_size)`
- Multi-head Self-Attention (1 head, batch_first=True)
- Residual connection + LayerNorm
- Transformer-style Feedforward Network:
  - Linear(128→2048) → ReLU → Linear(2048→128)
- Residual connection + LayerNorm
- Output: `nn.Linear(embed_size, vocab_size)`
- Takes last time step’s output as prediction logits

---

## 📊 Results & Analysis

**Training Loss Decrease Speed:**  
FFN > RNN > Self-Attention

**Test Perplexity (Lower is Better):**  
Self-Attention > RNN > FFN

| Model         | Best Test Perplexity | Overfitting Epoch |
|---------------|----------------------|-------------------|
| FFN           | 263.0073             | After epoch 3     |
| RNN           | 246.0834             | After epoch 3     |
| Self-Attention| **195.5417**         | After epoch 7     |

---

## 📌 Notes & Tips
- Self-Attention captures long-range dependencies and positional interactions better, even with small `context_size`.
- Overfitting occurred earlier in FFN and RNN compared to Self-Attention.

---

## 🚀 Usage

### Training
```bash
sh train.sh
```

### Testing
```bash
sh test.sh
```

---

## 📷 Visuals

**Training Loss Trend:**
![Training Loss](https://tianchou.oss-cn-beijing.aliyuncs.com/img/20251206170631533.png)

**Test Perplexity Trend:**
![Test Perplexity](https://tianchou.oss-cn-beijing.aliyuncs.com/img/20251206170723472.png)

---

## 📜 License
This project is provided for research and educational purposes.

---

If you want, I can also **add GitHub badges and a polished header section** to make it look even more like a professional open-source README.  
Do you want me to add that?