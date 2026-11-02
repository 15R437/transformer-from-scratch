import torch
import numpy as np
from transformer import Transformer
from data import get_batch

N_CTX = 64

def train(model : Transformer, data_set : str, batch_size : int, n_ctx : int, n_epochs : int, lr : float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_history = []
    val_history = []

    for epoch in range(n_epochs):
        model.train()
        input, target = get_batch(data_set, batch_size, n_ctx, "train")
        optimizer.zero_grad()
        logits, _ = model(input)
        train_loss = loss_fn(logits.view(-1, model.d_vocab), target.view(-1))
        train_history.append(train_loss.item())
        
        with torch.no_grad():
            val_input, val_target = get_batch(data_set, batch_size, n_ctx, "val")
            val_loss = loss_fn(model(val_input)[0].view(-1, model.d_vocab), val_target.view(-1))
            val_history.append(val_loss.item())

        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

    return train_history, val_history

if __name__ == "__main__":
    model = Transformer(layer_configs=[{'name': 'attn', 'n_heads': 2, 'd_head': 64, 'd_model': 128}, {'name': 'mlp', 'd_mlp': 256, 'd_model': 128, 'bias': True}], n_ctx=N_CTX, d_vocab=50257, d_model=128)
    train(model, data_set="tiny_shakespeare", batch_size=32, n_ctx=N_CTX, n_epochs=100, lr=0.001,
          plot=True, plot_path="analysis/loss.png")