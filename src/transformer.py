import tiktoken
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Literal

class Attention(nn.Module):
    def __init__(self, n_heads: int, d_head : int, d_model : int, n_ctx: int, W_P: torch.Tensor):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.n_ctx = n_ctx

        self.W_P = W_P #positional embedding matrix passed from Transformer

        self.W_V = nn.Parameter(torch.randn(n_heads, d_head, d_model) * d_model ** -0.5)
        self.W_K = nn.Parameter(torch.randn(n_heads, d_head, d_model) * d_model ** -0.5)
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_head, d_model) * d_model ** -0.5)
        self.W_O = nn.Parameter(torch.randn(n_heads, d_model, d_head) * d_head ** -0.5)

        causal_mask = torch.triu(torch.ones(n_ctx, n_ctx, dtype=bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        """ 
        Shapes
            x : (batch, n_tokens, d_model)
            v : (n_heads, batch, n_tokens, d_head)
            k : (n_heads, batch, n_tokens, d_head)
            q : (n_heads, batch, n_tokens, d_head)
            A : (n_heads, batch, n_tokens, n_tokens)
            r : (n_heads, batch, n_tokens, d_head)
            output: (batch, n_tokens, d_model)
        
        A[h, b, i, j] has i as destination and j as source
        """
        n_tokens = x.shape[-2]
        if n_tokens > self.n_ctx:
            raise ValueError(f"Context length exceeded. Maximum of {self.n_ctx} tokens allowed. Received {n_tokens}.")
        v = torch.einsum("hkr,bjr -> hbjk", self.W_V, x)
        k = torch.einsum("hkr,bjr -> hbjk", self.W_K, x + self.W_P[:n_tokens, :])
        q = torch.einsum("hkr,bir -> hbik", self.W_Q, x + self.W_P[:n_tokens, :])

        A = torch.einsum("hbik,hbjk -> hbij", q, k) / self.d_head ** 0.5
        A = A.masked_fill(self.causal_mask[:n_tokens, :n_tokens], -torch.inf)
        A = torch.softmax(A, dim=-1)
        r = torch.einsum("hbij,hbjk -> hbik", A, v)
        output = torch.einsum("hrk,hbik -> bir", self.W_O, r)

        return output


class MLP(nn.Module):
    def __init__(self, d_mlp : int, d_model : int, bias=False):
        super().__init__()
        self.W_I = nn.Parameter(torch.randn(d_mlp, d_model) * d_model ** -0.5)
        self.W_O = nn.Parameter(torch.randn(d_model, d_mlp) * d_mlp ** -0.5)

        self.b_I = nn.Parameter(torch.randn(d_mlp)) if bias else 0
        self.b_O = nn.Parameter(torch.randn(d_model)) if bias else 0

    def forward(self, x):
        """
        x : (batch, n_tokens, d_model)
        """
        x = torch.einsum('mr,btr -> btm', self.W_I, x) + self.b_I
        x = F.gelu(x, approximate='tanh')
        x = torch.einsum('rm,btm -> btr', self.W_O, x) + self.b_O

        return x
    
class Transformer(nn.Module):
    def __init__(self, layer_configs : List[dict], 
                 n_ctx : int, d_vocab : int, d_model : int):
        super().__init__()
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.d_vocab = d_vocab

        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab))
        self.W_P = nn.Parameter(torch.randn(n_ctx, d_model))
        self.W_U = nn.Parameter(torch.randn(d_vocab, d_model) * d_model ** -0.5)

        self._build_layers(layer_configs)

    def _build_layers(self, layer_configs):
        layers = []
        for cfg in layer_configs:
            if cfg['name'] == 'attn':
                layers.append(Attention(n_heads = cfg['n_heads'], 
                                             d_head = cfg['d_head'],
                                             d_model = self.d_model,
                                             n_ctx = self.n_ctx,
                                             W_P = self.W_P))
            elif cfg['name'] == 'mlp':
                layers.append(MLP(d_mlp = cfg['d_mlp'], d_model = self.d_model, bias = cfg['bias']))

        self.layers = nn.ModuleList(layers)

    def forward(self, token_ids: torch.Tensor):
        """
        token_ids : (batch, n_tokens)
        logits : (batch, n_tokens, d_vocab)
        """
        # embedding
        x = self.W_E[:, token_ids]          # [d_model, batch, n_tokens]
        x = torch.einsum('rbj -> bjr', x) # [batch, n_tokens, d_model]
        activations = [x]

        # computation
        for layer in self.layers:
            x = x + layer(x)
            activations.append(x)

        # unembedding
        logits = torch.einsum('vr,bjr -> bjv', self.W_U, x)
        activations.append(logits)

        return logits , activations

    def fit(self, data):
        """
        data : ()
        """
        pass

if __name__== "__main__":
    encoding = tiktoken.get_encoding("gpt2")
    text = "Hey everyone! This is my first implementation of a transformer."
    token_ids = encoding.encode(text)
    print(f"Token IDs: {token_ids} \t Text: {[encoding.decode([id]) for id in token_ids]}")
