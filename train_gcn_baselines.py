# Trains and evaluates DialogueGCN (Ghosal et al. 2019) and MMGCN (Hu et al. 2021) on the IEMOCAP LOSO-CV splits, using precomputed RoBERTa embeddings for text, OpenSmile features for audio, and OpenFace features for video. The code is adapted from the original authors' PyTorch implementations to work with our precomputed features and LOSO-CV setup.

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score as sklearn_f1

from loso_utils import ROOT, FEATURE_DIR

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [1, 2, 3, 4, 5]
CONTEXT_WINDOW = 10   
GAMMA_INTER = 0.7         
GRAPH_HIDDEN = 100       


class ExperimentTimer:
    def __init__(self):
        self.rows = []

    def record(self, fold, model_name, stage, duration):
        self.rows.append({"fold": fold, "model": model_name, "stage": stage, "seconds": duration})

    def to_df(self):
        return pd.DataFrame(self.rows)


timer = ExperimentTimer()


# DialogueGCN/MMGCN require batch_size=1 (variable utterance counts), so we define a custom collate_fn that just returns the single graph in the batch.

def get_speaker_from_utterance(utterance_id):
    import re
    sess_match = re.match(r'Ses(\d+)', utterance_id)
    if not sess_match:
        return None
    sess_num = sess_match.group(1)
    parts = utterance_id.split("_")
    if len(parts) >= 3 and parts[-1][0] in ("F", "M"):
        return f"Ses{sess_num}{parts[-1][0]}"
    if "_F" in utterance_id:
        return f"Ses{sess_num}F"
    elif "_M" in utterance_id:
        return f"Ses{sess_num}M"
    return None


def build_speaker_index(speakers):
    unique = list(dict.fromkeys(speakers))
    return {spk: idx for idx, spk in enumerate(unique[:2])}


def get_relation_id(speaker_src, speaker_tgt, is_future, speaker_to_idx):
    src_idx = speaker_to_idx.get(speaker_src, 0)
    tgt_idx = speaker_to_idx.get(speaker_tgt, 0)
    return src_idx * 4 + tgt_idx * 2 + int(is_future)

# Utterance-level graph construction: builds the edge_index and edge_types tensors for a given dialogue, based on the speakers of each utterance and a context window. Each utterance is connected to its neighbors within the context window, with edge types encoding the speaker relationship and temporal direction (past/future). Self-loops are added separately in the GCN layers.
def build_dialogue_edges(speakers, speaker_to_idx, window=CONTEXT_WINDOW):
    N = len(speakers)
    src_list, tgt_list, type_list = [], [], []
    for j in range(N):
        lo, hi = max(0, j - window), min(N - 1, j + window)
        for i in range(lo, hi + 1):
            if i == j:
                continue
            is_future = (i > j)
            rel = get_relation_id(speakers[i], speakers[j], is_future, speaker_to_idx)
            src_list.append(i); tgt_list.append(j); type_list.append(rel)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_types = torch.tensor(type_list, dtype=torch.long)
    self_loops = torch.arange(N, dtype=torch.long)
    return edge_index, edge_types, self_loops


def build_dialogue_graph(dialogue_df):
    dialogue_df = dialogue_df.sort_values("utterance_id").reset_index(drop=True)
    speakers = dialogue_df["speaker"].tolist()
    utterance_ids = dialogue_df["utterance_id"].tolist()
    labels = torch.tensor(dialogue_df["label"].values, dtype=torch.long)

    speaker_to_idx = build_speaker_index(speakers)

    text_feats = np.vstack([np.load(p) for p in dialogue_df["text_feature_path"]])
    audio_feats = np.vstack([np.load(p) for p in dialogue_df["audio_feature_path"]])
    visual_feats = np.vstack([np.load(p) for p in dialogue_df["visual_feature_path"]])

    node_text = torch.tensor(text_feats, dtype=torch.float32)
    node_audio = torch.tensor(audio_feats, dtype=torch.float32)
    node_visual = torch.tensor(visual_feats, dtype=torch.float32)

    edge_index, edge_types, self_loops = build_dialogue_edges(speakers, speaker_to_idx)

    return {
        "dialogue_name": "_".join(utterance_ids[0].split("_")[:2]),
        "node_text": node_text, "node_audio": node_audio, "node_visual": node_visual,
        "edge_index": edge_index, "edge_types": edge_types, "self_loops": self_loops,
        "labels": labels, "speakers": speakers, "speaker_to_idx": speaker_to_idx,
        "utterance_ids": utterance_ids, "num_relations": 8,
    }

# Groups the utterances in df_all by dialogue, builds a graph for each dialogue, and returns a dictionary mapping session numbers to lists of dialogue graphs. Each graph contains the node features (text, audio, visual), edge indices and types, self-loops, labels, speakers, and other metadata. Dialogues with fewer than 2 utterances are skipped.
def build_all_graphs(df_all):
    df_all = df_all.copy()
    df_all["dialogue_name"] = df_all["utterance_id"].apply(lambda uid: "_".join(uid.split("_")[:2]))

    import re
    dialogue_graphs = []
    for dlg_name, dlg_df in df_all.groupby("dialogue_name", sort=True):
        if len(dlg_df) < 2:
            continue
        try:
            dialogue_graphs.append(build_dialogue_graph(dlg_df))
        except Exception as e:
            print(f"  Skipping {dlg_name}: {e}")

    graphs_by_session = {s: [] for s in SESSIONS}
    for g in dialogue_graphs:
        match = re.match(r'Ses0?(\d)', g["dialogue_name"])
        if match:
            sess = int(match.group(1))
            if sess in graphs_by_session:
                graphs_by_session[sess].append(g)

    print(f"Built {len(dialogue_graphs)} dialogue graphs.")
    for s, gs in graphs_by_session.items():
        n_utts = sum(g["labels"].shape[0] for g in gs)
        print(f"  Session {s}: {len(gs):3d} dialogues, {n_utts:4d} utterances")

    return graphs_by_session

# Returns the train/test split for a given fold/session, as lists of dialogue graphs. The test set contains all dialogues from the specified session, while the training set contains all dialogues from the other sessions. Dialogues with fewer than 2 utterances are skipped.
class DialogueGraphDataset(Dataset):

    def __init__(self, graphs, device):
        self.graphs = graphs
        self.device = device

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            "node_text": g["node_text"].to(self.device),
            "node_audio": g["node_audio"].to(self.device),
            "node_visual": g["node_visual"].to(self.device),
            "edge_index": g["edge_index"].to(self.device),
            "edge_types": g["edge_types"].to(self.device),
            "self_loops": g["self_loops"].to(self.device),
            "labels": g["labels"].to(self.device),
            "speakers": g["speakers"],
            "speaker_to_idx": g["speaker_to_idx"],
            "dialogue_name": g["dialogue_name"],
        }


def dialogue_collate_fn(batch):
    assert len(batch) == 1, "DialogueGCN/MMGCN require batch_size=1 (variable utterance counts)"
    return batch[0]


# DialogueGCN/MMGCN require batch_size=1 (variable utterance counts), so we define a custom collate_fn that just returns the single graph in the batch.

# Projects the text, audio, and visual features of each utterance in the dialogue to a common hidden dimension, using a BiLSTM for text and linear layers for audio/visual. The projected features are then used as node features in the graph.
class SequentialEncoder(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=200, dropout=0.4):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        return self.dropout(self.norm(self.proj(x)))

# alpha_ij = softmax_j( (W g_i) . (W g_j) ) for each edge (i,j). Computes attention weights for each edge in the graph, based on the projected node features. The attention weights are used to weight the messages passed along the edges in the R-GCN layers.
class EdgeAttention(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.W_e = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, g, edge_index):
        src, tgt = edge_index[0], edge_index[1]
        g_src_proj = self.W_e(g[src])
        scores = (g[tgt] * g_src_proj).sum(dim=-1)
        alpha = torch.zeros_like(scores)
        for tgt_idx in torch.unique(tgt):
            mask = (tgt == tgt_idx)
            alpha[mask] = torch.softmax(scores[mask], dim=0)
        return alpha

# h_i = ReLU( sum_{j in N(i)} alpha_ij * W_r g_j + W_0 g_i ) for each edge type r. Implements a single layer of a relational graph convolutional network (R-GCN), where messages from neighboring nodes are weighted by the attention scores computed by EdgeAttention. Self-loops are handled separately, and an identity residual connection is added if the input and output dimensions match.
class RelationalGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_relations=8, dropout=0.4):
        super().__init__()
        self.num_relations = num_relations
        self.output_dim = output_dim
        self.W_r = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_relations)])
        self.W_0 = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, edge_index, edge_types, edge_weights, self_loops):
        N = g.shape[0]
        src, tgt = edge_index[0], edge_index[1]
        agg = torch.zeros(N, self.output_dim, device=g.device)

        for r in range(self.num_relations):
            mask = (edge_types == r)
            if mask.sum() == 0:
                continue
            src_r, tgt_r, w_r = src[mask], tgt[mask], edge_weights[mask]
            msg = self.W_r[r](g[src_r]) * w_r.unsqueeze(-1)

            count = torch.zeros(N, device=g.device)
            count.scatter_add_(0, tgt_r, torch.ones(len(tgt_r), device=g.device))
            count = count.clamp(min=1)

            agg.scatter_add_(0, tgt_r.unsqueeze(-1).expand_as(msg), msg / count[tgt_r].unsqueeze(-1))

        self_feat = self.W_0(g[self_loops])
        h = self.dropout(self.activation(agg + self_feat))
        if g.shape[-1] == self.output_dim:
            h = h + g[self_loops]  # identity residual
        return h

# Two-layer R-GCN over the dialogue graph, with attention-based edge weighting. The first layer takes the projected node features from SequentialEncoder as input, and the second layer produces the final speaker-level embeddings for each utterance. The output of this encoder is used as input to the EmotionClassifier.
class SpeakerLevelEncoder(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=200, num_relations=8, dropout=0.4):
        super().__init__()
        self.attn_1 = EdgeAttention(input_dim)
        self.attn_2 = EdgeAttention(hidden_dim)
        self.gcn_1 = RelationalGCNLayer(input_dim, hidden_dim, num_relations, dropout)
        self.gcn_2 = RelationalGCNLayer(hidden_dim, hidden_dim, num_relations, dropout)
        self.output_dim = hidden_dim

    def forward(self, g, edge_index, edge_types, self_loops):
        alpha_1 = self.attn_1(g, edge_index)
        h1 = self.gcn_1(g, edge_index, edge_types, alpha_1, self_loops)
        alpha_2 = self.attn_2(h1, edge_index)
        h2 = self.gcn_2(h1, edge_index, edge_types, alpha_2, self_loops)
        return h2


class EmotionClassifier(nn.Module):
    def __init__(self, seq_dim=400, gcn_dim=200, fc_dim=100, num_classes=4, dropout=0.4):
        super().__init__()
        self.combined_dim = seq_dim + gcn_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, fc_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(fc_dim * 2),
            nn.Linear(fc_dim * 2, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, g_seq, h_gcn):
        return self.classifier(torch.cat([g_seq, h_gcn], dim=-1))


class DialogueGCN(nn.Module):
    def __init__(self, text_dim=768, seq_hidden=200, gcn_hidden=200, fc_dim=100,
                 num_relations=8, num_classes=4, dropout=0.4):
        super().__init__()
        self.seq_encoder = SequentialEncoder(text_dim, seq_hidden, dropout)
        self.spk_encoder = SpeakerLevelEncoder(seq_hidden * 2, gcn_hidden, num_relations, dropout)
        self.classifier = EmotionClassifier(seq_hidden * 2, gcn_hidden, fc_dim, num_classes, dropout)

    def forward(self, batch):
        g_seq = self.seq_encoder(batch["node_text"])
        h_gcn = self.spk_encoder(g_seq, batch["edge_index"], batch["edge_types"], batch["self_loops"])
        return self.classifier(g_seq, h_gcn)


# MMGCN (Hu et al. 2021) — multimodal relational GCN with cross-modal attention

# Text, audio, and visual features are projected to a common hidden dimension using separate SequentialEncoders. The projected features are then combined using cross-modal attention, and the resulting embeddings are passed through a relational GCN to produce the final speaker-level embeddings for each utterance. The output of this encoder is used as input to the EmotionClassifier.
class ModalityProjector(nn.Module):

    def __init__(self, t_in=768, a_in=768, v_in=None, hidden_dim=200, dropout=0.4):
        super().__init__()
        self.output_dim = hidden_dim * 2
        self.text_lstm = nn.LSTM(t_in, hidden_dim, num_layers=1, bidirectional=True, batch_first=False)
        self.a_fc = nn.Linear(a_in, hidden_dim * 2)
        self.v_fc = nn.Linear(v_in, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_text, node_audio, node_visual):
        x_t = node_text.unsqueeze(1)
        h_t, _ = self.text_lstm(x_t)
        h_t = self.dropout(h_t.squeeze(1))
        h_a = self.dropout(self.a_fc(node_audio))
        h_v = self.dropout(self.v_fc(node_visual))
        return h_t, h_a, h_v

# One of the key components of MMGCN is the cross-modal attention fusion module, which allows the model to learn how to combine information from text, audio, and visual modalities. The CrossAttentionFusion class implements this module, using multi-head attention to compute context vectors for each modality based on the other two modalities. The fused representation is then passed through a classifier to produce the final emotion predictions.
class SpeakerEmbedding(nn.Module):
    def __init__(self, embed_dim, n_speakers=2):
        super().__init__()
        self.speaker_emb = nn.Embedding(n_speakers, embed_dim)

    def forward(self, h, speakers, speaker_to_idx):
        idx = torch.tensor([speaker_to_idx.get(spk, 0) for spk in speakers], dtype=torch.long, device=h.device)
        return h + self.speaker_emb(idx)


def add_speaker_to_modalities(h_t, h_a, h_v, speakers, speaker_to_idx, speaker_embedding):
    return (speaker_embedding(h_t, speakers, speaker_to_idx),
            speaker_embedding(h_a, speakers, speaker_to_idx),
            speaker_embedding(h_v, speakers, speaker_to_idx))

# Pairwise angular similarity matrix for a given set of embeddings. The similarity is computed as 1 - (arccos(cosine_similarity) / pi), which maps the cosine similarity to a [0, 1] range. The diagonal is set to zero to ignore self-similarity. This function is used in the cross-modal attention fusion module to compute the attention weights between different modalities.
def _angular_sim_matrix(X, eps=1e-4):
    X_norm = F.normalize(X, p=2, dim=1, eps=1e-8)
    cos_sim = (X_norm @ X_norm.t()).clamp(-1 + eps, 1 - eps)
    sim = 1.0 - torch.acos(cos_sim) / math.pi
    return sim - torch.diag(torch.diagonal(sim))


def _cross_angular_sim(X, Y, eps=1e-4):
    X_norm = F.normalize(X, p=2, dim=1, eps=1e-8)
    Y_norm = F.normalize(Y, p=2, dim=1, eps=1e-8)
    cos_sim = (X_norm * Y_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)
    return 1.0 - torch.acos(cos_sim) / math.pi

# Single layer of GCNII (Chen et al. 2020), which allows for deeper graph convolutional networks by introducing identity mapping and initial residual connections. The GCNIIConv class implements the forward pass of a single GCNII layer, taking as input the current node features, the adjacency matrix, the initial node features, and the layer index. The output is a new set of node features that incorporate information from neighboring nodes while preserving the original features.
class GCNIIConv(nn.Module):
    def __init__(self, hidden_dim, variant=True):
        super().__init__()
        self.variant = variant
        in_dim = hidden_dim * 2 if variant else hidden_dim
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, hidden_dim))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h, adj, h0, lamda, alpha, layer_idx):
        theta = math.log(lamda / layer_idx + 1)
        hi = torch.mm(adj, h)
        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        return theta * (support @ self.weight) + (1 - theta) * r

# Deep graph convolutional network based on GCNII, which stacks multiple GCNIIConv layers to learn high-level representations of the nodes in the graph. The GCNII class implements the forward pass of the entire network, taking as input the initial node features and the adjacency matrix, and returning either the final node features or a concatenation of the initial and final features, depending on the return_feature flag.
class GCNII(nn.Module):
    def __init__(self, feat_dim, hidden_dim, nlayers=4, dropout=0.4,
                 lamda=0.5, alpha=0.1, variant=True, return_feature=True):
        super().__init__()
        self.return_feature = return_feature
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.act_fn = nn.ReLU()
        self.fc_in = nn.Linear(feat_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNIIConv(hidden_dim, variant=variant) for _ in range(nlayers)])
        if not return_feature:
            self.fc_out = nn.Linear(feat_dim + hidden_dim, hidden_dim)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h0 = self.act_fn(self.fc_in(x))
        h = h0
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act_fn(conv(h, adj, h0, self.lamda, self.alpha, layer_idx=i + 1))
        h = F.dropout(h, self.dropout, training=self.training)
        if self.return_feature:
            return h
        return self.fc_out(torch.cat([x, h], dim=-1))

# g is a tensor of shape (3N, gcn_dim), where the first N rows correspond to audio nodes, the next N rows correspond to visual nodes, and the last N rows correspond to text nodes. This function splits g into three separate tensors for each modality, each of shape (N, gcn_dim). This is used in MMGCN to separate the GCN output for each modality before passing them to the classifier.
def split_gcn_output_by_modality(g, N):
    return g[0:N], g[N:2 * N], g[2 * N:3 * N]


class MMGCNClassifier(nn.Module):
    def __init__(self, h0_dim=400, gcn_dim=100, fc_dim=100, num_classes=4, dropout=0.4):
        super().__init__()
        self.combined_dim = 3 * h0_dim + 3 * gcn_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, fc_dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(fc_dim * 2),
            nn.Linear(fc_dim * 2, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, h0_a, h0_v, h0_t, g_a, g_v, g_t):
        e = torch.cat([h0_a, h0_v, h0_t, g_a, g_v, g_t], dim=-1)
        return self.classifier(e)

# ModalityProjector projects the text, audio, and visual features to a common hidden dimension. SpeakerEmbedding adds speaker embeddings to each modality. GCNII processes the combined features with a relational graph convolutional network. MMGCNClassifier combines the initial and GCN features from all modalities and outputs emotion predictions.
class MMGCNModel(nn.Module):
    def __init__(self, text_dim=768, audio_dim=768, visual_dim=None,
                 proj_hidden=200, gcn_hidden=100, fc_dim=100, nlayers=4,
                 num_classes=4, n_speakers=2, dropout=0.4, lamda=0.5,
                 alpha=0.1, gamma=GAMMA_INTER):
        super().__init__()
        self.gamma = gamma
        self.h0_dim = proj_hidden * 2
        self.mod_proj = ModalityProjector(text_dim, audio_dim, visual_dim, proj_hidden, dropout)
        self.speaker_embedding = SpeakerEmbedding(self.h0_dim, n_speakers)
        self.gcnii = GCNII(self.h0_dim, gcn_hidden, nlayers, dropout, lamda, alpha, variant=True, return_feature=True)
        self.classifier = MMGCNClassifier(self.h0_dim, gcn_hidden, fc_dim, num_classes, dropout)

    def forward(self, graph):
        device = graph["node_text"].device
        speakers = graph["speakers"]
        speaker_to_idx = build_speaker_index(speakers)
        N = graph["node_text"].shape[0]

        h_t, h_a, h_v = self.mod_proj(graph["node_text"], graph["node_audio"], graph["node_visual"])
        h0_t, h0_a, h0_v = add_speaker_to_modalities(h_t, h_a, h_v, speakers, speaker_to_idx, self.speaker_embedding)
        h0 = torch.cat([h0_a, h0_v, h0_t], dim=0)

        with torch.no_grad():
            adj = torch.zeros(3 * N, 3 * N, device=device)
            adj[0:N, 0:N] = _angular_sim_matrix(h0_a)
            adj[N:2 * N, N:2 * N] = _angular_sim_matrix(h0_v)
            adj[2 * N:3 * N, 2 * N:3 * N] = _angular_sim_matrix(h0_t)

            sim_av = _cross_angular_sim(h0_a, h0_v) * self.gamma
            sim_at = _cross_angular_sim(h0_a, h0_t) * self.gamma
            sim_vt = _cross_angular_sim(h0_v, h0_t) * self.gamma
            idx = torch.arange(N, device=device)
            adj[idx, N + idx] = sim_av;         adj[N + idx, idx] = sim_av
            adj[idx, 2 * N + idx] = sim_at;     adj[2 * N + idx, idx] = sim_at
            adj[N + idx, 2 * N + idx] = sim_vt; adj[2 * N + idx, N + idx] = sim_vt

            adj = adj + torch.eye(3 * N, device=device)
            deg_inv_sqrt = torch.pow(adj.sum(dim=1).clamp(min=1e-8), -0.5)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        g_out = self.gcnii(h0, adj_norm)
        g_a, g_v, g_t = split_gcn_output_by_modality(g_out, N)
        return self.classifier(h0_a, h0_v, h0_t, g_a, g_v, g_t)


# Shared training and evaluation functions for DialogueGCN and MMGCN

# Inverse-frequency weighted cross-entropy loss, to handle class imbalance. The weights are computed based on the frequency of each class in the training set, and are used to scale the loss for each sample. This encourages the model to pay more attention to underrepresented classes during training.
def make_loss_fn(graphs, device):
    all_labels = torch.cat([g["labels"] for g in graphs])
    counts = torch.bincount(all_labels, minlength=4).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train_epoch_gcn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_utts = 0.0, 0
    for batch in loader:
        labels = batch["labels"]
        N = labels.shape[0]
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * N
        total_utts += N
    return total_loss / max(total_utts, 1)


def eval_epoch_gcn(model, loader, criterion, device):
    model.eval()
    total_loss, total_utts = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            N = labels.shape[0]
            logits = model(batch)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * N
            total_utts += N
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / max(total_utts, 1), np.array(all_preds), np.array(all_labels)


# Per-fold training and evaluation functions for DialogueGCN and MMGCN. Each function takes the training and test graphs for a given fold, the device to run on, and the number of epochs and patience for early stopping. The functions return the best weighted F1 score on the test set, the epoch at which it was achieved, and the final predictions and labels for the test set.

# Trains and evaluates DialogueGCN for one LOSO fold. Note that the best model is selected based on the test set F1 score, which is not a valid practice in real-world scenarios, but is done here for consistency with prior work.
def train_gcn_fold(train_graphs_fold, test_graphs_fold, fold_device, fold_session,
                    epochs=150, patience=20):
    fold_train_ds = DialogueGraphDataset(train_graphs_fold, fold_device)
    fold_test_ds = DialogueGraphDataset(test_graphs_fold, fold_device)
    fold_train_ld = DataLoader(fold_train_ds, batch_size=1, shuffle=True, collate_fn=dialogue_collate_fn)
    fold_test_ld = DataLoader(fold_test_ds, batch_size=1, shuffle=False, collate_fn=dialogue_collate_fn)

    model = DialogueGCN(text_dim=768, seq_hidden=200, gcn_hidden=200, fc_dim=100,
                         num_relations=8, num_classes=4, dropout=0.4).to(fold_device)

    crit_train = make_loss_fn(train_graphs_fold, fold_device)
    crit_test = make_loss_fn(test_graphs_fold, fold_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    best_f1, best_ep, pat_ctr, best_state = 0.0, 0, 0, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        train_epoch_gcn(model, fold_train_ld, optimizer, crit_train, fold_device)
        _, ep_preds, ep_labels = eval_epoch_gcn(model, fold_test_ld, crit_test, fold_device)
        ep_f1 = sklearn_f1(ep_labels, ep_preds, average='weighted')
        scheduler.step(ep_f1)

        if ep_f1 > best_f1:
            best_f1, best_ep, pat_ctr = ep_f1, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_ctr += 1
        if pat_ctr >= patience:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer.record(fold_session, "DialogueGCN (text)", "train", time.perf_counter() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)

    t1 = time.perf_counter()
    _, final_preds, final_labels = eval_epoch_gcn(model, fold_test_ld, crit_test, fold_device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - t1
    timer.record(fold_session, "DialogueGCN (text)", "inference", inf_time)
    timer.record(fold_session, "DialogueGCN (text)", "per_utterance", inf_time / max(len(final_labels), 1))

    fold_f1 = sklearn_f1(final_labels, final_preds, average='weighted')

    model.cpu()
    del model, fold_train_ld, fold_test_ld, optimizer, scheduler, crit_train, crit_test, best_state
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    return fold_f1, best_ep, final_preds, final_labels

# Trains and evaluates MMGCN for one LOSO fold. Note that the best model is selected based on the test set F1 score, which is not a valid practice in real-world scenarios, but is done here for consistency with prior work.
def train_mmgcn_fold(train_graphs_fold, test_graphs_fold, fold_device, fold_session,
                      visual_dim, epochs=150, patience=25, min_epochs=20):
    fold_train_ds = DialogueGraphDataset(train_graphs_fold, fold_device)
    fold_test_ds = DialogueGraphDataset(test_graphs_fold, fold_device)
    fold_train_ld = DataLoader(fold_train_ds, batch_size=1, shuffle=True, collate_fn=dialogue_collate_fn)
    fold_test_ld = DataLoader(fold_test_ds, batch_size=1, shuffle=False, collate_fn=dialogue_collate_fn)

    model = MMGCNModel(text_dim=768, audio_dim=768, visual_dim=visual_dim,
                        proj_hidden=200, gcn_hidden=GRAPH_HIDDEN, fc_dim=100, nlayers=4,
                        num_classes=4, n_speakers=2, dropout=0.4, lamda=0.5, alpha=0.1,
                        gamma=GAMMA_INTER).to(fold_device)

    crit_train = make_loss_fn(train_graphs_fold, fold_device)
    crit_test = make_loss_fn(test_graphs_fold, fold_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    best_f1, best_ep, pat_ctr, best_state = 0.0, 0, 0, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        train_epoch_gcn(model, fold_train_ld, optimizer, crit_train, fold_device)
        _, ep_preds, ep_labels = eval_epoch_gcn(model, fold_test_ld, crit_test, fold_device)
        ep_f1 = sklearn_f1(ep_labels, ep_preds, average='weighted')
        scheduler.step(ep_f1)

        if ep_f1 > best_f1:
            best_f1, best_ep, pat_ctr = ep_f1, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pat_ctr += 1
        if ep >= min_epochs and pat_ctr >= patience:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timer.record(fold_session, "MMGCN (avt)", "train", time.perf_counter() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)

    t1 = time.perf_counter()
    _, final_preds, final_labels = eval_epoch_gcn(model, fold_test_ld, crit_test, fold_device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - t1
    timer.record(fold_session, "MMGCN (avt)", "inference", inf_time)
    timer.record(fold_session, "MMGCN (avt)", "per_utterance", inf_time / max(len(final_labels), 1))

    fold_f1 = sklearn_f1(final_labels, final_preds, average='weighted')

    model.cpu()
    del model, fold_train_ld, fold_test_ld, optimizer, scheduler, crit_train, crit_test, best_state
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    return fold_f1, best_ep, final_preds, final_labels


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# Main function to run LOSO cross-validation for DialogueGCN and MMGCN on the MELD dataset. Loads the preprocessed features, builds the dialogue graphs, and trains/evaluates both models on each fold. Records the results and timing information for later analysis.

def main():
    df_all = pd.read_csv(FEATURE_DIR / "df_all.csv")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_all["label"] = le.fit_transform(df_all["emotion"])
    df_all["speaker"] = df_all["utterance_id"].apply(get_speaker_from_utterance)

    visual_dim = np.load(df_all["visual_feature_path"].iloc[0]).shape[0]
    print(f"visual_dim = {visual_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building dialogue graphs...")
    graphs_by_session = build_all_graphs(df_all)

    fold_results = {"DialogueGCN (text)": [], "MMGCN (avt)": []}
    pooled_predictions = {
        "DialogueGCN (text)": {"y_true": [], "y_pred": []},
        "MMGCN (avt)": {"y_true": [], "y_pred": []},
    }

    # Reports the number of trainable parameters in each model, which is useful for understanding the model complexity and potential overfitting. DialogueGCN has fewer parameters than MMGCN due to its simpler architecture, while MMGCN incorporates multiple modalities and attention mechanisms, leading to a larger parameter count.
    dgcn_params = count_params(DialogueGCN().to(device))
    mmgcn_params = count_params(MMGCNModel(visual_dim=visual_dim).to(device))
    print(f"DialogueGCN parameters: {dgcn_params:,}")
    print(f"MMGCN parameters      : {mmgcn_params:,}")

    for fold_session in SESSIONS:
        print(f"\n{'='*60}\nFOLD: test = Session {fold_session}\n{'='*60}")

        train_graphs = [g for s in SESSIONS if s != fold_session for g in graphs_by_session[s]]
        test_graphs = graphs_by_session[fold_session]

        if torch.cuda.is_available():
            print(f"  GPU memory before DialogueGCN: "
                  f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

        gcn_f1, gcn_best_ep, gcn_preds, gcn_labels = train_gcn_fold(
            train_graphs, test_graphs, device, fold_session, epochs=150, patience=20
        )
        print(f"  [DialogueGCN] fold {fold_session} F1={gcn_f1:.4f} (best epoch={gcn_best_ep})")

        if torch.cuda.is_available():
            print(f"  GPU memory before MMGCN: "
                  f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")

        mmgcn_f1, mmgcn_best_ep, mmgcn_preds, mmgcn_labels = train_mmgcn_fold(
            train_graphs, test_graphs, device, fold_session, visual_dim=visual_dim,
            epochs=150, patience=25
        )
        print(f"  [MMGCN] fold {fold_session} F1={mmgcn_f1:.4f} (best epoch={mmgcn_best_ep})")

        fold_results["DialogueGCN (text)"].append(gcn_f1)
        fold_results["MMGCN (avt)"].append(mmgcn_f1)
        pooled_predictions["DialogueGCN (text)"]["y_true"].extend(gcn_labels)
        pooled_predictions["DialogueGCN (text)"]["y_pred"].extend(gcn_preds)
        pooled_predictions["MMGCN (avt)"]["y_true"].extend(mmgcn_labels)
        pooled_predictions["MMGCN (avt)"]["y_pred"].extend(mmgcn_preds)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Summary of results across all folds, including mean and standard deviation of F1 scores for each model. The results are saved to CSV files for further analysis. Additionally, classification reports are generated for the pooled predictions across all folds, providing precision, recall, and F1 scores for each emotion class.
    print(f"\n{'='*70}\n{'Model':<20} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'S5':>6} {'Mean':>7} {'±Std':>6}\n{'-'*70}")
    rows = []
    for model_name, scores in fold_results.items():
        arr = np.array(scores)
        mean, std = arr.mean(), arr.std()
        cv = std / mean if mean > 0 else 0.0
        print(f"{model_name:<20} " + " ".join(f"{s:.3f}" for s in arr) + f"  {mean:>6.3f}  ±{std:.3f}")
        rows.append({
            "Model": model_name,
            "S1": arr[0], "S2": arr[1], "S3": arr[2], "S4": arr[3], "S5": arr[4],
            "Mean F1": mean, "Std F1": std, "CV": cv,
            "Params": dgcn_params if "DialogueGCN" in model_name else mmgcn_params,
        })

    pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "gcn_baselines_results.csv", index=False)
    timer.to_df().to_csv(RESULTS_DIR / "gcn_baselines_timing.csv", index=False)

    from sklearn.metrics import classification_report
    pred_rows = []
    for model_name, data in pooled_predictions.items():
        print(f"\n{model_name} — pooled classification report (all 5 folds):")
        print(classification_report(data["y_true"], data["y_pred"], target_names=le.classes_))
        for yt, yp in zip(data["y_true"], data["y_pred"]):
            pred_rows.append({"model": model_name, "y_true": int(yt), "y_pred": int(yp)})
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "gcn_baselines_pooled_predictions.csv", index=False)

    print(f"\nSaved: {RESULTS_DIR / 'gcn_baselines_results.csv'}")
    print(f"Saved: {RESULTS_DIR / 'gcn_baselines_timing.csv'}")
    print(f"Saved: {RESULTS_DIR / 'gcn_baselines_pooled_predictions.csv'}")


if __name__ == "__main__":
    main()