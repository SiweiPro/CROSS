from torch import nn
import numpy as np
import torch
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from models.modules import MergeLayer

class TemporalChain(nn.Module):

    def __init__(self, dataset_name: str, node_feat_dim: int, temporal_chain_length: int, temporal_chain_features: dict):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param feature_dim: int, dimension of node memories
        """
        super(TemporalChain, self).__init__()

        # dict {node_id: (timetamps: np.array, features: np.array)}
        self.is_DGELT = 'GDELT' in dataset_name
        self.temporal_chain_features = temporal_chain_features
        self.feature_dim = node_feat_dim
        self.temporal_chain_length = temporal_chain_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def find_temporal_chain_features(self, node_ids: np.array, node_interact_times: np.ndarray, temporal_chain_length: int):
        """
        find temporal chain features based on node ids and current times.
        :param node_ids:
        :param node_interact_times:
        :return:
        """
        assert temporal_chain_length - 1 >= 0, 'Maximal number of temporal chain features for each node should be greater than 1!'

        temporal_chain_features, temporal_chain_timestamps = [], []
        max_chain_length = 0

        for index, node_id in enumerate(node_ids):
            if node_id != 0:
                temporal_chain_timestamp, temporal_chain_feature = self.temporal_chain_features[node_id]
                if self.is_DGELT:
                    temporal_chain_timestamp = temporal_chain_timestamp // 15
                else:
                    temporal_chain_timestamp = temporal_chain_timestamp // 1
                i = np.searchsorted(temporal_chain_timestamp, node_interact_times[index])
                max_chain_length = max(max_chain_length, len(temporal_chain_feature[: i]))
                temporal_chain_features.append(temporal_chain_feature[: i])
                # temporal_chain_timestamp[0] = float(0.0)
                # print(temporal_chain_timestamp[0])
                temporal_chain_timestamps.append(temporal_chain_timestamp[: i])
                assert len(temporal_chain_timestamp[: i]) > 0, f"node_id: {node_id} has no temporal chain features!"
            else:
                temporal_chain_features.append(np.zeros(self.feature_dim))
                temporal_chain_timestamps.append(np.array([0.0]))

        batch_chain_length = min(max_chain_length, temporal_chain_length)

        padded_temporal_chain_features = np.zeros((len(node_ids), batch_chain_length, self.feature_dim)).astype(np.float32)
        padded_temporal_chain_timestamps = np.zeros((len(node_ids), batch_chain_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            if node_ids[idx] != 0:
                node_chain_length = min(len(temporal_chain_timestamps[idx]), batch_chain_length)
                # Check information leakage
                assert np.max(temporal_chain_timestamps[idx]) < node_interact_times[idx], "Timestamps of the historical temporal chain should be ste less than the current timestamp!"
                temporal_chain_timestamps[idx][0] = 0.0
                padded_temporal_chain_features[idx, -node_chain_length:] = temporal_chain_features[idx][-node_chain_length:]
                padded_temporal_chain_timestamps[idx, -node_chain_length:] = temporal_chain_timestamps[idx][-node_chain_length:]
        # print(padded_temporal_chain_timestamps)
        padded_temporal_chain_features = torch.from_numpy(np.array(padded_temporal_chain_features).astype(np.float32)).to(self.device)

        return padded_temporal_chain_features, padded_temporal_chain_timestamps


class RawTemporalChainFusion4TGAT(nn.Module):
    def __init__(self, node_feat_dim: int, time_feat_dim: int, attention_dim: int, dropout: float = 0.1):
        super(RawTemporalChainFusion4TGAT, self).__init__()
        self.node_feat_dim = node_feat_dim
        self.time_feat_dim = time_feat_dim
        self.attention_dim = attention_dim

        self.fusion_layer = MergeLayer(input_dim1=node_feat_dim + self.time_feat_dim, input_dim2=attention_dim,
                                       hidden_dim=node_feat_dim + self.time_feat_dim + attention_dim, output_dim=node_feat_dim + time_feat_dim + attention_dim)

        self.layer_norm = nn.LayerNorm(self.node_feat_dim + self.time_feat_dim)

        self.residual_fc = nn.Linear(self.node_feat_dim + self.time_feat_dim, self.node_feat_dim + self.time_feat_dim)

        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(in_features=self.attention_dim, out_features=self.node_feat_dim, bias=True)


    def forward(self, raw_embedding: torch.Tensor, temporal_chain_embedding: torch.Tensor, residual: torch.Tensor):
        """
        Fusion features for the raw embeddings and the temporal chain embeddings.
        :param raw_embedding: Tensor, shape (batch_size, 1, num_heads * self.head_dim), where num_heads * self.head_dim is equal to node_feat_dim + time_feat_dim
        :param temporal_chain_embedding: Tensor, shape (batch_size, num_neighbor + 1, self.attention_dim)
        :param residual: The residual layer for the TGAT layer. Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        :return:
        """
        # prepare the interatcion embeddings
        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        graph_modal_embedding = raw_embedding.squeeze(dim=1)
        # Tensor, shape (batch_size, attention_dim)
        text_modal_embedding = temporal_chain_embedding[:, -1, :]

        # fusion oparation
        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim + attention_dim)
        fusion_embeddings = self.fusion_layer(graph_modal_embedding, text_modal_embedding)
        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        graph_modal_embedding = fusion_embeddings[:, :self.node_feat_dim + self.time_feat_dim]
        # Tensor, shape (batch_size, attention_dim)
        text_modal_embedding = fusion_embeddings[:, self.node_feat_dim + self.time_feat_dim:]

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        graph_modal_embedding = graph_modal_embedding.unsqueeze(dim=1)
        # Tensor, shape (batch_size, num_neighbor + 1, attention_dim)
        temporal_chain_embedding[:, -1, :] = text_modal_embedding

        # Get graph embeddings
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        graph_modal_embedding = self.dropout(self.residual_fc(graph_modal_embedding))
        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        graph_modal_embedding = self.layer_norm(graph_modal_embedding + residual)
        # Tensor, shape (batch_size, node_feat_dim + time_feat_dim)
        graph_modal_embedding = graph_modal_embedding.squeeze(dim=1)

        # Get text embeddings
        # Tensor, shape (batch_size, attention_dim)
        temporal_chain_embedding = torch.mean(temporal_chain_embedding, dim=1)
        temporal_chain_embedding = self.output_layer(temporal_chain_embedding)

        return graph_modal_embedding, temporal_chain_embedding, fusion_embeddings


class RawTemporalChainFusion4DyGFormer(nn.Module):
    def __init__(self, attention_dim: int, dropout: float = 0.1):
        super(RawTemporalChainFusion4DyGFormer, self).__init__()
        self.attention_dim = attention_dim

        self.fusion_layer = MergeLayer(input_dim1=self.attention_dim, input_dim2=attention_dim,
                                       hidden_dim=attention_dim * 2, output_dim=attention_dim * 2)

        self.layer_norm = nn.LayerNorm(self.attention_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, raw_embedding: torch.Tensor, temporal_chain_embedding: torch.Tensor):
        """
        Fusion features for the raw embeddings and the temporal chain embeddings.
        :param raw_embedding: Tensor, shape (batch_size, num_patches, attention_dim)
        :param temporal_chain_embedding: Tensor, shape (batch_size, temporal_chain_length, self.attention_dim)
        :return:
        """
        # prepare the interatcion embeddings
        # Tensor, shape (batch_size, attention_dim)
        graph_modal_embedding = raw_embedding[:, 0, :]
        # Tensor, shape (batch_size, attention_dim)
        text_modal_embedding = temporal_chain_embedding[:, -1, :]

        # fusion oparation
        # Tensor, shape (batch_size, attention_dim)
        fusion_embeddings = self.fusion_layer(graph_modal_embedding, text_modal_embedding)
        # Tensor, shape (batch_size, attention_dim)
        graph_modal_embedding = fusion_embeddings[:, :self.attention_dim]
        # Tensor, shape (batch_size, attention_dim)
        text_modal_embedding = fusion_embeddings[:, self.attention_dim:]

        # Tensor, shape (batch_size, 1, node_feat_dim + time_feat_dim)
        raw_embedding[:, 0, :] = graph_modal_embedding
        # Tensor, shape (batch_size, num_neighbor + 1, attention_dim)
        temporal_chain_embedding[:, -1, :] = text_modal_embedding

        return raw_embedding, temporal_chain_embedding, fusion_embeddings


class TemporalChainEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TemporalChainEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=2 * attention_dim),
            nn.Linear(in_features=2 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, node_temporal_chain_features: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param node_temporal_chain_features: Tensor, shape (batch_size, 1 + temporal_chain_length, node_feat_dim + time_feat_dim)
        :return:
        """
        # Tensor, shape (batch_size, 1, self.node_feat_dim)
        # node_temporal_chain_features = node_temporal_chain_features.unsqueeze(1)
        # node_time_features = node_time_features.unsqueeze(1)

        # Tensor, shape (batch_size, num_neighbor + 1, self.node_feat_dim)
        # input_node_features = torch.cat([node_temporal_chain_features, neighbor_node_temporal_chain_features], dim=1)
        # input_time_features = torch.cat([node_time_features, neighbor_node_time_features], dim=1)

        # print([input_node_features.shape, input_time_features.shape])
        # Tensor, shape (batch_size, num_neighbor + 1), mask the pad features
        mask = torch.all((node_temporal_chain_features == 0.0), dim=2)
        # Tensor, shape (batch_size, num_neighbor + 1, self.attention_dim)
        # inputs = self.projection_layer(node_temporal_chain_features)

        inputs = node_temporal_chain_features
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_neighbor + 1, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs, key_padding_mask=mask)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs

