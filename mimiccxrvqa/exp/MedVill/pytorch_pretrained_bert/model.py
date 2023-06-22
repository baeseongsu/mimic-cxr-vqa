"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import shutil
import logging
import tarfile
import tempfile
import numpy as np

from .file_utils import cached_path
from .loss import LabelSmoothingLoss

import torch
import torchvision
import torch.nn.functional as F

from torch import nn


class pixel_full_sampling(nn.Module):
    def __init__(self):
        super(pixel_full_sampling, self).__init__()
        # self.args = args
        model = torchvision.models.resnet50(pretrained=True)
        # modules = list(model.children())[:-1]
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = nn.AdaptiveAvgPool2d
        self.pool = pool_func((3, 1))

    def forward(self, x):
        out = self.model(x)
        # print("11 out", out.shape)
        out = torch.flatten(out, start_dim=2)  # out torch.Size([100, 2048, 3])
        # print("22 out", out.shape)
        out = out.transpose(1, 2).contiguous()  # out torch.Size([100, 3, 2048])
        # print("33 out", out.shape)

        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])

        # print("out", out.shape)
        # print("vis_pe", vis_pe.shape)
        # exit()
        return out, vis_pe


class pixel_random_sample(nn.Module):
    def __init__(self, args):
        super(pixel_random_sample, self).__init__()
        self.args = args
        model = torchvision.models.densenet121(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)  # 512x512: torch.Size([16, 2048, 16, 16])
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()  # B x N x 2048
        vis_pe = torch.arange(out.size()[1], dtype=torch.long).cuda()
        vis_pe = vis_pe.unsqueeze(0).expand(out.size()[0], out.size()[1])
        num_range = out.size()[1]
        random_sampling = torch.randperm(num_range)[: self.args.len_vis_input]
        random_sampling, _ = torch.sort(random_sampling)
        pixel_random_sample = out[:, random_sampling]
        random_position = vis_pe[:, random_sampling]
        return pixel_random_sample, random_position


logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = "bert_config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        relax_projection=0,
        initializer_range=0.02,
        task_idx=None,
        fp32_embedding=False,
        label_smoothing=None,
    ):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.label_smoothing = label_smoothing
        else:
            raise ValueError("First argument must be either a vocabulary size (int)" "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root)."""
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        if hasattr(config, "fp32_embedding"):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, vis_feats=None, vis_pe=None, vis_input=False, len_vis_input=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention " "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None, output_attentions=False):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # return context_layer

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None, output_attentions=False):
        self_output = self.self(input_tensor, attention_mask, history_states=history_states, output_attentions=output_attentions)
        # attention_output = self.output(self_output, input_tensor)
        # return attention_output
        attention_output = self.output(self_output[0], input_tensor)
        outputs = (attention_output,) + self_output[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None, output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            history_states=history_states,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # return layer_output
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        prev_embedding=None,
        prev_encoded_layers=None,
        output_all_encoded_layers=True,
        output_attentions=False,
    ):
        assert (prev_embedding is None) == (prev_encoded_layers is None), "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []  # if output_all_encoded_layers else None
        all_self_attentions = () if output_attentions else None

        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                # hidden_states = layer_module(hidden_states, attention_mask, history_states=history_states)
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    history_states=history_states,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                # hidden_states = layer_module(hidden_states, attention_mask)
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions=output_attentions)
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        # return all_encoder_layers
        return tuple(
            v
            for v in [
                all_encoder_layers,
                all_self_attentions,
            ]
            if v is not None
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, "relax_projection") and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        if hasattr(config, "relax_projection") and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            hidden_states = hidden_states.view(num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        seq_relationship_score = None
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(self.__class__.__name__, self.__class__.__name__)
            )
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(pretrained_model_name, ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()), archive_file)
            )
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ("config_path" in kwargs) and kwargs["config_path"]:
            config_file = kwargs["config_path"]
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if "type_vocab_size" in kwargs:
            config.type_vocab_size = kwargs["type_vocab_size"]
        # define new relax_projection
        if ("relax_projection" in kwargs) and kwargs["relax_projection"]:
            config.relax_projection = kwargs["relax_projection"]
        # define new relax_projection
        if ("task_idx" in kwargs) and kwargs["task_idx"]:
            config.task_idx = kwargs["task_idx"]
        # define new max position embedding for length expansion
        if ("max_position_embeddings" in kwargs) and kwargs["max_position_embeddings"]:
            config.max_position_embeddings = kwargs["max_position_embeddings"]
        # use fp32 for embeddings
        if ("fp32_embedding" in kwargs) and kwargs["fp32_embedding"]:
            config.fp32_embedding = kwargs["fp32_embedding"]
        # label smoothing
        if ("label_smoothing" in kwargs) and kwargs["label_smoothing"]:
            config.label_smoothing = kwargs["label_smoothing"]
        if "drop_prob" in kwargs:
            print("setting the new dropout rate!", kwargs["drop_prob"])
            config.attention_probs_dropout_prob = kwargs["drop_prob"]
            config.hidden_dropout_prob = kwargs["drop_prob"]

        logger.info("Model config {}".format(config))
        # clean the arguments in kwargs
        for arg_clean in ("config_path", "type_vocab_size", "relax_projection", "task_idx", "max_position_embeddings", "fp32_embedding", "label_smoothing", "drop_prob"):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = "bert.embeddings.token_type_embeddings.weight"
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].resize_(config.type_vocab_size, state_dict[_k].shape[1]).data
                if config.type_vocab_size >= 6:
                    # L2R
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    # R2L
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                    # S2S
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[: config.type_vocab_size, :]

        # initialize new position embeddings
        _k = "bert.embeddings.position_embeddings.weight"
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start : start + chunk_size, :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[: config.max_position_embeddings, :]

        # initialize relax projection
        _k = "cls.predictions.transform.dense.weight"
        n_config_relax = 1 if (config.relax_projection < 1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax * config.hidden_size != state_dict[_k].shape[0]):
            logger.info("n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = state_dict[_k].shape[0] // config.hidden_size
            assert (n_state_relax == 1) != (n_config_relax == 1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = "cls.predictions.transform.dense.weight"
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(n_config_relax, 1, 1).reshape((n_config_relax * config.hidden_size, config.hidden_size))
                for _k in ("cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias"):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, "task_idx") and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = "cls.predictions.transform.dense.weight"
                state_dict[_k].data = state_dict[_k].data.view(n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ("cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias"):
                    state_dict[_k].data = state_dict[_k].data.view(n_state_relax, config.hidden_size).select(0, _task_idx)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(model, prefix="" if hasattr(model, "bert") else "bert.")
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info("\n".join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        try:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        vis_feats,
        vis_pe,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        len_vis_input=None,
        output_attentions=False,
    ):
        extended_attention_mask = self.get_extended_attention_mask(input_ids, token_type_ids, attention_mask)

        # hack to load vis feats
        embedding_output = self.embeddings(vis_feats, vis_pe, input_ids, token_type_ids, len_vis_input=len_vis_input)
        # encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        # encoded_layers = self.encoder(
        #     embedding_output,
        #     extended_attention_mask,
        #     output_all_encoded_layers=output_all_encoded_layers,
        #     output_attentions=output_attentions,
        # )
        # sequence_output = encoded_layers[-1]

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_attentions=output_attentions,
        )
        encoded_layers = encoder_outputs[0]
        self_attentions = encoder_outputs[1] if not output_attentions else None
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return (encoded_layers, pooled_output, self_attentions)


class BertModelIncr(BertModel):
    def __init__(self, config, args):
        super(BertModelIncr, self).__init__(config)

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True, len_vis_input=None):

        extended_attention_mask = self.get_extended_attention_mask(input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(vis_feats, vis_pe, input_ids, token_type_ids, position_ids, vis_input=(prev_encoded_layers is None), len_vis_input=len_vis_input)
        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers, output_all_encoded_layers=output_all_encoded_layers
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):  # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_size)  # img_hidden_sz=2048, hidden_size=768

        if self.args.img_postion:  # True
            self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_id, input_imgs, vis_pe, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)
        bsz = input_imgs.size(0)
        seq_len = self.args.len_vis_input + 2
        cls_input_id = self.word_embeddings(input_id[:, :1])
        sep_input_id = self.word_embeddings(input_id[:, -1:])
        imgs_embeddings = self.img_embeddings(input_imgs)  # torch.Size([32, 5, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # torch.Size([32, 5, 768])

        token_embeddings = torch.cat([cls_input_id, imgs_embeddings, sep_input_id], dim=1)

        if self.args.img_postion:  # True
            position_ids = torch.tensor([0, seq_len - 1], dtype=torch.long).cuda()
            position_ids = position_ids.unsqueeze(0).expand(bsz, 2)
            position_embeddings = self.position_embeddings(position_ids)
            pos_vis_embeddings = self.position_embeddings(vis_pe)
            token_position_embeddings = torch.cat((position_embeddings[:, :1], pos_vis_embeddings, position_embeddings[:, -1:]), dim=1)
            embeddings = token_embeddings + token_position_embeddings + token_type_embeddings  # should be tensor
        else:
            embeddings = token_embeddings + token_type_embeddings  # should be tensor

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


""" for VLP, based on UniLM """


class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, args, num_labels=2, len_vis_input=None, tasks="img2txt"):
        super(BertForPreTrainingLossMask, self).__init__(config)
        bert = BertModel(config)

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        if args.img_encoding == "random_sample":
            self.img_encoder = pixel_random_sample(args)
        elif args.img_encoding == "fully_use_cnn":
            self.img_encoder = pixel_full_sampling()
        for p in self.img_encoder.parameters():
            p.requires_grad = False
        for c in list(self.img_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction="none")
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        if hasattr(config, "label_smoothing") and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=0, reduction="none")
        else:
            self.crit_mask_lm_smoothed = None

        self.tasks = tasks
        if self.tasks == "vqa":
            self.ans_classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size * 2), nn.ReLU(), nn.Linear(config.hidden_size * 2, 458))
            self.vqa_crit = nn.BCEWithLogitsLoss()
        else:
            self.cls = BertPreTrainingHeads(config, bert.embeddings.word_embeddings.weight, num_labels=num_labels)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        try:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        img,
        _,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        ans_labels=None,
        masked_pos=None,
        masked_weights=None,
        task_idx=None,
        drop_worst_ratio=0.2,
        vqa_inference=False,
        ans_type=None,
    ):
        vis_feats, vis_pe = self.img_encoder(img)  # image region features
        extended_attention_mask = self.get_extended_attention_mask(input_ids, token_type_ids, attention_mask)
        img_embed_out = self.img_embeddings(input_ids[:, : self.len_vis_input + 2], vis_feats, vis_pe, token_type_ids[:, : self.len_vis_input + 2])  # img_embed_out: torch.Size([32, 5, 768])

        txt_embed_out = self.txt_embeddings(
            input_ids[:, self.len_vis_input + 2 :], token_type_ids[:, self.len_vis_input + 2 :]
        )  # , attention_mask[:, self.len_vis_input+2:, self.len_vis_input+2:])  # txt_embed_out: torch.Size([32, 507, 768])
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID
        encoded_layers = self.encoder(encoder_input, extended_attention_mask)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if vqa_inference:
            assert ans_labels == None
            vqa_embed = sequence_output[:, 0] * sequence_output[:, self.len_vis_input + 1]
            vqa_pred = self.ans_classifier(vqa_embed)
            ans_idx = torch.max(vqa_pred[:, 1:], -1)[1] + 1
            return ans_idx

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def gather_seq_out_by_pos_average(seq, pos, mask):
            batch_size, max_token_num = pos.size(0), pos.size(-1)
            pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
            mask = mask.type_as(pos_vec)
            pos_vec_masked_sum = (pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
            return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

        def loss_mask_and_normalize(loss, mask, drop_worst_ratio):
            mask = mask.type_as(loss)
            loss = loss * mask
            # Ruotian Luo's drop worst
            keep_loss, keep_ind = torch.topk(loss.sum(-1), int(loss.size(0) * (1 - drop_worst_ratio)), largest=False)

            denominator = torch.sum(mask.sum(-1)[keep_ind]) + 1e-5
            return (keep_loss / denominator).sum()

        def compute_score_with_logits(logits, labels):
            logits = torch.max(logits, 1)[1].data  # argmax
            one_hots = torch.zeros(*labels.size()).to(logits.device)
            one_hots.scatter_(1, logits.view(-1, 1), 1)
            scores = one_hots * labels
            return scores  # (B, 456)

        dummy_value = pooled_output.new(1).fill_(0)

        if self.tasks == "vqa":
            score = 0
            num_data = 0
            assert ans_labels is not None
            # vqa_embed = sequence_output[:, 0]
            vqa_embed = sequence_output[:, 0] * sequence_output[:, self.len_vis_input + 1]
            vqa_pred = self.ans_classifier(vqa_embed)  # Linear & activation & Linear 가능한 정답 개수인 458차원으로 줄여줌
            vqa_loss = self.vqa_crit(vqa_pred, ans_labels)

            batch_score = compute_score_with_logits(vqa_pred, ans_labels).sum(dim=1)  # 맞춘 개수   torch.Size([B])
            vqa_acc = batch_score.sum() / vqa_pred.size(0)

            # print("vqa_ total acc", vqa_acc)
            closed_ans_score, open_ans_score = [], []
            for i in range(len(ans_type)):
                if ans_type[i] == 0:
                    # print("batch_score[i].item()", batch_score[i].item())
                    closed_ans_score.append(batch_score[i].item())
                elif ans_type[i] == 1:
                    # print("batch_score[i].item()", batch_score[i].item())
                    open_ans_score.append(batch_score[i].item())

            if len(closed_ans_score) != 0:
                closed_ans_score = sum(closed_ans_score) / len(closed_ans_score)

            if len(open_ans_score) != 0:
                open_ans_score = sum(open_ans_score) / len(open_ans_score)

            return dummy_value, vqa_loss, vqa_acc, closed_ans_score, open_ans_score
        else:
            sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos)
            prediction_scores_masked, _ = self.cls(sequence_output_masked, pooled_output, task_idx=task_idx)
            if self.crit_mask_lm_smoothed:
                masked_lm_loss = self.crit_mask_lm_smoothed(F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
            else:
                masked_lm_loss = self.crit_mask_lm(prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
            masked_lm_loss = loss_mask_and_normalize(masked_lm_loss.float(), masked_weights, drop_worst_ratio)
            return masked_lm_loss, dummy_value, dummy_value, dummy_value, dummy_value


class PixelRandomSamplingModule(nn.Module):
    """
    Modified Version (2022.07.13, Seongsu Bae)
    """

    def __init__(self, len_vis_input=180):
        super(PixelRandomSamplingModule, self).__init__()
        self.len_vis_input = len_vis_input
        # load resnet model (strips off second to last layer)
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)  # (bsz, 3, 512, 512) => (bsz, 2048, 16, 16)
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()  # (bsz, 2048, 16, 16) => (bsz, 256, 2048)

        vis_pe = torch.arange(out.size(1), dtype=torch.long).cuda()  # (256)
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))  # (256) => (bsz, 256)

        random_sampling = torch.randperm(out.size(1))[: self.len_vis_input]  # args.num_image_embeds
        random_sampling, _ = torch.sort(random_sampling)

        out = out[:, random_sampling]  # (bsz, 180, 2048)
        vis_pe = vis_pe[:, random_sampling]  # (bsz, 180)
        return out, vis_pe


class PixelFullUseModule(nn.Module):
    """
    Modified Version (2022.07.13, Seongsu Bae)
    """

    def __init__(self, len_vis_input=256):
        super(PixelFullUseModule, self).__init__()
        self.len_vis_input = len_vis_input
        # load resnet model (strips off second to last layer)
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        out = self.model(x)  # (bsz, 3, 512, 512) => (bsz, 2048, 16, 16)
        out = torch.flatten(out, start_dim=2).transpose(1, 2).contiguous()  # (bsz, 2048, 16, 16) => (bsz, 256, 2048)

        vis_pe = torch.arange(out.size(1), dtype=torch.long).cuda()  # (256)
        vis_pe = vis_pe.unsqueeze(0).expand(out.size(0), out.size(1))  # (256) => (bsz, 256)
        return out, vis_pe


class MedViLLForVQA(PreTrainedBertModel):
    def __init__(self, config, args, num_labels=458):
        super(MedViLLForVQA, self).__init__(config)

        # init arguments
        self.config = config
        self.args = args
        self.num_labels = num_labels
        self.len_vis_input = args.len_vis_input

        # Image/Text embeddings
        bert = BertModel(config)
        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        # Image Encoder
        if args.img_encoding == "random_sample":
            self.img_encoder = PixelRandomSamplingModule(len_vis_input=180)  # 180
        elif args.img_encoding == "fully_use_cnn":
            self.img_encoder = PixelFullUseModule(len_vis_input=256)

        for p in self.img_encoder.parameters():
            p.requires_grad = False

        # We train some part of the layer on the cnn model.
        for c in list(self.img_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = True

        self.encoder = bert.encoder
        self.pooler = bert.pooler

        # NOTE: for scratch
        self.apply(self.init_bert_weights)

        self.ans_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, num_labels),
        )

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        try:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_vqa_embeding(self, img, input_ids, token_type_ids=None, attention_mask=None, output_attentions=False):

        vis_feats, vis_pe = self.img_encoder(img)  # image region features
        img_embed_out = self.img_embeddings(input_ids[:, : self.len_vis_input + 2], vis_feats, vis_pe, token_type_ids[:, : self.len_vis_input + 2])  # img_embed_out: torch.Size([32, 5, 768])

        # text embedding
        txt_embed_out = self.txt_embeddings(
            input_ids[:, self.len_vis_input + 2 :], token_type_ids[:, self.len_vis_input + 2 :]
        )  # , attention_mask[:, self.len_vis_input+2:, self.len_vis_input+2:])  # txt_embed_out: torch.Size([32, 507, 768])

        # joint embedding
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID
        extended_attention_mask = self.get_extended_attention_mask(input_ids, token_type_ids, attention_mask)
        encoded_outputs = self.encoder(encoder_input, extended_attention_mask, output_attentions=output_attentions)
        encoded_layers = encoded_outputs[0]
        self_attentions = encoded_outputs[1] if output_attentions else None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)  # (bsz, seq_len, h_dim) => (bsz, h_dim)

        return sequence_output, pooled_output

    def forward(
        self,
        img,
        _,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        ans_labels=None,
        vqa_inference=False,
        ans_type=None,
        output_attentions=False,
    ):
        sequence_output, pooled_output = self.get_vqa_embeding(img, input_ids, token_type_ids, attention_mask, output_attentions)

        if vqa_inference:
            assert ans_labels == None
            vqa_embed = sequence_output[:, 0] * sequence_output[:, self.len_vis_input + 1]
            vqa_pred = self.ans_classifier(vqa_embed)
            ans_idx = torch.max(vqa_pred[:, 1:], -1)[1] + 1
            return ans_idx

        assert ans_labels != None
        vqa_embed = sequence_output[:, 0] * sequence_output[:, self.len_vis_input + 1]
        vqa_pred = self.ans_classifier(vqa_embed)

        # compute loss
        loss_fct = nn.BCEWithLogitsLoss()
        vqa_loss = loss_fct(vqa_pred, ans_labels)  # (16, 458) | (16, 458)

        if self.args.vqa_dataset in ["vqa-rad", "slake"]:

            def compute_score_with_logits(logits, labels):
                logits = torch.max(logits, 1)[1].data  # argmax
                one_hots = torch.zeros(*labels.size()).to(logits.device)
                one_hots.scatter_(1, logits.view(-1, 1), 1)
                scores = one_hots * labels
                return scores  # (B, 456)

            # compute score (torch.max)
            vqa_ans_score = compute_score_with_logits(vqa_pred, ans_labels).sum(dim=1).cpu().tolist()
            vqa_pred_logits = vqa_pred

            # compute open/closed accuracy
            closed_ans_score, open_ans_score = [], []
            assert len(ans_type) == len(vqa_ans_score)
            for idx, a_type in enumerate(ans_type):
                if a_type == 0:  # closed
                    flag = vqa_ans_score[idx]
                    closed_ans_score.append(flag)
                elif a_type == 1:  # open
                    flag = vqa_ans_score[idx]
                    open_ans_score.append(flag)
                else:
                    raise ValueError()
            assert len(vqa_ans_score) == len(closed_ans_score) + len(open_ans_score)

        elif self.args.vqa_dataset == "vqa-mimic":

            def compute_score_with_logits_multilabel(logits, labels):
                logits = torch.sigmoid(logits)
                one_hots = torch.zeros(*labels.size()).to(logits.device)
                one_hots[logits > 0.5] = 1
                # scores = accuracy_score(y_true=labels.cpu(), y_pred=one_hots.cpu(), normalize=False)
                scores = torch.all(labels.cpu() == one_hots.cpu(), axis=1).tolist()
                return scores, logits

            closed_ans_score, open_ans_score = [], []
            # compute score
            vqa_ans_score, vqa_pred_logits = compute_score_with_logits_multilabel(vqa_pred, ans_labels)

        # return vqa_loss, vqa_ans_score, vqa_pred_logits, closed_ans_score, open_ans_score
        output = (vqa_loss, vqa_ans_score, vqa_pred_logits, closed_ans_score, open_ans_score)
        if output_attentions:
            self_attentions = list(self_attentions)
            output = output + (self_attentions,)
        return output
