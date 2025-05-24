# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder definition."""
from typing import Tuple, Optional

import torch
import math


from model.attention import MultiHeadedAttention
from model.attention import StreamingRelPositionMultiHeadedAttention
from model.convolution import ConvolutionModule
from model.embedding import StreamingRelPositionalEncoding
from model.encoder_layer import ChunkFormerEncoderLayer
from model.positionwise_feed_forward import PositionwiseFeedForward
from model.subsampling import DepthwiseConvSubsampling
from model.utils.common import get_activation
from model.utils.mask import make_pad_mask

class BaseEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        use_limited_chunk: bool = False,
        limited_decoding_chunk_sizes: list = [],
        limited_left_chunk_sizes: list = [],
        use_context_hint_chunk: bool = False,
        right_context_sizes: list = [],
        right_context_probs: list = [],
        freeze_subsampling_layer: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
        super().__init__()
        self._output_size = output_size
        self.pos_enc_layer_type = pos_enc_layer_type
        self.attention_heads = attention_heads
        self.input_layer = input_layer


        pos_enc_class = StreamingRelPositionalEncoding
        subsampling_class = DepthwiseConvSubsampling

        self.global_cmvn = global_cmvn
        if subsampling_class == DepthwiseConvSubsampling:
            self.embed = subsampling_class(
                subsampling="dw_striding",
                subsampling_factor=8,
                feat_in=input_size,
                feat_out=output_size,
                conv_channels=output_size,
                pos_enc_class=pos_enc_class(output_size, positional_dropout_rate),
                subsampling_conv_chunking_factor=1,
                activation=torch.nn.ReLU(),
                is_causal=False,
            )
        else:
            self.embed = subsampling_class(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size * 1, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.use_limited_chunk = use_limited_chunk
        self.limited_decoding_chunk_sizes = torch.IntTensor(limited_decoding_chunk_sizes)
        self.limited_left_chunk_sizes = torch.IntTensor(limited_left_chunk_sizes)
        self.use_context_hint_chunk = use_context_hint_chunk
        self.right_context_sizes = torch.IntTensor(right_context_sizes)
        self.right_context_probs = torch.FloatTensor(right_context_probs)

        if freeze_subsampling_layer:
            self.freeze_subsampling_layer()


    def output_size(self) -> int:
        return self._output_size

    def freeze_subsampling_layer(self):
        for param in self.embed.parameters():
            param.requires_grad = False
    
    def forward_parallel_chunk(
        self,
        xs,
        xs_origin_lens,
        chunk_size: int = -1,
        left_context_size: int = -1,
        right_context_size: int = -1,
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        truncated_context_size:int = 0,
        offset: torch.Tensor = torch.zeros(0),
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        assert offset.shape[0] == len(xs), f"{offset.shape[0]} - {len(xs)}"
        
        # --------------------------Chunk Batching-------------------------------------------
        subsampling = self.embed.subsampling_factor
        context = self.embed.right_context + 1 # Add current frame
        size = (chunk_size - 1) * subsampling + context
        step = subsampling * chunk_size
        device = xs_origin_lens.device

        conv_lorder = self.cnn_module_kernel // 2

        upper_bounds = []
        lower_bounds = []
        upper_bounds_conv = []
        lower_bounds_conv = []
        x_pad = []
        xs_lens = []
        n_chunks = []
        for xs_origin_len, x, offs in zip(xs_origin_lens, xs, offset): # cost O(input_batch_size | ccu)
            x = x.to(device)
            # if x.size(0) >= size:
            #     n_frames_pad = (step - ((x.size(0) - size) %  step)) % step
            # else:
            #     n_frames_pad = size - x.size(0)
            # x = torch.nn.functional.pad(x, (0, 0, 0, n_frames_pad)) # (T, 80)
            
            #ĐỔI từ GPT :)))
            if x.size(0) >= size:
                n_frames_pad = (step - ((x.size(0) - size) % step)) % step
            else:
                n_frames_pad = size - x.size(0)
                # Đảm bảo n_frames_pad là tensor và chỉ định value rõ ràng
            n_frames_pad = torch.tensor(n_frames_pad, dtype=torch.int64, device=x.device)
            x = torch.nn.functional.pad(x, (0, 0, 0, n_frames_pad.item()), mode="constant", value=0.0) # (T, 80)


            n_chunk = ((x.size(0) - size) // step) + 1

            # Manual unfold for dimension 0 to replace x.unfold(0, size=size, step=step)
            # Original x has shape (T_padded, D_feat)
            # x.unfold(0, size, step) would produce (n_chunk, D_feat, size)
            if n_chunk > 0:
                unfolded_slices = []
                for i in range(n_chunk):
                    start_idx = i * step
                    # slice_ M_ N means take N elements from Mth dim
                    # narrow(dimension, start, length)
                    # Each slice will be of shape (size, D_feat)
                    a_slice = x.narrow(0, start_idx, size)
                    unfolded_slices.append(a_slice)
                
                # Stack along new dimension 0. Result: (n_chunk, size, D_feat)
                stacked_unfold = torch.stack(unfolded_slices, dim=0)
                # Transpose to match PyTorch's unfold(0,...) behavior for a 2D tensor which outputs (n_chunk, D_feat, size)
                x = stacked_unfold.transpose(1, 2).contiguous() # Now x is (n_chunk, D_feat, size)
            elif n_chunk == 0 and x.size(0) < size : # If input is smaller than one window
                 # This case means no full chunk can be formed. 
                 # The original unfold would produce an empty tensor on the first dimension.
                 # Shape: (0, D_feat, size)
                 x = torch.empty((0, x.size(1), size), device=x.device, dtype=x.dtype)
            else: # n_chunk is 0 or negative for other reasons (should ideally not occur with proper padding/size checks)
                 # Default to empty tensor matching the expected output structure of unfold
                 x = torch.empty((0, x.size(1), size), device=x.device, dtype=x.dtype)
            
            # Original next line was: x = x.transpose(2, 1)
            # If current x is (n_chunk, D_feat, size), then x.transpose(2,1) becomes (n_chunk, size, D_feat)
            x = x.transpose(2, 1) 

            max_len = 1  + (xs_origin_len - context)//subsampling
            upper_bound = chunk_size + right_context_size + torch.arange(0, 1 + (xs_origin_len + n_frames_pad - context)//subsampling, 1 + (size - context)//subsampling, device=device)
            lower_bound = upper_bound - max_len
            upper_bound += offs
            
            upper_bound_conv = chunk_size + conv_lorder + torch.arange(0, 1  + (xs_origin_len + n_frames_pad - context)//subsampling, 1 + (size - context)//subsampling, device=device)
            lower_bound_conv = torch.maximum(upper_bound_conv - max_len, torch.full_like(upper_bound_conv, conv_lorder - right_context_size))
            upper_bound_conv += offs


            xs_lens += [size] * (n_chunk - 1) + [size - n_frames_pad]
            upper_bounds.append(upper_bound)
            lower_bounds.append(lower_bound)
            upper_bounds_conv.append(upper_bound_conv)
            lower_bounds_conv.append(lower_bound_conv)
            x_pad.append(x)
            n_chunks.append(n_chunk)


        xs = torch.cat(x_pad, dim=0).to(device)
        xs_lens = torch.tensor(xs_lens).to(device)
        upper_bounds = torch.cat(upper_bounds).unsqueeze(1).to(device)
        lower_bounds = torch.cat(lower_bounds).unsqueeze(1).to(device)
        upper_bounds_conv = torch.cat(upper_bounds_conv).unsqueeze(1).to(device)
        lower_bounds_conv = torch.cat(lower_bounds_conv).unsqueeze(1).to(device)


        # forward model
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)


        xs, pos_emb, xs_lens = self.embed(xs, xs_lens, offset=left_context_size, right_context_size=right_context_size)
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1)  # (B, 1, T)


        mask_pad = torch.arange(0, conv_lorder + chunk_size + conv_lorder, device=masks.device).unsqueeze(0).repeat(xs.size(0), 1) # [B, left_context_size + chunksize]
        mask_pad = (lower_bounds_conv <= mask_pad) & (mask_pad < upper_bounds_conv)
        mask_pad = mask_pad.flip(-1).unsqueeze(1)
        att_mask = torch.arange(0, left_context_size + chunk_size + right_context_size, device=masks.device).unsqueeze(0).repeat(xs.size(0), 1) # [B, left_context_size + chunksize]
        att_mask = (lower_bounds <= att_mask) & (att_mask < upper_bounds)
        att_mask = att_mask.flip(-1).unsqueeze(1)


        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer.forward_parallel_chunk(xs, att_mask, pos_emb, 
                mask_pad=mask_pad,
                right_context_size=right_context_size,
                left_context_size=left_context_size,
                att_cache=att_cache[i].to(device) if att_cache.size(0) > 0 else att_cache,
                cnn_cache=cnn_cache[i].to(device) if cnn_cache.size(0) > 0 else cnn_cache,
                truncated_context_size=truncated_context_size

            )
            r_att_cache.append(new_att_cache)
            r_cnn_cache.append(new_cnn_cache)

        del att_cache
        del cnn_cache
        if self.normalize_before:
            xs = self.after_norm(xs)

        xs_lens = self.embed.calc_length(xs_origin_lens)
        offset += xs_lens


        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.stack(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.stack(r_cnn_cache, dim=0)
        return xs, xs_lens, n_chunks, r_att_cache, r_cnn_cache, offset
    
    def ctc_forward(self, xs, xs_lens=None, n_chunks=None):
        ctc_probs = self.ctc.log_softmax(xs)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        hyps = topk_index.squeeze(-1)  # (B, maxlen)

        if (n_chunks is not None) and (xs_lens is not None):
            hyps = hyps.split(n_chunks, dim=0)   
            hyps = [hyp.flatten()[:x_len] for hyp, x_len in zip(hyps, xs_lens)]
        return hyps  


    def rearrange(
        self, 
        xs,
        xs_lens,
        n_chunks
    ):
        xs = xs.split(n_chunks, dim=0)   
        xs_lens = self.embed.calc_length(xs_lens)
        xs = [x.reshape(-1, self._output_size)[:x_len] for x, x_len in zip(xs, xs_lens)]



        xs = torch.nn.utils.rnn.pad_sequence(xs,
                                    batch_first=True,
                                    padding_value=0)
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1).to(xs.device) # (B, 1, T)
        return xs, masks

    def forward(self,
                xs,
                xs_origin_lens,
                chunk_size: int,
                left_context_size: int,
                right_context_size: int,
                att_cache: torch.Tensor,
                cnn_cache: torch.Tensor,
                truncated_context_size: int,
                offset: torch.Tensor):
        # This forward method is specifically for ONNX export.
        # It calls the main processing method and returns Tensors compatible with ONNX.
        encoder_out, encoder_out_lens, _n_chunks, att_cache_out, cnn_cache_out, offset_out = self.forward_parallel_chunk(
            xs=xs,
            xs_origin_lens=xs_origin_lens,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            truncated_context_size=truncated_context_size,
            offset=offset
        )
        # Return only the Tensors expected by the ONNX graph and matching output_names
        # from export_onnx.py (encoder_out, encoder_out_lens, att_cache_out, cnn_cache_out, offset_out)
        return encoder_out, encoder_out_lens, att_cache_out, cnn_cache_out, offset_out

class ChunkFormerEncoder(BaseEncoder):
    """ChunkFormer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        use_limited_chunk: bool = False,
        limited_decoding_chunk_sizes: list = [],
        limited_left_chunk_sizes: list = [],
        use_dynamic_conv: bool = False,
        use_context_hint_chunk: bool = False,
        right_context_sizes: list = [],
        right_context_probs: list = [],
        freeze_subsampling_layer: bool = False,
    ):
        """Construct ChunkFormerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk, 
                         use_limited_chunk=use_limited_chunk,
                         limited_decoding_chunk_sizes=limited_decoding_chunk_sizes,
                         limited_left_chunk_sizes=limited_left_chunk_sizes,
                         use_context_hint_chunk=use_context_hint_chunk,
                         right_context_sizes=right_context_sizes,
                         right_context_probs=right_context_probs,
                         freeze_subsampling_layer=freeze_subsampling_layer)
        self.cnn_module_kernel = cnn_module_kernel
        activation = get_activation(activation_type)
        self.num_blocks = num_blocks
        self.use_dynamic_conv = use_dynamic_conv
        self.input_size = input_size
        self.attention_heads = attention_heads

        # self-attention module definition
        if pos_enc_layer_type == "abs_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        elif pos_enc_layer_type == "rel_pos":
            encoder_selfattn_layer = StreamingRelPositionMultiHeadedAttention
        elif pos_enc_layer_type == "stream_rel_pos":
            encoder_selfattn_layer = StreamingRelPositionMultiHeadedAttention
        
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, True, use_dynamic_conv)

        self.encoders = torch.nn.ModuleList([
            ChunkFormerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                aggregate=2 if ((i % 3 == 0) and  (i > 0)) else 1
            ) for i in range(num_blocks)
        ])
