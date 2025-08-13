import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_Transformer import GPTDecoderLayer, GPTDecoder
from MCMG_utils.utils import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransformerEncoder(nn.Module):
    """简单的 Transformer Encoder,处理 scaffold"""
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, 
                 max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            batch_first=False  # (seq, batch, dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    
    def forward(self, src, con_token=None, src_key_padding_mask=None):  # 添加 
        # src: (batch, seq) -> (seq, batch)
        src = src.transpose(0, 1)

        con_token = con_token.transpose(0, 1)
        src_embedded = self.embed(src) + self.embed(con_token).sum(dim=0).unsqueeze(0)
        
        # Embedding + Position encoding
        src = self.pos_enc(src_embedded * math.sqrt(self.d_model))
        
        # Encoder
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return memory  # (seq, batch, d_model)

class TransformerDecoder(nn.Module):
    """修改现有的 Decoder，增加 cross-attention 到 encoder memory"""
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, 
                 max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionEncoding(d_model, pos_dropout, max_seq_length)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            batch_first=False  # (seq, batch, dim)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory, con_token=None, tgt_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tgt: (batch, seq) -> (seq, batch)
        tgt = tgt.transpose(0, 1)
        
        # 添加条件 token（如果有）
        
        con_token = con_token.transpose(0, 1)
        tgt_embedded = self.embed_tgt(tgt) + self.embed_tgt(con_token).sum(dim=0).unsqueeze(0)

        
        # Position encoding
        tgt = self.pos_enc(tgt_embedded * math.sqrt(self.d_model))
        
        # Decoder with cross-attention
        output = self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # (seq, batch, d_model) -> (batch, seq, d_model)
        output = output.transpose(0, 1)
        
        return self.fc(output)


class EncoderDecoderTransformer(nn.Module):
    """Encoder-Decoder Transformer,类似 LibInvent 的架构"""
    def __init__(self, scaffold_vocab_size, decoration_vocab_size, d_model, nhead, 
                 num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length,
                 pos_dropout, trans_dropout):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            scaffold_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward,
            max_seq_length, pos_dropout, trans_dropout
        )
        
        self.decoder = TransformerDecoder(
            decoration_vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward,
            max_seq_length, pos_dropout, trans_dropout
        )
        
        self.d_model = d_model
    
    def forward(self, scaffold_seqs, decoration_seqs, con_token=None,
                scaffold_key_padding_mask=None, decoration_key_padding_mask=None,
                decoration_mask=None):
        """
        Args:
            scaffold_seqs: (batch, scaffold_seq_len)
            decoration_seqs: (batch, decoration_seq_len)  
            con_token: (batch, con_len) 条件 token
        """
        # Encode scaffolds
        memory = self.encoder(scaffold_seqs, con_token, scaffold_key_padding_mask)
        
        # Decode decorations with cross-attention to scaffolds
        logits = self.decoder(
            decoration_seqs, memory, con_token,
            tgt_mask=decoration_mask,
            tgt_key_padding_mask=decoration_key_padding_mask,
            memory_key_padding_mask=scaffold_key_padding_mask
        )
        
        return logits

    def encode(self, scaffold_seqs, con_token=None, scaffold_key_padding_mask=None):
        """仅编码 scaffold，用于推理时"""
        return self.encoder(scaffold_seqs, con_token, scaffold_key_padding_mask)
    
    def decode_step(self, decoration_seq, memory, con_token=None, decoration_mask=None):
        """单步解码，用于推理时"""
        return self.decoder(decoration_seq, memory, con_token, decoration_mask)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self, d_model, dropout=0.1, max_len=140):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)
    
class DecoratorTransformerRL:
    """Decorator Transformer RL 类，类似你现有的 transformer_RL"""
    def __init__(self, decorator_voc, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        
        self.model = EncoderDecoderTransformer(
            scaffold_vocab_size=decorator_voc.len_scaffold(),
            decoration_vocab_size=decorator_voc.len_decoration(),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            max_seq_length=max_seq_length,
            pos_dropout=pos_dropout,
            trans_dropout=trans_dropout
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.decorator_voc = decorator_voc
        self._nll_loss = nn.NLLLoss(ignore_index=0, reduction="none")
    
    def likelihood(self, scaffold_batch, decoration_batch):
        
        scaffold_seqs = scaffold_batch
        decoration_seqs= decoration_batch
        
        batch_size, decoration_seq_length = decoration_seqs.size()
        
        # 准备输入和目标
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.decorator_voc.decoration_vocabulary.vocab['GO']
        
        # 分离条件 token 和真实序列
        con_token = scaffold_seqs[:, :2]  # 前两个是条件 token
        scaffold_only = scaffold_seqs[:, 2:]  # scaffold 本身
        
        decoration_input = torch.cat((start_token, decoration_seqs[:, :-1]), 1)
        decoration_target = decoration_seqs[:, :].contiguous().view(-1)
        
        # 生成 mask
        decoration_mask = gen_nopeek_mask(decoration_input.shape[1])
        
        # Forward pass
        logits = self.model(scaffold_only, decoration_input, con_token, 
                           decoration_mask=decoration_mask)
        
        expected_shape = (batch_size, decoration_input.shape[1], self.decorator_voc.len_decoration())
    
        if logits.shape != expected_shape:
            print(f"Warning: Unexpected logits shape {logits.shape}, expected {expected_shape}")
            # 如果第1维是1，就压缩
            if len(logits.shape) == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
        # 计算损失
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        log_probs = criterion(logits.view(-1, self.decorator_voc.len_decoration()), 
                             decoration_target)
        
        mean_log_probs = log_probs.mean()
        log_probs_each_molecule = log_probs.view(batch_size, -1).sum(dim=1)
        
        return mean_log_probs, log_probs_each_molecule
    
    def sample(self, scaffold_batch, max_length=140):
        """
        采样 decorations + 计算 log probabilities（用于强化学习）
        返回：sequences, log_probs
        """
        scaffold_seqs = scaffold_batch
        batch_size = scaffold_seqs.size(0)
        
        # 分离条件 token 和 scaffold
        con_token = scaffold_seqs[:, :2]
        scaffold_only = scaffold_seqs[:, 2:]
        
        # 编码 scaffold 一次（避免重复编码）
        memory = self.model.encode(scaffold_only, con_token)
        
        # 初始化
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.decorator_voc.decoration_vocabulary.vocab['GO']
        
        sequences = start_token
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte().to(self.device)
        
        for step in range(max_length):
            # 关键：每步都重新计算完整的 logits（类似原代码的 sample_forward_model）
            logits = sample_forward_model_encoder_decoder(
                self.model, sequences, memory, con_token
            )
            
            # 取当前步的 logits
            logits_step = logits[:, step, :]
            
            # 计算概率
            prob = F.softmax(logits_step, dim=1)
            log_prob = F.log_softmax(logits_step, dim=1)  # 你提到的这一步！
            
            # 采样
            next_token = torch.multinomial(prob, 1)
            
            # 更新序列
            sequences = torch.cat((sequences, next_token), 1)
            
            # 累积 log probability（强化学习需要）
            log_probs += self._nll_loss(log_prob, next_token.view(-1))
            
            # 检查结束条件
            EOS_sampled = (next_token.view(-1) == self.decorator_voc.decoration_vocabulary.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            
            if torch.prod(finished) == 1:
                break
        
        return sequences[:, 1:].data, log_probs  # 去掉 start token，返回 log_probs
    
    def generate(self, scaffold_batch, max_length=140):
        """
        仅生成 decorations（用于推理展示）
        返回：sequences（不计算 log_probs，更高效）
        """
        scaffold_seqs = scaffold_batch
        batch_size = scaffold_seqs.size(0)
        
        con_token = scaffold_seqs[:, :2]
        scaffold_only = scaffold_seqs[:, 2:]
        
        memory = self.model.encode(scaffold_only, con_token)
        
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.decorator_voc.decoration_vocabulary.vocab['GO']
        
        sequences = start_token
        finished = torch.zeros(batch_size).byte().to(self.device)
        
        for step in range(max_length):
            # 同样需要每步重新计算 logits
            logits = sample_forward_model_encoder_decoder(
                self.model, sequences, memory, con_token
            )
            
            logits_step = logits[:, step, :]
            prob = F.softmax(logits_step, dim=1)
            # 注意：这里不计算 log_prob，节省计算
            
            next_token = torch.multinomial(prob, 1)
            sequences = torch.cat((sequences, next_token), 1)
            
            EOS_sampled = (next_token.view(-1) == self.decorator_voc.decoration_vocabulary.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            
            if torch.prod(finished) == 1:
                break
        
        return sequences[:, 1:].data  # 仅返回序列



def gen_nopeek_mask(length):
    """生成 look-ahead mask"""
    mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)
    
def sample_forward_model_encoder_decoder(model, decoration_seq, memory, con_token):
    """
    对应原代码的 sample_forward_model，但适配 Encoder-Decoder 架构
    每步都重新计算完整的 logits（因为 Transformer 需要完整上下文）
    """
    # 动态生成 mask（序列长度在每步增长）
    decoration_mask = gen_nopeek_mask(decoration_seq.shape[1])
    
    # 调用 decoder，传入完整的 memory 和当前序列
    output = model.decoder(
        decoration_seq, memory, con_token, 
        tgt_mask=decoration_mask
    )
    
    return output
