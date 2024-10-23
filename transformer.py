import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

def generate_token_dict(vocab):
    """
    The function creates a hash map from the elements in the vocabulary to
    to a unique positive integer value.

    args:
        vocab: This is a 1D list of strings containing all the items in the vocab

    Returns:
        token_dict: a python dictionary with key as the string item in the vocab
            and value as a unique integer value
    """
    token_dict = {}
    i = 0
    for word in vocab: 
        token_dict[word] = i
        i+=1
    return token_dict


def prepocess_input_sequence(input_str: str, token_dict: dict, spc_tokens: list) -> list:
    """
    The goal of this fucntion is to convert an input string into a list of positive
    integers that will enable us to process the string using neural nets further. We
    will use the dictionary made in the previous function to map the elements in the
    string to a unique value. Keep in mind that we assign a value for each integer
    present in the input sequence. For example, for a number present in the input
    sequence "33", you should break it down to a list of digits,
    ['0', '3'] and assign it to a corresponding value in the token_dict.

    args:
        input_str: A single string in the input data
                 e.g.: "BOS POSITIVE 0333 add POSITIVE 0696 EOS"

        token_dict: The token dictionary having key as elements in the string and
            value as a unique positive integer. This is generated  using
            generate_token_dict fucntion

        spc_tokens: The special tokens apart from digits.
    Returns:
        out_tokens: a list of integers corresponding to the input string


    """
    input_words = input_str.split(" ")
    out = []
    for word in input_words:
          is_word_digit = word.isnumeric()
          if is_word_digit == False:
              val = token_dict[word]
              out.append(val)
          else:     
              for digit in word: 
                  val = token_dict[digit]
                  out.append(val)                   
    return out


def scaled_dot_product(query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values


    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values

        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    _, _, M = query.shape    
    similarities = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(M))

    if mask is not None:
        N, K, K = mask.shape 
        for n in range(N): 
            for k1 in range(K): 
                for k2 in range(K):
                    if mask[n, k1, k2] == True:
                        similarities[n, k1, k2] = -1e9
        
    weights_softmax = torch.softmax(similarities, dim = 2) # N, K, K
    y = torch.bmm(weights_softmax, value) # N, K, M
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and key vector
            dim_v: an int value for output dimension for value vectors

        """
        self.linear1 = nn.Linear(dim_in, dim_q)
        self.linear2 = nn.Linear(dim_in, dim_q)
        self.linear3 = nn.Linear(dim_in, dim_v)

        c1 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_q))
        c2 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_q))
        c3 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_v))

        nn.init.uniform_(self.linear1.weight, -c1, c1)
        nn.init.uniform_(self.linear2.weight, -c2, c2)
        nn.init.uniform_(self.linear3.weight, -c3, c3)   
            
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (N, K, K)
        return:
            y: Tensor of shape (N, K, dim_v)
        """

        self.q = self.linear1(query)
        self.k = self.linear2(key)
        self.v = self.linear3(value)
        
        y, self.weights_softmax = scaled_dot_product(self.q, self.k, self.v)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the same input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the sequence length and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block

        NOTE: Here, when we say dimension, we mean the dimension of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.

        """

        dim_q = dim_out
        dim_k = dim_out
        dim_v = dim_out

        self.heads = nn.ModuleList(
          [SelfAttention(dim_in, dim_q, dim_v) for i in range(num_heads)]
          )
        self.linear = nn.Linear(num_heads*dim_out, dim_in)
        
        c = torch.sqrt(torch.tensor(6.) / (num_heads*dim_out + dim_in))
        nn.init.uniform_(self.linear.weight, -c, c)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            key: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            value: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            mask: Tensor of shape (N, K, K) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

        returns:
            y: Tensor of shape (N, K, M)
        """

        output = []
        for self_attention in self.heads:
            x = self_attention.forward(query, key, value)        
            output.append(x)
        
        output_concat = torch.concat(output, dim = 2)
        y = self.linear.forward(output_concat)

        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BatchNorm, it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length and M is the sequence length embedding. We 
        compute the mean with shape (N, K) and standard deviation with shape (N, K) 
        and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value
        """

        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim)) 
        self.beta = nn.Parameter(torch.zeros(emb_dim)) 

    def forward(self, x: Tensor):
        """
        An implementation of the forward pass of the Layer Normalization.

        args:
            x: a Tensor of shape (N, K, M) or (N, K) where N is the batch size, K
                is the sequence length and M is the embedding dimension

        returns:
            y: a Tensor of shape (N, K, M) or (N, K) after applying layer
                normalization
        """

        mean = x.mean(dim=-1, keepdim=True)  # Shape: (N, K, 1)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # Shape: (N, K, 1)
        std = torch.sqrt(var + self.epsilon)  # Shape: (N, K, 1)
        x_normalized = (x - mean) / std  # Shape: (N, K, M)
        y = self.gamma.unsqueeze(0).unsqueeze(0) * x_normalized + self.beta.unsqueeze(0).unsqueeze(0)
        
        return y 


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu - linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """
        
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)

        c1 = torch.sqrt(torch.tensor(6.) / (inp_dim + hidden_dim_feedforward))
        c2 = torch.sqrt(torch.tensor(6.) / (hidden_dim_feedforward + inp_dim))

        nn.init.uniform_(self.linear1.weight, -c1, c1)
        nn.init.uniform_(self.linear2.weight, -c2, c2)


    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        
        return y


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()
        """
        This class implements the encoder block for the Transformer model, the 
        original paper used 6 of these blocks sequentially to train the final model. 
        Here, we will first initialize the required layers using the building  
        blocks we have already  implemented, and then finally write the forward     
        pass using these initialized layers, residual connections and dropouts.        
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of four components:
        
        1. MultiHead Attention
        2. FeedForward layer
        3. Residual connections after MultiHead Attention and feedforward layer
        4. LayerNorm
        
        The architecture is as follows:
        
        inp - multi_head_attention - out1 - layer_norm(out1 + inp) - dropout - out2 \ 
        - feedforward - out3 - layer_norm(out3 + out2) - dropout - out
        
        Here, inp is input of the MultiHead Attention of shape (N, K, M), out1, 
        out2 and out3 are the outputs of the corresponding layers and we add these 
        outputs to their respective inputs for implementing residual connections.

        args:
            num_heads: int value specifying the number of heads in the
                MultiHeadAttention block of the encoder

            emb_dim: int value specifying the embedding dimension of the input
                sequence

            feedforward_dim: int value specifying the number of hidden units in the 
                FeedForward layer of Transformer

            dropout: float value specifying the dropout value


        """

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        
        self.MultiHeadBlock = MultiHeadAttention(num_heads, emb_dim, int(emb_dim/num_heads))
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        An implementation of the forward pass of the EncoderBlock of the
        Transformer model.
        args:
            x: a Tensor of shape (N, K, M) as input sequence
        returns:
            y: a Tensor of shape (N, K, M) as the output of the forward pass
        """       
        residual = x

        out_multihead = self.MultiHeadBlock.forward(x, x, x)
        out_multihead_residual = out_multihead + residual

        out_norm1 = self.norm1.forward(out_multihead_residual)
        out_dropout1 = self.dropout(out_norm1)

        out_mlp = self.feedforward.forward(out_dropout1)
        out_norm2 = self.norm2.forward(out_mlp + out_dropout1)
        y = self.dropout(out_norm2)
        
        return y


def get_subsequent_mask(seq):
    """
    An implementation of the decoder self attention mask. This will be used to
    mask the target sequence while training the model. The input shape here is
    (N, K) where N is the batch size and K is the sequence length.

    args:
        seq: a tensor of shape (N, K) where N is the batch size and K is the
             length of the sequence
    return:
        mask: a tensor of shape (N, K, K) where N is the batch size and K is the
              length of the sequence

    Given a sequence of length K, we want to mask the weights inside the function
    `self_attention_no_loop_batch` so that it prohibits the decoder to look ahead
    in the future
    """

    N, K = seq.shape

    mask = torch.full((N, K, K), False)

    for n in range(N): 
        for k in range(K):
            for j in range(k):
                mask[n, j, k] = True

    return mask


class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        """
        The function implements the DecoderBlock for the Transformer model. In the 
        class we learned about encoder only model that can be used for tasks like 
        sequence classification but for more complicated tasks like sequence to 
        sequence we need a decoder network that can transformt the output of the 
        encoder to a target sequence. This kind of architecture is important in 
        tasks like language translation where we have a sequence as input and a 
        sequence as output. 
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of 5 components:   
        
        1. Masked MultiHead Attention
        2. MultiHead Attention
        3. FeedForward layer
        4. Residual connections after MultiHead Attention and feedforward layer
        5. LayerNorm        
        
        The Masked MultiHead Attention takes the target, masks it as per the 
        function get_subsequent_mask and then gives the output as per the MultiHead  
        Attention layer. Further, another Multihead Attention block here takes the  
        encoder output and the output from Masked Multihead Attention layer giving  
        the output that helps the model create interaction between input and 
        targets. As this block helps in interaction between the input and target, it  
        is also sometimes called the cross attention.

        The architecture is as follows:
        
        inp - masked_multi_head_attention - out1 - layer_norm(inp + out1) - \
        dropout - (out2 and enc_out) -  multi_head_attention - out3 - \
        layer_norm(out3 + out2) - dropout - out4 - feed_forward - out5 - \
        layer_norm(out5 + out4) - dropout - out
        
        Here, out1, out2, out3, out4, out5 are the corresponding outputs for the 
        layers, enc_out is the encoder output and we add these outputs to their  
        respective inputs for implementing residual connections.
        
        args:
            num_heads: int value representing number of heads

            emb_dim: int value representing embedding dimension

            feedforward_dim: int representing hidden layers in the feed forward 
                model

            dropout: float representing the dropout value
        """
        
        self.attention_self = MultiHeadAttention(num_heads, emb_dim, int(emb_dim/num_heads))
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, int(emb_dim/num_heads))
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.norm3 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_inp: Tensor, enc_out: Tensor, mask: Tensor = None) -> Tensor:

        """
        args:
            dec_inp: a Tensor of shape (N, K, M)
            enc_inp: a Tensor of shape (N, K, M)
            mask: a Tensor of shape (N, K, K)

        This function will handle the forward pass of the Decoder block. It takes
        in input as enc_inp which is the encoder output and a tensor dec_inp which
        is the target sequence shifted by one in case of training and an initial
        token "BOS" during inference
        """

        out1 = self.attention_self.forward(dec_inp, dec_inp, dec_inp, mask)
        out1_norm = self.norm1.forward(dec_inp + out1)
        out1_norm = self.dropout(out1_norm)

        out2 = self.attention_cross.forward(out1_norm, enc_out, enc_out)
        out2_norm = self.norm2.forward(out2 + out1_norm)
        out2_norm = self.dropout(out2_norm)

        out3 = self.feed_forward.forward(out2_norm)
        out3_norm = self.norm3.forward(out3 + out2_norm)
        y = self.dropout(out3_norm)

        return y

class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        The class encapsulates the implementation of the final Encoder that use
        multiple EncoderBlock layers.

        args:
            num_heads: int representing number of heads to be used in the
                EncoderBlock
            emb_dim: int repreesenting embedding dimension for the Transformer
                model
            feedforward_dim: int representing hidden layer dimension for the
                feed forward block

        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        vocab_len: int,
    ):
        super().__init__()
        """
        The Decoder takes the input from the encoder and the target
        sequence to generate the final sequence for the output. We
        first pass the input through stacked DecoderBlocks and then
        project the output to vocab_len which is required to get the
        actual sequence.
        
        args:
            num_heads: Int representing number of heads in the MultiheadAttention
            for Transformer
            emb_dim: int representing the embedding dimension
            of the sequence
            feedforward_dim: hidden layers in the feed forward block
            num_layers: int representing the number of DecoderBlock in Decoder
            dropout: float representing the dropout in each DecoderBlock
            vocab_len: length of the vocabulary
        """

        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):

        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


def position_encoding_simple(K: int, M: int) -> Tensor:
    """
    An implementation of the simple positional encoding using uniform intervals
    for a sequence.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)
    """    
    y = torch.zeros(size = (1, K, M))
    for n in range(K): 
        y[0,n,:] = n/K
    
    return y


def position_encoding_sinusoid(K: int, M: int) -> Tensor:

    """
    An implementation of the sinousoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)

    """
    
    y = torch.zeros(1, K, M)
    for p in range(K): 
        i = 0
        while (2*i + 1 <= M):
            a = (2 * i)/M
            a = int(a)
            sincos_input = torch.tensor(p / 10000**a)
            y[0, p, 2*i] = torch.sin(sincos_input)
            if (2*i + 1 < M):
                y[0, p, 2*i+1] = torch.cos(sincos_input)
            i += 1
    
    return y


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()

        """
        The class implements Transformer model with encoder and decoder. The input
        to the model is a tensor of shape (N, K) and the output is a tensor of shape
        (N*O, V). Here, N is the batch size, K is the input sequence length, O is  
        the output sequence length and V is the Vocabulary size. The input is passed  
        through shared nn.Embedding layer and then added to input positonal 
        encodings. Similarily, the target is passed through the same nn.Embedding
        layer and added to the target positional encodings. The only difference
        is that we take last but one  value in the target. The summed 
        inputs(look at the code for detials) are then sent through the encoder and  
        decoder blocks  to get the  final output.
        args:
            num_heads: int representing number of heads to be used in Encoder
                       and decoder
            emb_dim: int representing embedding dimension of the Transformer
            dim_feedforward: int representing number of hidden layers in the
                             Encoder and decoder
            dropout: a float representing probability for dropout layer
            num_enc_layers: int representing number of encoder blocks
            num_dec_layers: int representing number of decoder blocks

        """
        
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)
        self.linear = nn.Linear(emb_dim, vocab_len)
        
        self.encoder = Encoder(num_heads, 
                               emb_dim, 
                               feedforward_dim, 
                               num_enc_layers, 
                               dropout)
        
        self.decoder = Decoder(num_heads,
                               emb_dim,
                               feedforward_dim,
                               num_dec_layers,
                               dropout,
                               vocab_len)

    def forward(self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor) -> Tensor:

        """

        An implementation of the forward pass of the Transformer.

        args:
            ques_b: Tensor of shape (N, K) that consists of input sequence of
                the arithmetic expression
            ques_pos: Tensor of shape (N, K, M) that consists of positional
                encodings of the input sequence
            ans_b: Tensor of shape (N, K) that consists of target sequence
                of arithmetic expression
            ans_pos: Tensor of shape (N, K, M) that consists of positonal
                encodings of the target sequence

        returns:
            dec_out: Tensor of shape (N*O, M) where O is the size of
                the target sequence.
        """
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]

        enc_out = self.encoder(q_emb_inp)
        mask = get_subsequent_mask(ans_b)
        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        dec_out = dec_out.view(-1, dec_out.size(2)) 
        return dec_out


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):

        """
        The class implements the dataloader that will be used for the toy dataset.

        args:
            input_seqs: A list of input strings
            target_seqs: A list of output strings
            convert_str_to_tokens: Dictionary to convert input string to tokens
            special_tokens: A list of strings
            emb_dim: embedding dimension of the transformer
            pos_encode: A function to compute positional encoding for the data
        """

        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(inp, 
                                        self.convert_str_to_tokens, 
                                        self.special_tokens)

    def __getitem__(self, idx):
        """
        The core fucntion to get element with index idx in the data.
        args:
            idx: index of the element that we need to extract from the data
        returns:
            preprocess_inp: A 1D tensor of length K, where K is the input sequence
                length
            inp_pos_enc: A tensor of shape (K, M), where K is the sequence length
                and M is the embedding dimension
            preprocess_out: A 1D tensor of length O, where O is the output
                sequence length
            out_pos_enc: A tensor of shape (O, M), where O is the sequence length
                and M is the embedding dimension
        """

        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)

        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def CrossEntropyLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    loss = F.cross_entropy(pred, ground, reduction="sum")
    return loss
