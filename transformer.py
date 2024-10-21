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
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the
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
    y = None
    weights_softmax = None
    
    similarities = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(M))

    if mask is not None:
        N, K, K = mask.shape 
        for n in range(N): 
            for k1 in range(K): 
                for k2 in range(K):
                    similarities[n, k1, k2] = -1e9
        
    weights_softmax = torch.softmax(similarities, dim = 2) 
    y = torch.bmm(weights_softmax, value) 

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
        self.q = None 
        self.k = None 
        self.v = None 
        self.weights_softmax = None

        self.linear_transformation1 = nn.Linear(dim_in, dim_q)
        self.linear_transformation2 = nn.Linear(dim_in, dim_q)
        self.linear_transformation3 = nn.Linear(dim_in, dim_v)

        c1 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_q))
        c2 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_q))
        c3 = torch.sqrt(torch.tensor(6.) / (dim_in + dim_v))

        self.linear_transformation1.weights = torch.FloatTensor(dim_in, dim_q).uniform_(0, c1)
        self.linear_transformation2.weights = torch.FloatTensor(dim_in, dim_q).uniform_(0, c2)
        self.linear_transformation3.weights = torch.FloatTensor(dim_in, dim_v).uniform_(0, c3)


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
        self.q = self.linear_transformation1(query)
        self.k = self.linear_transformation2(key)
        self.v = self.linear_transformation3(value)
        
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

        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.

        """

        dim_q = dim_out
        dim_k = dim_out
        dim_v = dim_out
        
        c = torch.sqrt(torch.tensor(6.) / (num_heads*dim_out + dim_in))

        self.multiple_self_attention = nn.ModuleList([SelfAttention(dim_in, dim_q, dim_v) for i in range(num_heads)])
        self.linear_transformation = nn.Linear(num_heads*dim_out, dim_in)
        self.linear_transformation.weights = torch.FloatTensor(num_heads*dim_out, dim_in).uniform_(0, c)
        
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
        for self_attention in self.multiple_self_attention:
            x = self_attention.forward(query, key, value)        
            output.append(x)
        
        output_concat = torch.concat(output, dim = 2)
        y = self.linear_transformation.forward(output_concat)

        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BathcNorm ,it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length and M is the sequence length embedding. We 
        compute the  mean with shape (N, K) and standard deviation with shape (N, K) 
        and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value

        """

        self.epsilon = epsilon        
        self.gamma = torch.ones(emb_dim)
        self.beta = torch.zeros(emb_dim)
        self.emb_dim = emb_dim
        
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
        N = x.shape[0]
        y = torch.zeros_like(x)

        for n in range(N): 
            x_n = x[n]
            mean = torch.mean(x_n, dim = -1)
            var = torch.sum((x_n - mean)**2)/self.emb_dim
            std = torch.sqrt(var)
            scaled = (x_n - mean) / std
            y[n, :] = self.gamma * scaled + self.beta

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
   
        self.linear1.weights = torch.FloatTensor(inp_dim, hidden_dim_feedforward).uniform_(0, 1)
        self.linear2.weights = torch.FloatTensor(hidden_dim_feedforward, inp_dim).uniform_(0, 1)
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
            raise ValueError(f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value.""")
        
        self.multihead = MultiHeadAttention(num_heads, emb_dim, emb_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.dropout_ = nn.Dropout(dropout)

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

        out_multihead = self.multihead(x, x, x)
        out_multihead_residual = out_multihead + residual

        out_norm1 = self.norm1(out_multihead_residual)
        out_dropout1 = self.dropout_(out_norm1)

        out_mlp = self.feedforward(out_dropout1)
        out_norm2 = self.norm2(out_mlp + out_dropout1)
        y = self.dropout_(out_norm2)

        return y


