from torch.nn.utils import weight_norm
from fastai.text import *
#from .text_tcn import *

class Chop1d(nn.Module):
    '''
    module to remove excess padding
    '''
    def __init__(self,pad):
        super(Chop1d,self).__init__()
        self.pad = pad
        
    def forward(self,x):
        return x[:,:,:-self.pad].contiguous()
    
class BasicTempConvBlock(nn.Module):
    '''
    basic temporal convolution (1-D causal) block
    '''
    def __init__(self,n_inputs,n_outputs,kernel_size,stride,dilation,padding, dropout=0.2):
        super(BasicTempConvBlock,self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs,n_outputs,kernel_size,stride=stride,
                                           padding=padding,dilation=dilation,bias=False))
        self.chop1 = Chop1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout,inplace=True)
        
        self.conv2 = weight_norm(nn.Conv1d(n_outputs,n_outputs,kernel_size,stride=stride,
                                           padding=padding,dilation=dilation,bias=False))
        self.chop2 = Chop1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout,inplace=True)
        
        self.downsample = nn.Conv1d(n_inputs,n_outputs,1,bias=False) if n_inputs != n_outputs else None #one by one conv
        self.relu3 = nn.ReLU(inplace=True)
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0,0.01)
        self.conv2.weight.data.normal_(0,0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0,0.01)
    
    def forward(self,x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.chop1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chop2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = residual + out
        out = self.relu3(out)
        return out

class TemporalConvModule(nn.Module):
    def __init__(self,emb_size,nhid_list,kernel_size=3,dropout=0.2):
        """
        emb_size:       int, Embedding dimension 
        nhid_list:      list, number of hidden units in each TCN module
        kernel_size:    int, size of convolutional kernel
        dropout:        float, dropout after temporal conv block
        """
        super(TemporalConvModule,self).__init__()
        layers = []
        num_levels = len(nhid_list)
        for i in range(num_levels):
            dilation_factor = 2**i
            in_channels = emb_size if i == 0 else nhid_list[i-1]
            out_channels = nhid_list[i]
            layers += [BasicTempConvBlock(in_channels, out_channels,kernel_size, stride=1,dilation=dilation_factor,
                                         padding=(kernel_size-1)*dilation_factor,dropout=dropout)]
        self.tcn_module = nn.Sequential(*layers)
        #self.tcn_module = nn.ModuleList(layers)
        
    def forward(self,x):
        """
        output = x
        for layer in self.tcn_module:
            output = layer(output)
        return output
        """
        return self.tcn_module(x)
    

class TCN_Encoder(nn.Module):
    """
    Temporal Conv Encoder Model 
    """
    def __init__(self,emb_size,vocab_size,nhid_list,kernel_size=3,dropoute=0.1,dropoutc=0.2):
        """
        emb_size:       int, Embedding dimension 
        vocab_size:     int, Number of words in vocabulary, i.e vocabulary size
        nhid_list:      list, number of hidden units in each TCN module
        kernel_size:    int, size of convolutional kernel
        dropoutc:       float, dropout for temporal conv block
        dropoute:       float, dropout following embedding layer
        """
        super().__init__()
        self.vocab_size,self.emb_size = vocab_size,emb_size
        
        self.encoder = nn.Embedding(self.vocab_size,self.emb_size)
        self.dropoute = nn.Dropout(dropoute,inplace=True)
        self.tcn_module = TemporalConvModule(self.emb_size,nhid_list,kernel_size,dropoutc)
        self.encoder.weight.data.normal_(0,0.01)
        
    def forward(self,x):
        #raw_output,outputs = x
        output = self.encoder(x)
        output = self.dropoute(output)
        output = self.tcn_module(output.transpose(1,2))
        output = output.transpose(1,2)
        return output
    

class TCN_Decoder(nn.Module):
    """
    Temporal Conv Decoder Model 
    """
    def __init__(self,emb_size,vocab_size,nhidden,dropoutd=0.1,tied_encoder=None):
        """
        emb_size:       int, Embedding dimension 
        vocab_size:     int, Number of words in vocabulary, i.e vocabulary size
        nhidden:        int, number of hidden units in decoder (linear) layer
        dropoutd:       float, dropout for decoder
        """
        super().__init__()
        self.vocab_size,self.nhidden = vocab_size,nhidden

        self.decoder = nn.Linear(self.nhidden,self.vocab_size)
        self.dropout = nn.Dropout(dropoutd,inplace=True)

        if tied_encoder:
            if nhidden != emb_size:
                raise ValueError ('Dimensions do not must match.')

            self.decoder.weight = tied_encoder.weight
            print("Tied weights")
        
        self.decoder.weight.data.normal_(0,0.01)
        self.decoder.bias.data.fill_(0)
        
    def forward(self,x):
        output = self.dropout(x)
        output = self.decoder(output)
        result = output.contiguous().view(-1,self.vocab_size)  #output shape: (samples,vocab_size), where samples = batch size * seq_len
        return result #,raw_output,outputs

    
def get_tcn_model(emb_size,vocab_size,nhid_list,kernel_size=3,dropoute=0.1,dropoutc=0.2,dropoutd=0.1,tie_weights=True):
    """
    Build TCN model
    """
    tcn_encoder = TCN_Encoder(emb_size,vocab_size,nhid_list,kernel_size, dropoute,dropoutc)
    tie_enc = tcn_encoder.encoder if tie_weights else None
    tcn_decoder = TCN_Decoder(emb_size,vocab_size,nhid_list[-1],dropoutd,tied_encoder=tie_enc)
    
    return nn.Sequential(tcn_encoder,tcn_decoder)

class TCNLanguageModel(BasicModel):
    """ Necessary to define get_layer_groups() for weight-tying to work"""
    def get_layer_groups(self):
        layer_groups = list(self.model[0].tcn_module.tcn_module)
        return layer_groups + [self.model[1]]
    
class TCNLearner(Learner):
    """
    Copied from RNN_Learner
    """
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy

    def save_encoder(self, name): 
        save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): 
        load_model(self.model[0], self.get_model_path(name))
        
        
class TCNLanguageModelData():
    """
    Loads data into model and initializes model
    """
    def __init__(self,path,vocab_sz,train_dl,valid_dl,test_dl=None):
        self.path,self.vocab_sz = path,vocab_sz
        self.trn_dl,self.val_dl,self.test_dl = train_dl,valid_dl,test_dl
        
    def get_model(self,opt_fn,emb_sz,nhid_list,**kwargs):
        m = get_tcn_model(emb_sz,self.vocab_sz,nhid_list,**kwargs) 
        model = TCNLanguageModel(to_gpu(m))
        return TCNLearner(self,model,opt_fn=opt_fn)
    
class TCNLanguageModelDataLoader():
    def __init__(self,data_ids,bs,seq_len):
        self.bs,self.seq_len = bs,seq_len
        self.data = self.batchify(data_ids)  #convert to batches x sequence length
        self.i,self.iter = 0,0               #initialize counters: i (index )and iter (iteration)
        self.n = self.data.size(1)           #length of sequence per batch
    
    def __len__(self): return self.n // self.seq_len #number of iterations
    
    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter < len(self): #loop through data
            result = self.get_batch(self.i)   #get batch data
            self.iter += 1
            self.i += self.seq_len
            yield result
            
    def batchify(self,data):
        nbatches = data.shape[0]//self.bs         #calculate number of batches
        data = np.array(data[:nbatches*self.bs])  #grab data equal to the number of batches x batch size
        data = data.reshape(self.bs,-1)           #reshape into batches x sequence
        return T(data)                            #convert to pytorch tensor
    
    def get_batch(self,i):
        source = self.data
        #set to seq_len, except towards the end of data sequence which may be shorter than seq_len
        seq_len = min(self.seq_len, source.size(1)-1-i) 
        data = source[:,i:i+seq_len]              #grab seq_len at a time (shape: (batch_size,seq_len))
        target = source[:,i+1:i+1+seq_len].contiguous().view(-1) #flatten, shape=(samples,), where len(samples)= batch size * seq_len
        return data,target