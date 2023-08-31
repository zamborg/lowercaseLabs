# similarity module -- contains class architecture AND training code.
from transformers import GPT2Model, GPT2Tokenizer
from transformers.activations import NewGELUActivation
import torch
import torch.nn as nn
from collections import OrderedDict
import os
import glob
import random

# os.chdir("./catchAll")

class Similarity(nn.Module):
    # TODO: refactor this should probably be two modules that I can compose seperately because training is getting annoying like this. 

    weight_init = nn.init.kaiming_normal_
    bias_init = lambda x : torch.rand_like(x)*1e-2

    def __init__(self, num_layers=4) -> None:
        """
        Similarity model -- placeholder for future class topology
        """
        super().__init__()
        self.device = 'cpu'
        self.gpt = GPT2Model.from_pretrained("./models/gpt2/")
        self.tokenizer = GPT2Tokenizer.from_pretrained("./models/gpt2/")

        self.ff = self._initialize_ff(
            [(self.gpt.embed_dim, self.gpt.embed_dim) for i in range(num_layers)]
        )

    def forward(self, x):
        """
        X is a tensor of shape (B, )
        """

    def inference(self, text):
        tokens, _ = self.tokenizer(text).values()
        with torch.no_grad():
            gpt_out = self.gpt(torch.LongTensor([tokens])).last_hidden_state # get the last hidden state of our gpt model
            # pool the last hidden state with avg pooling for the sequence length
        return self.ff(gpt_out.mean(1)) # avg-pool the sequence input.

    def infer(self, text):
        with torch.no_grad():
            return self.forward(text)

    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            m.weight = cls.weight_init(m.weight)
            m.bias.data = cls.bias_init(m.bias.data)
    

    @classmethod
    def _initialize_ff(cls, dims):
        """
        dims is a list of tuples [(700, 700), (700,900)]
        dims tuple corresponds to (in_dim, out_dim), out_dim[i] must match in_dim[i+1]
        """
        parameters = OrderedDict()
        for i, dim in enumerate(dims):
            #TODO: refactor this is disgusting
            parameters[f"Layer_{i}"] = nn.Linear(*dim)
            parameters[f"GeLu_{i}"] = NewGELUActivation()

        nn.Sequential(parameters).apply(cls._init_weights)

class NoteLoader():
    """
    Class to load and split and handle all data functions for the notes for training
    """
    def __init__(self, dir="./notesExport/") -> None:
        """
        Create a NotesLoader with filepath
        """
        self.Notes = self._load_notes(dir)
        self.batch_hist = []

    # @cached_property
    def __len__(self):
        return len(self.Notes)
    
    @classmethod
    def _load_notes(cls, dir):
        files = glob.glob(os.path.join(dir, "*.txt")) # get all txts
        return [open(f, "r").read() for f in files]

    def get_batch(self, batch_size):
        yield [self.Notes[random.randint(0,len(self)-1)] for i in range(batch_size)] # this is fucking disgusting I gotta just use rand.rand huh

    
s = Similarity()
n = NoteLoader()