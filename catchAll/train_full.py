from Similarity import *
import torch.nn as nn
import torch
import os
os.chdir("./catchAll")

# TODO:
# 1. Need to figure out if I ONLY need the last hiddenstate from gpt2 (might be the best practice)
# 2. Either way need to write the transform_tensor so I can get a batch input for the new forward func of my model
# 3. Need to write the loss function (thats super easy should be dot product loss sigmoid bounded between 0-1)
# 4. Put together the training loop!
# 0. Figure out what things i don't want to push to git lol 

# convert to an @attr.s decorated class
class TrainConfig():
    batch_size = 32
    num_iters = 10
    adam_params = {
        "lr":1e-3,
    }

model = Similarity()
notes = NoteLoader()

loss = nn.BCELoss()

optimizer = torch.optim.Adam(model.ff.parameters(), *TrainConfig.adam_params)

def transform_tensor(input_data):
    # we'll use the model for our tokenizer anyway:
    pass
    

def loss_func(x, y):
    pass


for iteration in TrainConfig.num_iters:
    # get batch_size samples:
    data = notes.get_batch(TrainConfig.batch_size)
    