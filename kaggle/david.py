import torch
import io
import pandas as pd
import gc
import numpy as np
import transformers

class BertAoAHead(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.hidden = config.hidden_size
        self.q_projection_1 = nn.Linear(config.hidden_size, 512)
        self.c_projection_1 = nn.Linear(config.hidden_size, 512)

        self.q_projection_2 = nn.Linear(config.hidden_size, 512)
        self.c_projection_2 = nn.Linear(config.hidden_size, 512)

    def forward(self, hidden_states, attention_mask):
        q_state, c_state = hidden_states[:, 1:2, :].clone(), hidden_states
        c_state[:, 0:2, :] = 0

        # take the question tokens and do linear projection to Qx1024
        q_state_1 = self.q_projection_1(q_state)
        q_state_2 = self.q_projection_2(q_state)

        # do the same for the context tokens for Cx1024
        c_state_1 = self.c_projection_1(c_state)
        c_state_2 = self.c_projection_2(c_state)

        # print("Q_state", q_state_1.shape)
        # print("C_state", c_state_2.shape)

        # perform matrix multiplication of the context and transposed question
        # Cx1024 * 1024xQ = CxQ
        M1 = torch.matmul(c_state_1, torch.transpose(q_state_1, 1, 2))
        M2 = torch.matmul(c_state_2, torch.transpose(q_state_2, 1, 2))
        # print("M", M1.shape)

        # make sure that padding wont affect the softmax
        M1[attention_mask != 1, :] = -999999
        M2[attention_mask != 1, :] = -999999

        # Softmax over the transposed result i.e softmax(QxC) = QxC
        alpha_1 = nn.functional.softmax(torch.transpose(M1, 1, 2), dim=2)
        alpha_2 = nn.functional.softmax(torch.transpose(M2, 1, 2), dim=2)
        # print("alpha", alpha_1)

        # Softmax over the result i.e softmax(CxQ) = CxQ
        beta_1 = nn.functional.softmax(M1, dim=2)
        beta_2 = nn.functional.softmax(M2, dim=2)
        # print("beta", beta_1)

        # Average sum(beta_1) by queries and divide by length and unsqeze CxQ ->  Qx1
        beta_1 = (torch.sum(beta_1, axis=1) / beta_1.size(1)).unsqueeze(1)
        beta_2 = (torch.sum(beta_2, axis=1) / beta_2.size(1)).unsqueeze(1)
        # print("beta sum", beta_1.shape)
        # print("beta", beta_1)

        # CxQ * Qx1 => Cx1 squeeze => C
        s1 = torch.matmul(torch.transpose(alpha_1, 1, 2), torch.transpose(beta_1, 1, 2)).squeeze(2)
        s2 = torch.matmul(torch.transpose(alpha_2, 1, 2),torch.transpose(beta_2, 1, 2)).squeeze(2)
        return s1, s2

'''

sudo apt update        # Fetches the list of available updates
sudo apt upgrade       # Installs some updates; does not remove packages
sudo apt full-upgrade  # Installs updates; may also remove some packages, if needed
sudo apt autoremove    # Removes any old packages that are no longer needed

Precision in our case would be how many images tagged as Spa are really Spa. 
And Recall would be how many of the total Spa images have been tagged correctly.

Original Code Author [@dlibenzi](https://github.com/dlibenzi)
Complete Colab Example [link](https://colab.research.google.com/drive/1IvCxIg-Q_DlI7UNJuajpl4UZXNiW5jMg)
NB I have trimmed to what i felt was new to me as the idea, please refer to the colab
for the rest of the code blocks..

The code i tricky to understand if you have never worked with Files directly before; 
Here on top of file handling concepts, it's  in binary format, so you need to be little more careful
with regards to writing and loading data..

Personal Notes
---------------

# A binary file is considered to be just a sequence of bytes - none of them has any special meaning, 
# in the sense that a text-reader would interpret them..
# Basically binary files contain data and each individual byte can be an ascii character, an integer, a tensor etc. 
# It's just how to write data to the file and how you rad it back.

# The io.BytesIO inherits from io.BufferedReader class comes with functions like read(), write(), peek(), getvalue(). 
# Binary data and strings are different types, so a str must be encoded to binary using ascii, utf-8, or other.
# It is a general buffer of bytes that you can work with.

# Seeking a specific position in a file

    You can move to a specific position in file before reading or writing using seek(). 
    You can pass a single parameter to seek() and it will move to that position, relative to the beginning of the file.

# Seek can be called one of two ways:
    #   x.seek(offset)
    #   x.seek(offset, starting_point)
    #   The offset is interpreted relative to the position indicated by whence

    # starting_point can be 0, 1, or 2
    # 0 - Default. Offset relative to beginning of file
    # 1 - Start from the current position in the file
    # 2 - Start from the end of a file (will require a negative offset)

i = 16
i.to_bytes(1, byteorder='big', signed=True)    # b'\x10' is an escape sequence that describes the byte with that hexadecimal value.
i.to_bytes(4, byteorder='big', signed=True)    # b'\x00\x00\x00\x10'
i.to_bytes(4, byteorder='little', signed=True) # b'\x10\x00\x00\x00'
b'\x00\xff' # Two bytes values 0 and 255

# https://docs.python.org/3/library/stdtypes.html#int.from_bytes

    If byteorder is "big", the most significant byte is at the beginning of the byte array.
    If byteorder is "little", the most significant byte is at the end of the byte array.

    # int.from_bytes(b'\x00\x10', byteorder='big')    # 16
    # int.from_bytes(b'\x00\x10', byteorder='little') # 4096

How can I read last 10 bytes from a text file ?? [solve it yourself first]
    
    # here's my sol
    f.seek(0, 2) # last byte
    nbytes = f.tell()
    f.seek(nbytes-10)
    last_ten = f.read(10)

# f.tell() Returns the current stream position.
# f.read(size=k) Read and return up to size bytes

# https://stackoverflow.com/questions/42800250/difference-between-open-and-io-bytesio-in-binary-streams
    when you use open(), The data wont be kept in memory after it's written to the file (unless being kept by a name).
    when you consider io.BytesIO() , Which instead of writing the contents to a file, it's written to an in memory buffer. 
    In other words a chunk of RAM.
    # The key difference here is optimization and performance.

# save to a file
x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, 'tensor.pt')

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.save(x, buffer)

# The getvalue() function just takes the value from the Buffer as a String 
# and Return bytes containing the entire contents of the buffer.

'''

model="xlm-roberta-large"
batch_size=2
splits="8,16,32,64"
train_ds="train_dataset"
valid_ds="valid_dataset"

class FileDataset(object):
    def __init__(self, path):

        # open binary files to write to
        self._data_file = open(path + '.data', 'rb')
        self._index_file = open(path + '.index', 'rb')
        self._index_file.seek(0, 2)                # seek the last byte of the file
        self._index_size = self._index_file.tell() # size of the current stream position, Basically Get the file size...
        assert self._index_size % 8 == 0
        self._data_file.seek(0, 2)                 # seek the last byte of the file
        self._data_size = self._data_file.tell()   # size of the current stream position, Basically Get the file size...

    def read_sample(self, idx):
        '''
        The idea is basically that first you seek next 8 bytes from your current position (where ever you are in the file).
        After that you check that whether we can get the next's next whole 8 bytes as well; (so it's current + 16).
        If we can, then we calculate the next offset.
        '''
        index_offset = idx * 8
        assert index_offset < self._index_size
        self._index_file.seek(index_offset) # move to this position relative to the beginning of the file
        data_offset = int.from_bytes(self._index_file.read(8), byteorder='little') # read eight_bytes in little endian byte order
        if index_offset + 16 <= self._index_size:
            next_offset = int.from_bytes(self._index_file.read(8), byteorder='little') # next 8 bytes set-up for next seek
        else:
            next_offset = self._data_size  # else set it to end of the file
        self._data_file.seek(data_offset)  # move to this position relative to the beginning of the file wrt data file
        sample_data = self._data_file.read(next_offset - data_offset) # read these many
        return torch.load(io.BytesIO(sample_data)) # return as a tensor

    def get_num_samples(self):
        return self._index_size // 8


def bytes_from_file(fname, ck_sz=8192):
    '''
    simple func to stream bytes from the given file
    '''
    with open(fname, "rb") as f:
        while True:
            chunk = f.read(ck_sz)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break


def regular_encode_on_fly(texts, tokenizer, splits):
    '''
    pad only to the length that's needed to make the batch padded to same length
    aka bucketing
    '''
    max_len = max(len(x.split()) for x in texts)
    for l in splits:
      if l >= max_len:
        max_len = l
        break
    max_len = min(max_len, splits[-1])
    enc_di = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_masks=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=max_len,
    )
    return np.array(enc_di['input_ids']), np.array(enc_di["attention_mask"])


def indices_for_ordinal(ordinal, world_size, count, shuffle=True):
    '''
    it's a 3 line sampler; 
    ordinal denotes TPU_IDX
    world_size denotes how many TPU_CORES
    count denotes the #samples you have in your dataset file
    '''
    count = (count // world_size) * world_size
    indices = list(range(ordinal, count, world_size)) # start:end:step_size
    if shuffle:
        np.random.shuffle(indices)
    return indices


def prep_data(bs, df):
    '''
    basically the idea is to create the batches ourselves;
    NB we are using dynamic padding here (splits variable);
    '''
    sentences = df['comment_text'].astype(str).values
    sort_idx = np.argsort(np.array([len(x.split()) for x in sentences]))
    sentences = sentences[sort_idx]
    targets = df['toxic'].values[sort_idx]
    num_samples = (len(sentences) // bs) * bs
    sentences = sentences[: num_samples]
    targets = targets[: num_samples]
    return sentences.reshape(len(sentences) // bs, bs), targets.reshape(len(targets) // bs, bs)


def write_sample(s, data_file, index_file):
    bio = io.BytesIO() # get a buffer
    torch.save(s, bio) # save it to the buffer
    offset = data_file.tell() # [int] what's the current position of the "data_file" stream
    index_file.write((offset).to_bytes(8, byteorder='little')) # write the index for this tensor batch
    data_file.write(bio.getvalue())


def create_dataset(df, tokenizer, batch_size, splits, path):
    x, y = prep_data(batch_size, df) # grab the batches (raw-text, targets)
    xt = [torch.tensor(regular_encode_on_fly(t, tokenizer, splits)) for t in x] # converting them to tokens; each batch is dynamically padded
    yt = [torch.tensor(t, dtype=torch.float) for t in y] # targets
    with open(path + '.data', 'wb') as data_file:
        with open(path + '.index', 'wb') as index_file:
            for s in zip(xt, yt):
                # since we are using zip, so it's packing the items from x and y respectively together;
                write_sample(s, data_file, index_file)


def generate_index():
    global splits
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(model)

    train1 = pd.read_csv(
        './jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv',
        usecols=["comment_text", "toxic"], 
        nrows = 8
    )

    all_train = train1[['comment_text', 'toxic']]

    del train1
    gc.collect(); gc.collect();

    all_train = all_train.sample((all_train.shape[0]//batch_size)*batch_size)
    
    print('DF:', all_train.shape,)

    splits = sorted([int(x) for x in splits.split(',')])
    create_dataset(all_train, tokenizer, batch_size, splits, train_ds)


if __name__== "__main__":
    generate_index()