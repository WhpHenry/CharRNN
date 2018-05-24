import copy
import pickle
import numpy as np

def batch_generate(arr, n_seqs, n_steps) -> (np.array, np.array):
    # cut arr as matrix (n_batches * n_steps, n_seqs)
    # colums size - n_seqs
    # row size - n_batches * n_steps
    # remove more other by arr cut arr[: n_batches * batch_size]
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    arr = arr[: n_batches * batch_size]
    arr = arr.reshape((n_seqs, -1))

    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n+n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:,-1] = x[:,1:], x[:,0]
            yield x, y

class TextConvert():
    def __init__(self, text=None, max_vocab=5000, fname=None):
        if fname:
            with open(fname, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            word_count = {w:0 for w in set(text)}
            for w in text:
                word_count[w] += 1
            word_count_list = []
            for k in word_count:
                word_count_list.append((k, word_count[k]))
            word_count_list.sort(key=lambda w:w[1], reverse=True)
            if len(word_count_list) > max_vocab:
                word_count_list = word_count_list[:max_vocab] 
            self.vocab = [w[0] for w in word_count_list]
        e = enumerate(self.vocab)
        self.int2word_dict = dict(e)
        self.word2int_dict = {c:i for i, c in e}
    
    @property
    def vocab_size(self) -> int: 
        # @property defines 'vocab_size' is read-only 
        return len(self.vocab) + 1
    
    def word2int(self, word) -> int:
        return self.word2int_dict[word] if word in self.word2int_dict else len(self.vocab)
    
    def int2word(self,idx) -> str:
        if idx < len(self.vocab):
            return self.int2word_dict[idx] 
        elif idx == len(self.vocab):
            return '<unk>'
        else:
            raise Exception('index-{} non-existent'.format(idx))
    
    def text2arr(self, text):
        arr = []
        for w in text:
            arr.append(self.word2int(w))
        return np.array(arr)
    
    def arr2text(self, arr):
        words = []
        for index in arr:
            words.append(self.int2word(index))
        return "".join(words)
    
    def save_vocab(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.vocab, f)
         
