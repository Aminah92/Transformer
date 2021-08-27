#Config
dmodel = 512 
h = 8  #number of heads
N = 6  #number of layers in decoder and encoder

en = "..../english.txt"
fr = ".../french.txt"

src_vocab = len(en)
trg_vocab = len(fr)


batch_size = 4
epoch = 10
lr = 0.0001 #random selection
b1 = 0.9 #beta 1
b2 = 0.98 #beta 2
eps = 1e-9
