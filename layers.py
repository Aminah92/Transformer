class EncoderLayer(nn.Module):
    "Implements the encoder with the two key features(Feed forward and Attention)"
    def __init__(self, dmodel,h):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(dmodel,h)
        self.pffn = PFFN(dmodel)
        self.sublayer1 = LayerNorm(dmodel)
        self.sublayer2 = LayerNorm(dmodel)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, mask):
      xnew = self.sublayer1(x)
      x = x + self.dropout(self.attention(xnew,xnew,xnew,mask))
      xnew = self.sublayer2(x)
      x = x + self.dropout(self.pffn(xnew))
      return x
    
    
    
class DecoderLayer(nn.Module):
    "Implements the decoder L"
    def __init__(self, dmodel,h):
        super(DecoderLayer, self).__init__()
        self.attention = MultiheadAttention(dmodel,h)
        self.pffn = PFFN(dmodel)
        self.sublayer1 = LayerNorm(dmodel)
        self.sublayer2 = LayerNorm(dmodel)
        self.sublayer3 = LayerNorm(dmodel)
        self.dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x, encoder_out, src_mask, trg_mask): 
      "Takes Encoder output in the second layer and a source and target mask"
      xnew = self.sublayer1(x)
      x = x + self.dropout(self.attention(xnew,xnew,xnew,trg_mask))
      xnew = self.sublayer2(x)
      x = x + self.dropout(self.attention(xnew,encoder_out,encoder_out,src_mask))
      xnew = self.sublayer3(x)
      x = x + self.dropout(self.pffn(xnew))
      return x
