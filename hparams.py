# hyperparameter 설정

#INPUT_DIM: len(SRC.vocab)
#OUTPUT_DIM: len(TRG.vocab)

HIDDEN_DIM = 512  #d_model

ENC_LAYERS = 6
DEC_LAYERS = 6

ENC_HEADS = 8
DEC_HEADS = 8

ENC_PF_DIM = 2048 
DEC_PF_DIM = 2048

ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1