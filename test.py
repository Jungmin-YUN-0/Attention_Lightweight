import spacy
import torch
from torchtext.data.metrics import bleu_score  # BLEU
import dill
from torchtext.data import Dataset, BucketIterator
import hparams
from model.Transformer import Transformer

#############
# inference #
#############

## translation
def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50, logging=True):
    model.eval() # 평가 모드

    # tokenization
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # start with <sos> token, end with <eos> token
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    # source -> mask 생성
    src_mask = model.make_src_mask(src_tensor)

    # encoder(input sentence) -> output
    with torch.no_grad():  # gradient 계산 x (inference니까 필요x, memory,, 연산속도,,)
        enc_src = model.encoder(src_tensor, src_mask)

    ## start with a <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # target -> mask 생성
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # 출력 문장에서 가장 마지막 단어만 사용
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # end with <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 각 출력 단어의 인덱스 -> 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외 후, 출력 문장 반환
    return trg_tokens[1:], attention

########################
# calculate bleu_score #
########################
def show_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    index = 0

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len, logging=False)

        # <eos> token 제거
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])
        index += 1
        if (index + 1) % 100 == 0:
            print(f"[{index + 1}/{len(data)}]")
            print(f"예측: {pred_trg}")
            print(f"정답: {trg}")

    bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
    return bleu

##############################################################################

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # cuda(gpu) 사용
batch_size = 64

data = dill.load(open("data.pickle", 'rb'))

SRC_PAD_IDX = data['vocab']['src'].vocab.stoi["<blank>"]
TRG_PAD_IDX = data['vocab']['trg'].vocab.stoi["<blank>"]

INPUT_DIM = len(data['vocab']['src'].vocab)  # src_vocab_size
OUTPUT_DIM = len(data['vocab']['trg'].vocab)  # trg_vocab_size

fields = {'src': data['vocab']['src'], 'trg' : data['vocab']['trg']}

#train_data = Dataset(examples=data['train_data'], fields=fields)
#val_data = Dataset(examples=data['valid_data'], fields=fields)
test_data = Dataset(examples=data['test_data'], fields=fields)

test_iterator = BucketIterator(test_data, batch_size=batch_size, device=device)

## Create Model
model = Transformer(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX,hparams.HIDDEN_DIM,
                                hparams.ENC_LAYERS, hparams.DEC_LAYERS, hparams.ENC_HEADS, hparams.DEC_HEADS,
                                hparams.ENC_PF_DIM, hparams.DEC_PF_DIM, hparams.ENC_DROPOUT, hparams.DEC_DROPOUT, device).to(device)



bleu = show_bleu(test_data, data['vocab']['src'], data['vocab']['trg'], model, device)
print(bleu)