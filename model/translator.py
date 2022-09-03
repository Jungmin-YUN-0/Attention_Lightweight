import torch
import torch.nn as nn
import torch.nn.functional as F


class Translator(nn.Module):
    def __init__(self, model, beam_size, max_seq_len, src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx, device):
        super().__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.device= device
        self.model = model
        self.model.eval()

        #register_buffer -> 모델 매개 변수로 간주하지 x
        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer('blank_seqs', torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer('len_map', torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))


    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = (1 - torch.triu(torch.ones((1, trg_seq.shape[1], trg_seq.shape[1]), device=self.device), diagonal=1)).bool()
        dec_output = self.model.decoder(trg_seq, enc_output, trg_mask, src_mask)
        return F.softmax(self.model.trg_word_prj(dec_output), dim=-1)  #!#


    def _get_init_state(self, src_seq, src_mask):
        enc_output = self.model.encoder(src_seq, src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)
        best_k_probs, best_k_idx = dec_output[:, -1, :].topk(self.beam_size)
        
        scores = torch.log(best_k_probs).view(self.beam_size)   
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(self.beam_size, 1, 1)
        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        # each beam마다 k개의 후보 사용 (total candidate: k^2)
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(self.beam_size)
        # previous score를 포함 (누적확률_곱)
        scores = torch.log(best_k2_probs).view(self.beam_size, -1) + scores.view(self.beam_size, 1)
        # k^2개의 후보 중에서, best k개 선정
        scores, best_k_idx_in_k2 = scores.view(-1).topk(self.beam_size)
        # 선정된 best k개의 idx 정보
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // self.beam_size, best_k_idx_in_k2 % self.beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]
        # Copy the corresponding previous tokens
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Beam Search -> best token 선정
        gen_seq[:, step] = best_k_idx
        return gen_seq, scores


    def translate_sentence(self, src_seq):
        # batch size = 1
        assert src_seq.size(0) == 1
        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx 
        with torch.no_grad():
            src_mask = (src_seq != src_pad_idx).unsqueeze(-2)
            enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

            ans_idx = 0   # default
            for step in range(2, self.max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position
                # => for Length Penalty (빔의 길이가 길어질수록 누적확률의 값이 작아지므로)
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, self.max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** self.alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
