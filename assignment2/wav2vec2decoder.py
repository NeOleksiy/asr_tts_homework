import math
from typing import List, Tuple
import heapq
import csv
import sys
import kenlm
import torch
import torchaudio
import jiwer
from collections import defaultdict
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=2,
            alpha=0.01,
            beta=1,
            temperature=0.5,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = torch.argmax(log_probs, dim=-1).tolist()
        final_token_ids = []
        prev_id = None
        for token_id in token_ids:
            if token_id != prev_id:
                if token_id != self.blank_token_id:
                    final_token_ids.append(token_id)
                prev_id = token_id
        
        return self._ids_to_text(final_token_ids)


    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """

        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        T, V = log_probs.shape

        beams = {(): (0.0, float('-inf'))}

        for t in range(T):
            next_beams = {}
            frame = log_probs[t]

            for tokens, (blank, nonblank) in beams.items():
                total = _log_add(blank, nonblank)

                for v in range(V):
                    p = frame[v]

                    if v == self.blank_token_id:
                        new_tokens = tokens
                        new_blank = total + p
                        new_nonblank = float('-inf')
                    elif tokens and tokens[-1] == v:
                        new_tokens = tokens
                        new_blank = float('-inf')
                        new_nonblank = nonblank + p

                        if blank > float('-inf'):
                            emit_tokens = tokens + (v,)
                            emit_nonblank = blank + p
                            if emit_tokens in next_beams:
                                eb, enb = next_beams[emit_tokens]
                                next_beams[emit_tokens] = (eb, _log_add(enb, emit_nonblank))
                            else:
                                next_beams[emit_tokens] = (float('-inf'), emit_nonblank)
                    else:
                        new_tokens = tokens + (v,)
                        new_blank = float('-inf')
                        new_nonblank = total + p

                    if new_tokens in next_beams:
                        b, nb = next_beams[new_tokens]
                        next_beams[new_tokens] = (_log_add(b, new_blank),
                                                _log_add(nb, new_nonblank))
                    else:
                        next_beams[new_tokens] = (new_blank, new_nonblank)

            heap_data = [(-_log_add(pb, pn), tokens, (pb, pn)) for tokens, (pb, pn) in next_beams.items()]
            heapq.heapify(heap_data)

            # topk
            best = heapq.nsmallest(self.beam_width, heap_data)

            beams = {tokens: (pb, pn) for _, tokens, (pb, pn) in best}

        results = [(list(tokens), _log_add(blank, nonblank))
                for tokens, (blank, nonblank) in beams.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results if return_beams else self._ids_to_text(results[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        T, V = log_probs.shape
        
        init_state = kenlm.State()
        self.lm_model.BeginSentenceWrite(init_state)
        
        beams = {(): (0.0, float('-inf'), init_state, 0.0, 0)}
        
        for t in range(T):
            next_beams = {}
            frame = log_probs[t]
            
            for tokens, (p_blank, p_non_blank, state, lm_score, words) in beams.items():
                total = _log_add(p_blank, p_non_blank)
                
                for v in range(V):
                    p = frame[v]
                    
                    if v == self.blank_token_id:
                        new_tokens = tokens
                        new_p_blank = total + p
                        new_p_non_blank = float('-inf')
                        new_state = state
                        new_lm_score = lm_score
                        new_words = words
                    elif tokens and tokens[-1] == v:
                        new_tokens = tokens
                        new_p_blank = float('-inf')
                        new_p_non_blank = p_non_blank + p
                        new_state = state
                        new_lm_score = lm_score
                        new_words = words
                        
                        if p_blank > float('-inf'):
                            char = self.vocab[v]
                            emit_state = kenlm.State()
                            char_score = self.lm_model.BaseScore(state, char, emit_state)
                            emit_tokens = tokens + (v,)
                            emit_p_non_blank = p_blank + p
                            emit_lm_score = lm_score + char_score
                            emit_words = words + (1 if char == self.word_delimiter else 0)
                            
                            emit_combined = emit_p_non_blank + self.alpha * emit_lm_score + self.beta * emit_words
                            
                            if emit_tokens in next_beams:
                                ex_p_b, ex_p_nb, ex_state, ex_lm, ex_words = next_beams[emit_tokens]
                                ex_combined = _log_add(ex_p_b, ex_p_nb) + self.alpha * ex_lm + self.beta * ex_words
                                if emit_combined > ex_combined:
                                    next_beams[emit_tokens] = (float('-inf'), emit_p_non_blank, emit_state, emit_lm_score, emit_words)
                                else:
                                    merged_p_nb = _log_add(ex_p_nb, emit_p_non_blank)
                                    next_beams[emit_tokens] = (ex_p_b, merged_p_nb, ex_state, ex_lm, ex_words)
                            else:
                                next_beams[emit_tokens] = (float('-inf'), emit_p_non_blank, emit_state, emit_lm_score, emit_words)
                    else:
                        char = self.vocab[v]
                        new_state = kenlm.State()
                        char_score = self.lm_model.BaseScore(state, char, new_state)
                        new_tokens = tokens + (v,)
                        new_p_blank = float('-inf')
                        new_p_non_blank = total + p
                        new_lm_score = lm_score + char_score
                        new_words = words + (1 if char == self.word_delimiter else 0)
                    
                    if v == self.blank_token_id or (tokens and tokens[-1] == v) or v != self.blank_token_id:
                        combined = _log_add(new_p_blank, new_p_non_blank) + self.alpha * new_lm_score + self.beta * new_words
                        
                        if new_tokens in next_beams:
                            ex_p_b, ex_p_nb, ex_state, ex_lm, ex_words = next_beams[new_tokens]
                            ex_combined = _log_add(ex_p_b, ex_p_nb) + self.alpha * ex_lm + self.beta * ex_words
                            if combined > ex_combined:
                                next_beams[new_tokens] = (new_p_blank, new_p_non_blank, new_state, new_lm_score, new_words)
                        else:
                            if new_p_blank > float('-inf') or new_p_non_blank > float('-inf'):
                                next_beams[new_tokens] = (new_p_blank, new_p_non_blank, new_state, new_lm_score, new_words)
            
            heap_data = [(-(_log_add(pb, pn) + self.alpha * lms + self.beta * w), tokens, (pb, pn, state, lms, w))
                        for tokens, (pb, pn, state, lms, w) in next_beams.items()]
            heapq.heapify(heap_data)
            best_items = heapq.nsmallest(self.beam_width, heap_data)
            beams = {tokens: (pb, pn, state, lms, w) for _, tokens, (pb, pn, state, lms, w) in best_items}
        
        best_tokens = None
        best_combined = float('-inf')
        for tokens, (p_blank, p_non_blank, state, lm_score, words) in beams.items():
            combined = _log_add(p_blank, p_non_blank) + self.alpha * lm_score + self.beta * words
            if combined > best_combined:
                best_combined = combined
                best_tokens = tokens
        
        return self._ids_to_text(list(best_tokens))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        best_score = float('-inf')
        best_tokens = None
        
        for token_ids, acoustic_score in beams:
            lm_state = kenlm.State()
            self.lm_model.BeginSentenceWrite(lm_state)
            
            lm_score = 0.0
            num_words = 0
            
            for token_id in token_ids:
                char = self.vocab[token_id]
                new_lm_state = kenlm.State()
                char_score = self.lm_model.BaseScore(lm_state, char, new_lm_state)
                lm_score += char_score
                lm_state = new_lm_state
                
                if char == self.word_delimiter:
                    num_words += 1
            
            combined_score = acoustic_score + self.alpha * lm_score + self.beta * num_words
            
            if combined_score > best_score:
                best_score = combined_score
                best_tokens = token_ids
        
        return self._ids_to_text(best_tokens)

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------



def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str):
    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    results = {}  # метод -> (wer, cer)

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue

        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")

        results[method] = (wer, cer)

    return results


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # На всех слишком долго экспы делать, нарандомил штук 30

    test_samples = [
        ("data/librispeech_test_other/sample_16.wav","the window was barred but he went to it and tried the bars one by one to find them all solidly fitted into the stone sill"),
        ("data/librispeech_test_other/sample_17.wav","next moment as he felt his way about his hand touched an old fashioned marble mantelpiece fireplace chimney"),
        ("data/librispeech_test_other/sample_186.wav","when suddenly a violent squall of wind arose and smote the ship which rose out of the water and settled upon a great reef the haunt of sea monsters where it broke up and fell asunder into planks and all and everything on board were plunged into the sea"),
        ("data/librispeech_test_other/sample_19.wav","no that was too bad he could not do that"),
        ("data/librispeech_test_other/sample_20.wav","sympathy and pity for the dwellers in the hoze were completely gone now and he set his teeth fast and mentally called himself a weak idiot for ever thinking about such people"),
        ("data/librispeech_test_other/sample_21.wav","a narrow table against the wall in two places"),
        ("data/librispeech_test_other/sample_37.wav","what did they say false alarm tell sir risdon they would clear all away to night see if anything had been left about lobster boat"),
        ("data/librispeech_test_other/sample_28.wav","stop here till sir risdon comes down and tell him i'm very sorry that we should have cleared out last night only a born fool saw jerry nandy's lobster boat coming into the cove and came running to say it was a party from the cutter yes father"),
        ("data/librispeech_test_other/sample_112.wav","gurr glanced round to see if the men were looking and then said rather huskily but kindly"),
        ("data/librispeech_test_other/sample_127.wav","gurr saluted and stated his business while the baronet who had turned sallower and more careworn than his lot drew a breath full of relief one of your ship boys he said"),
        ("data/librispeech_test_other/sample_146.wav","there i abode a little and then went on to baghdad where i entered my quarter and found my house and foregathered with my family and saluted my friends who gave me joy of my safe return and i laid up all my goods and valuables in my storehouses"),
        ("data/librispeech_test_other/sample_147.wav","after which i returned to my old merry way of life and forgot all i had suffered in the great profit and gain i had made"),
        ("data/librispeech_test_other/sample_170.wav","by allah o my lord answered i thou hast indeed overwhelmed me with thy favours and well doings but i weary for a sight of my friends and family and native country"),
        ("data/librispeech_test_other/sample_171.wav","then i took leave of him and of all my intimates and acquaintances in the island and embarked with the merchants aforesaid"),
        ("data/librispeech_test_other/sample_172.wav","he asked me whence they came and i said to him by allah o commander of the faithful i know not the name of the city nor the way thither"),
        ("data/librispeech_test_other/sample_173.wav","for state processions a throne is set for him upon a huge elephant eleven cubits high and upon this he sitteth having his great lords and officers and guests standing in two ranks on his right hand and on his left"),
        ("data/librispeech_test_other/sample_174.wav","his letter hath shown me this and as for the mightiness of his dominion thou hast told us what thou hast eye witnessed"),
        ("data/librispeech_test_other/sample_175.wav","presently my friends came to me and i distributed presents among my family and gave alms and largesse after which i yielded myself to joyance and enjoyment mirth and merry making and forgot all that i had suffered"),
        ("data/librispeech_test_other/sample_176.wav","such then o my brothers is the history of what befel me in my sixth voyage and to morrow inshallah"),
        ("data/librispeech_test_other/sample_177.wav","i will tell you the story of my seventh and last voyage which is still more wondrous and marvellous than that of the first six"),
        ("data/librispeech_test_other/sample_178.wav","when it was the five hundred and sixty third night"),
        ("data/librispeech_test_other/sample_179.wav","she said it hath reached me o auspicious king that when sindbad the seaman had related the history of what befel him in his sixth voyage and all the company had dispersed sindbad the landsman went home and slept as of wont"),
        ("data/librispeech_test_other/sample_180.wav","the seventh voyage of sindbad the seaman"),
        ("data/librispeech_test_other/sample_181.wav","know o company that after my return from my sixth voyage which brought me abundant profit i resumed my former life in all possible joyance and enjoyment and mirth and making merry day and night and i tarried some time in this solace and satisfaction till my soul began once more to long to sail the seas and see foreign countries and company with merchants and hear new things"),
        ("data/librispeech_test_other/sample_182.wav","so having made up my mind i packed up in bales a quantity of precious stuffs suited for sea trade and repaired with them from baghdad city to bassorah town where i found a ship ready for sea and in her a company of considerable merchants"),
        ("data/librispeech_test_other/sample_183.wav","but the captain arose and tightening his girdle tucked up his skirts and after taking refuge with allah from satan the stoned clomb to the mast head whence he looked out right and left and gazing at the passengers and crew fell to buffeting his face and plucking out his beard"),
        ("data/librispeech_test_other/sample_184.wav","this he set in a saucer wetted with a little water and after waiting a short time smelt and tasted it and then he took out of the chest a booklet wherein he read awhile and said weeping know o ye passengers that in this book is a marvellous matter denoting that whoso cometh hither shall surely die without hope of escape for that this ocean is called the sea of the clime of the king wherein is the sepulchre of our lord solomon son of david on both be peace"),
        ("data/librispeech_test_other/sample_185.wav","a second fish made its appearance than which we had seen naught more monstrous"),
        ("data/librispeech_test_other/sample_186.wav","when suddenly a violent squall of wind arose and smote the ship which rose out of the water and settled upon a great reef the haunt of sea monsters where it broke up and fell asunder into planks and all and everything on board were plunged into the sea"),
        ("data/librispeech_test_other/sample_187.wav","although the plague was there in the most part of all the houses they nevertheless entered everywhere then plundered and carried away all that was within and yet for all this not one of them took any hurt which is a most wonderful case"),
        ("data/librispeech_test_other/sample_188.wav","i beseech you think upon it"),
        ("data/librispeech_test_other/sample_151.wav","presently the ship struck the mountain and broke up and all and everything on board of her were plunged into the sea")

    ]

    decoder = Wav2Vec2Decoder()  # set lm_model_path for Tasks 4+

    metrics = {method: {"wer": [], "cer": []} for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]}

    for audio_path, reference in test_samples:
        results = test(decoder, audio_path, reference)
        for method, (wer, cer) in results.items():
            metrics[method]["wer"].append(wer)
            metrics[method]["cer"].append(cer)

    print("\n" + "=" * 60)
    print("Average metrics:")
    for method in metrics:
        wer_list = metrics[method]["wer"]
        cer_list = metrics[method]["cer"]
        if wer_list:
            avg_wer = sum(wer_list) / len(wer_list)
            avg_cer = sum(cer_list) / len(cer_list)
            print(f"  [{method}] avg WER={avg_wer:.2%}  avg CER={avg_cer:.2%}")
        else:
            print(f"  [{method}] have not successfull decoding")