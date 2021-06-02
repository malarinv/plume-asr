from io import BytesIO
import warnings
import itertools as it

import torch
import soundfile as sf
import torch.nn.functional as F

try:
    from fairseq import utils
    from fairseq.models import BaseFairseqModel
    from fairseq.data import Dictionary
    from fairseq.models.wav2vec.wav2vec2_asr import base_architecture, Wav2VecEncoder
except ModuleNotFoundError:
    warnings.warn("Install fairseq")
try:
    from wav2letter.decoder import CriterionType
    from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
except ModuleNotFoundError:
    warnings.warn("Install wav2letter")


class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, target_dict):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, target_dict)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, model, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(model, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, model, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = model(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = list()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)

        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != "none":
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence


def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    if torch.cuda.is_available():
        feats = feats.cuda()
    feats = postprocess(feats, sample_rate)
    return feats


def load_model(ctc_model_path, w2v_model_path, target_dict):
    w2v = torch.load(ctc_model_path)
    w2v["args"].w2v_path = w2v_model_path
    model = Wav2VecCtc.build_model(w2v["args"], target_dict)
    model.load_state_dict(w2v["model"], strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class Wav2Vec2ASR(object):
    """docstring for Wav2Vec2ASR."""

    def __init__(self, ctc_path, w2v_path, target_dict_path):
        super(Wav2Vec2ASR, self).__init__()
        self.target_dict = Dictionary.load(target_dict_path)

        self.model = load_model(ctc_path, w2v_path, self.target_dict)
        self.model.eval()

        self.generator = W2lViterbiDecoder(self.target_dict)

    def transcribe(self, audio_data, greedy=True):
        aud_f = BytesIO(audio_data)
        # aud_seg = pydub.AudioSegment.from_file(aud_f)
        # feat_seg = aud_seg.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        # feat_f = io.BytesIO()
        # feat_seg.export(feat_f, format='wav')
        # feat_f.seek(0)
        net_input = {}
        feature = get_feature(aud_f)
        net_input["source"] = feature.unsqueeze(0)

        padding_mask = (
            torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)
        )
        if torch.cuda.is_available():
            padding_mask = padding_mask.cuda()

        net_input["padding_mask"] = padding_mask
        sample = {}
        sample["net_input"] = net_input

        with torch.no_grad():
            hypo = self.generator.generate(self.model, sample, prefix_tokens=None)
        hyp_pieces = self.target_dict.string(hypo[0][0]["tokens"].int().cpu())
        result = post_process(hyp_pieces, "letter")
        return result
