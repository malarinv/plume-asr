import typer
# from fairseq_cli.train import cli_main
import sys
from pathlib import Path
import shlex
from plume.utils import lazy_callable

cli_main = lazy_callable('fairseq_cli.train.cli_main')

app = typer.Typer()


@app.command()
def local(dataset_path: Path):
    args = f'''--distributed-world-size 1 {dataset_path} \
--save-dir /dataset/wav2vec2/model/wav2vec2_l_num_ctc_v1 --post-process letter --valid-subset \
valid --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 --max-update 80000 \
--sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path /dataset/wav2vec2/pretrained/wav2vec_vox_new.pt \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--zero-infinity --feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 \
--hold-steps 32000 --decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 \
--activation-dropout 0.1 --criterion ctc --attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format json \
--log-interval 500 --ddp-backend no_c10d  --reset-optimizer --normalize
'''
    new_args = ['train.py']
    new_args.extend(shlex.split(args))
    sys.argv = new_args
    cli_main()


if __name__ == "__main__":
    cli_main()
