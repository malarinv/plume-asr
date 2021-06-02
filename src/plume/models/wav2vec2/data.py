from pathlib import Path
from collections import Counter
import shutil
import io

# from time import time

# import pydub
import typer
from tqdm import tqdm

from plume.utils import (
    ExtendedPath,
    replace_redundant_spaces_with,
    lazy_module,
    random_segs,
    parallel_apply,
    batch,
    run_shell,
)

from plume.utils.vad import VADUtterance

soundfile = lazy_module("soundfile")
pydub = lazy_module("pydub")
app = typer.Typer()


@app.command()
def export_jasper(src_dataset_path: Path, dest_dataset_path: Path, unlink: bool = True):
    dict_ltr = dest_dataset_path / Path("dict.ltr.txt")
    (dest_dataset_path / Path("wavs")).mkdir(exist_ok=True, parents=True)
    tok_counter = Counter()
    shutil.copy(
        src_dataset_path / Path("test_manifest.json"),
        src_dataset_path / Path("valid_manifest.json"),
    )
    if unlink:
        src_wavs = src_dataset_path / Path("wavs")
        for wav_path in tqdm(list(src_wavs.glob("**/*.wav"))):
            audio_seg = (
                pydub.AudioSegment.from_wav(wav_path)
                .set_frame_rate(16000)
                .set_channels(1)
            )
            dest_path = dest_dataset_path / Path("wavs") / Path(wav_path.name)
            audio_seg.export(dest_path, format="wav")

    for dataset_kind in ["train", "valid"]:
        abs_manifest_path = ExtendedPath(
            src_dataset_path / Path(f"{dataset_kind}_manifest.json")
        )
        manifest_data = list(abs_manifest_path.read_jsonl())
        o_tsv, o_ltr = f"{dataset_kind}.tsv", f"{dataset_kind}.ltr"
        out_tsv = dest_dataset_path / Path(o_tsv)
        out_ltr = dest_dataset_path / Path(o_ltr)
        with out_tsv.open("w") as tsv_f, out_ltr.open("w") as ltr_f:
            if unlink:
                tsv_f.write(f"{dest_dataset_path}\n")
            else:
                tsv_f.write(f"{src_dataset_path}\n")
            for md in manifest_data:
                audio_fname = md["audio_filepath"]
                pipe_toks = replace_redundant_spaces_with(md["text"], "|").upper()
                # pipe_toks = "|".join(re.sub(" ", "", md["text"]))
                # pipe_toks = alnum_to_asr_tokens(md["text"]).upper().replace(" ", "|")
                tok_counter.update(pipe_toks)
                letter_toks = " ".join(pipe_toks) + " |\n"
                frame_count = soundfile.info(audio_fname).frames
                rel_path = Path(audio_fname).relative_to(src_dataset_path.absolute())
                ltr_f.write(letter_toks)
                tsv_f.write(f"{rel_path}\t{frame_count}\n")
    with dict_ltr.open("w") as d_f:
        for k, v in tok_counter.most_common():
            d_f.write(f"{k} {v}\n")
    (src_dataset_path / Path("valid_manifest.json")).unlink()


@app.command()
def set_root(dataset_path: Path, root_path: Path):
    for dataset_kind in ["train", "valid"]:
        data_file = dataset_path / Path(dataset_kind).with_suffix(".tsv")
        with data_file.open("r") as df:
            lines = df.readlines()
        with data_file.open("w") as df:
            lines[0] = str(root_path) + "\n"
            df.writelines(lines)


@app.command()
def convert_audio(log_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    all_wavs = list((log_dir).glob("**/*.wav"))
    name_wav_map = {i.name: i.absolute() for i in all_wavs}
    exists_wavs = list((out_dir).glob("**/*.wav"))
    rem_wavs = list(
        set((i.name for i in all_wavs)) - set((i.name for i in exists_wavs))
    )
    rem_wavs_real = [name_wav_map[i] for i in rem_wavs]

    def resample_audio(i):
        dest_wav = out_dir / i.name
        if dest_wav.exists():
            return
        run_shell(f"ffmpeg -i {i.absolute()} -ac 1 -ar 16000 {dest_wav}", verbose=False)

    parallel_apply(resample_audio, rem_wavs_real, workers=256)


@app.command()
def prepare_pretraining(
    log_dir: Path,
    dataset_path: Path,
    format: str = "wav",
    method: str = "random",
    max_silence: int = 3000,
    min_duration: int = 10000,
    max_duration: int = 30000,
    fixed_duration: int = 30000,
    batch_size: int = 100,
):
    audio_dir = dataset_path / "audio"
    audio_dir.mkdir(exist_ok=True, parents=True)
    cache_dir = dataset_path / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    all_wavs = list((log_dir).glob("**/*.wav"))
    if method not in ["vad", "random", "fixed"]:
        typer.echo("should be one of random|fixed")
        raise typer.Exit()

    def write_seg_arg(arg):
        seg, dest_wav = arg
        ob = io.BytesIO()
        seg.export(ob, format=format)
        dest_wav.write_bytes(ob.getvalue())
        ob.close()

    with (dataset_path / "failed.log").open("w") as fl:
        vad_utt = VADUtterance(
            max_silence=max_silence,
            min_utterance=min_duration,
            max_utterance=max_duration,
        )

        def vad_process_wav(wav_path):
            if (cache_dir / wav_path.stem).exists():
                return []
            try:
                aud_seg = pydub.AudioSegment.from_file(wav_path)
            except pydub.exceptions.CouldntDecodeError:
                fl.write(wav_path.name + "\n")
                return []
            full_seg = aud_seg
            # segs = random_segs(len(full_seg), min_duration, max_duration)
            segs = vad_utt.stream_segments(full_seg)
            audio_chunk_paths = []
            if len(full_seg) > min_duration:
                for (i, chunk_seg) in enumerate(segs):
                    dest_wav = audio_dir / (wav_path.stem + f"_{i}.{format}")
                    if dest_wav.exists():
                        continue
                    audio_chunk_paths.append((chunk_seg, dest_wav))
            (cache_dir / wav_path.stem).touch()
            return audio_chunk_paths

        def random_process_wav(wav_path):
            if (cache_dir / wav_path.stem).exists():
                return []
            try:
                aud_seg = pydub.AudioSegment.from_file(wav_path)
            except pydub.exceptions.CouldntDecodeError:
                fl.write(wav_path.name + "\n")
                return []
            full_seg = aud_seg
            segs = random_segs(len(full_seg), min_duration, max_duration)
            audio_chunk_paths = []
            if len(full_seg) > min_duration:
                for (i, (start, end)) in enumerate(segs):
                    dest_wav = audio_dir / (wav_path.stem + f"_{i}.{format}")
                    if dest_wav.exists():
                        continue
                    chunk_seg = aud_seg[start:end]
                    audio_chunk_paths.append((chunk_seg, dest_wav))
            (cache_dir / wav_path.stem).touch()
            return audio_chunk_paths

        def fixed_process_wav(wav_path):
            if (cache_dir / wav_path.stem).exists():
                return []
            try:
                aud_seg = pydub.AudioSegment.from_file(wav_path)
            except pydub.exceptions.CouldntDecodeError:
                fl.write(wav_path.name + "\n")
                return []
            full_seg = aud_seg
            audio_chunk_paths = []
            if len(full_seg) > min_duration:
                for (i, chunk_seg) in enumerate(full_seg[::fixed_duration]):
                    dest_wav = audio_dir / (wav_path.stem + f"_{i}.{format}")
                    if dest_wav.exists() or len(chunk_seg) < min_duration:
                        continue
                    audio_chunk_paths.append((chunk_seg, dest_wav))
            (cache_dir / wav_path.stem).touch()
            return audio_chunk_paths

        # warmup
        pydub.AudioSegment.from_file(all_wavs[0])
        # parallel_apply(process_wav, all_wavs, pool='process')
        # parallel_apply(process_wav, all_wavs)
        seg_f = (
            vad_process_wav
            if method == "vad"
            else (random_process_wav if method == "random" else fixed_process_wav)
        )
        for wp_batch in tqdm(batch(all_wavs, n=batch_size)):
            acp_batch = parallel_apply(seg_f, wp_batch)
            # acp_batch = list(map(seg_f, tqdm(wp_batch)))
            flat_acp_batch = [sd for acp in acp_batch for sd in acp]
            parallel_apply(write_seg_arg, flat_acp_batch)
            # for acp in acp_batch:
            #     for (seg, des) in acp:
            #         seg.export(des)
            # for seg_des in tqdm(flat_acp_batch):
            #     write_seg_arg(seg_des)
            del flat_acp_batch
            del acp_batch


def main():
    app()


if __name__ == "__main__":
    main()
