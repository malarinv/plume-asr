from pathlib import Path
from collections import Counter
import shutil

# import pydub
import typer
from tqdm import tqdm

from plume.utils import (
    ExtendedPath,
    replace_redundant_spaces_with,
    lazy_module,
)

soundfile = lazy_module("soundfile")
pydub = lazy_module("pydub")
app = typer.Typer()


@app.command()
def export_jasper(
    src_dataset_path: Path, dest_dataset_path: Path, unlink: bool = True
):
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
                pipe_toks = replace_redundant_spaces_with(
                    md["text"], "|"
                ).upper()
                # pipe_toks = "|".join(re.sub(" ", "", md["text"]))
                tok_counter.update(pipe_toks)
                letter_toks = " ".join(pipe_toks) + " |\n"
                frame_count = soundfile.info(audio_fname).frames
                rel_path = Path(audio_fname).relative_to(
                    src_dataset_path.absolute()
                )
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


def main():
    app()


if __name__ == "__main__":
    main()
