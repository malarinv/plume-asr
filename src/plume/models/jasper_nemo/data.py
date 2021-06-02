from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def set_root(dataset_path: Path, root_path: Path):
    pass
    # for dataset_kind in ["train", "valid"]:
    #     data_file = dataset_path / Path(dataset_kind).with_suffix(".tsv")
    #     with data_file.open("r") as df:
    #         lines = df.readlines()
    #     with data_file.open("w") as df:
    #         lines[0] = str(root_path) + "\n"
    #         df.writelines(lines)


def main():
    app()


if __name__ == "__main__":
    main()
