from plume.utils.regentity import infer_num_replacer


def test_infer_num():
    repl = infer_num_replacer()

    assert (
        repl(
            "SIX NINE TRIPL EIGHT SIX SIX DOULE NINE THREE ZERO TWO SEVENT-ONE"
        )
        == "69888669930271"
    )

    assert (
        repl("SIX NINE FSIX EIGHT IGSIX SIX NINE NINE THRE ZERO TWO SEVEN ONE")
        == "6968669930271"
    )

    assert (
        repl("FORTY-TWO SEVEN SIXTY-FOUR SEVEN THREE FIVE U OH FOUR SIX")
        == "42764735046"
    )
    assert (
        repl("FORTY-TWO SEVEN SIXTY-FOUR SEVEN THREE FIVE U OH FOUR SIX")
        == "42764735046"
    )
