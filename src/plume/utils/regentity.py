import re

from .lazy_import import lazy_callable, lazy_module

num2words = lazy_callable("num2words.num2words")
spellchecker = lazy_module("spellchecker")
editdistance = lazy_module("editdistance")

# from num2words import num2words


def entity_replacer_keeper(
    pre_rules=[], entity_rules=[], post_rules=[], verbose=False
):
    # def replacer_keeper_gen():
    pre_rules_c = [(re.compile(k), v) for (k, v) in pre_rules]
    entity_rules_c = [
        (re.compile(k, re.IGNORECASE), v) for (k, v) in entity_rules
    ]
    post_rules_c = [(re.compile(k), v) for (k, v) in post_rules]

    re_rules = pre_rules_c + entity_rules_c + post_rules_c

    def replacer(w2v_out):
        out = w2v_out
        for (k, v) in re_rules:
            orig = out
            out = k.sub(v, out)
            if verbose:
                print(f"rule |{k}|: sub:|{v}| |{orig}|=> |{out}|")
        return out

    def merge_intervals(intervals):
        # https://codereview.stackexchange.com/a/69249
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                # test for intersection between lower and higher:
                # we know via sorting that lower[0] <= higher[0]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (
                        lower[0],
                        upper_bound,
                    )  # replace by merged interval
                else:
                    merged.append(higher)
        return merged

    # optimal merging interval tree
    # https://www.geeksforgeeks.org/interval-tree/

    def keep_literals(w2v_out):
        # out = re.sub(r"[ ;,.]", " ", w2v_out).strip()
        out = w2v_out
        for (k, v) in pre_rules_c:
            out = k.sub(v, out)
        num_spans = []
        if verbose:
            print(f"num_rules: {len(entity_rules_c)}")
        for (k, v) in entity_rules_c:  # [94:]:
            matches = k.finditer(out)
            for m in matches:
                # num_spans.append(m.span())
                # look at space seprated internal entities
                (start, end) = m.span()
                for s in re.finditer(r"\S+", out[start:end]):
                    (start_e, end_e) = s.span()
                    num_spans.append((start_e + start, end_e + start))
                    if verbose:
                        t = out[start_e + start : end_e + start]
                        print(f"rule |{k}|: sub:|{v}| => |{t}|")

        merged = merge_intervals(num_spans)
        num_ents = len(merged)
        keep_out = " ".join((out[s[0] : s[1]] for s in merged))
        for (k, v) in post_rules_c:
            keep_out = k.sub(v, keep_out)
        return keep_out, num_ents

    return replacer, keep_literals


def default_num_only_rules(num_range):
    entity_rules = (
        [
            (
                r"\b" + num2words(i) + r"\b",
                str(i),
            )
            for i in reversed(range(num_range))
        ]
        + [
            (
                r"\b" + str(i) + r"\b",
                str(i),
            )
            for i in reversed(range(10))
        ]
        + [
            (r"\bhundred\b", "00"),
        ]
    )
    return entity_rules


def default_num_rules(num_range):
    entity_rules = default_num_only_rules(num_range) + [
        (r"\boh\b", "0"),
        (r"\bo\b", "0"),
        (r"\bdouble(?: |-)(\w+|\d+)\b", "\\1 \\1"),
        (r"\btriple(?: |-)(\w+|\d+)\b", "\\1 \\1 \\1"),
    ]
    return entity_rules


def infer_num_rules_vocab(num_range):
    vocab = [num2words(i) for i in reversed(range(num_range))] + [
        "hundred",
        "double",
        "triple",
    ]
    entity_rules = [
        (
            num2words(i),
            str(i),
        )
        for i in reversed(range(num_range))
    ] + [
        (r"\bhundred\b", "00"),
        (r"\boh\b", "0"),
        (r"\bo\b", "0"),
        (r"\bdouble(?: |-)(\w+|\d+)\b", "\\1 \\1"),
        (r"\btriple(?: |-)(\w+|\d+)\b", "\\1 \\1 \\1"),
    ]
    return entity_rules, vocab


def do_tri_verbose_list():
    return [
        num2words(i) for i in list(range(11, 19)) + list(range(20, 100, 10))
    ] + ["hundred"]


def default_alnum_rules(num_range, oh_is_zero, i_oh_limit):
    oh_is_zero_rules = [
        (r"\boh\b", "0"),
        (r"\bo\b", "0"),
    ]

    num_list = [num2words(i) for i in reversed(range(num_range))]
    al_num_regex = r"|".join(num_list) + r"|[0-9a-z]"
    o_i_vars = r"(\[?(?:Oh|O|I)\]?)"
    i_oh_limit_rules = [
        (r"\b([a-hj-np-z])\b", "\\1"),
        (
            r"\b((?:"
            + al_num_regex
            + r"|^)\b\s*)(I|O)(\s*\b)(?="
            + al_num_regex
            + r"\s+|$)\b",
            "\\1[\\2]\\3",
        ),
        # (
        #     r"\b" + o_i_vars + r"(\s+)" + o_i_vars + r"\b",
        #     "[\\1]\\2[\\3]",
        # ),
        (
            r"(\s+|^)" + o_i_vars + r"(\s+)\[?" + o_i_vars + r"\]?(\s+|$)",
            "\\1[\\2]\\3[\\4]\\5",
        ),
        (
            r"(\s+|^)\[?" + o_i_vars + r"\]?(\s+)" + o_i_vars + r"(\s+|$)",
            "\\1[\\2]\\3[\\4]\\5",
        ),
    ]
    entity_rules = (
        default_num_only_rules(num_range)
        + (oh_is_zero_rules if oh_is_zero else [(r"\boh\b", "o")])
        + [
            (r"\bdouble(?: |-)(\w+|\d+)\b", "\\1 \\1"),
            (r"\btriple(?: |-)(\w+|\d+)\b", "\\1 \\1 \\1"),
            # (r"\b([a-zA-Z])\b", "\\1"),
        ]
        + (i_oh_limit_rules if i_oh_limit else [(r"\b([a-zA-Z])\b", "\\1")])
    )
    return entity_rules


def num_replacer(num_range=100, condense=True):
    entity_rules = default_num_rules(num_range)
    post_rules = [(r"[^0-9]", "")] if condense else []
    replacer, keeper = entity_replacer_keeper(
        entity_rules=entity_rules, post_rules=post_rules
    )
    return replacer


def num_keeper(num_range=100):
    entity_rules = default_num_rules(num_range)
    pre_rules = [(r"[ ;,.]", " ")]
    post_rules = []
    replacer, keeper = entity_replacer_keeper(
        pre_rules=pre_rules, entity_rules=entity_rules, post_rules=post_rules
    )
    return keeper


def alnum_replacer(
    num_range=100, oh_is_zero=False, i_oh_limit=True, condense=True
):
    entity_rules = default_alnum_rules(
        num_range, oh_is_zero, i_oh_limit=i_oh_limit
    )
    # entity_rules = default_num_rules(num_range)
    pre_rules = [
        (r"[ ;,.]", " "),
        (r"[']", ""),
        # (
        #     r"((?:(?<=\w{2,2})|^)\s*)(?:\bI\b|\bi\b|\bOh\b|\boh\b)(\s*(?:\w{2,}|$))",
        #     "",
        # ),
    ]

    def upper_case(match_obj):
        char_elem = match_obj.group(0)
        return char_elem.upper()

    post_rules = (
        (
            (
                [
                    (r"(\s|^)(?:o|O|I|i)(\s|$)", "\\1\\2"),
                    (r"\[(\w)\]", "\\1"),
                ]
                if i_oh_limit
                else []
            )
            + [
                # (r"\b[a-zA-Z]+\'[a-zA-Z]+\b", ""),
                (r"\b[a-zA-Z]{2,}\b", ""),
                (r"[^a-zA-Z0-9]", ""),
                (r"([a-z].*)", upper_case),
            ]
        )
        if condense
        else []
    )
    replacer, keeper = entity_replacer_keeper(
        pre_rules=pre_rules, entity_rules=entity_rules, post_rules=post_rules
    )
    return replacer


def alnum_keeper(num_range=100, oh_is_zero=False):
    entity_rules = default_alnum_rules(num_range, oh_is_zero, i_oh_limit=True)

    # def strip_space(match_obj):
    #     # char_elem = match_obj.group(1)
    #     return match_obj.group(1).strip() + match_obj.group(2).strip()

    pre_rules = [
        (r"[ ;,.]", " "),
        (r"[']", ""),
        # (
        #     r"((?:(?<=\w{2,2})|^)\s*)(?:\bI\b|\bi\b|\bOh\b|\boh\b)(\s*(?:\w{2,}|$))",
        #     strip_space,
        # ),
    ]

    post_rules = [
        # (
        #     r"((?:(?<=\w{2,2})|^)\s*)(?:\bI\b|\bi\b|\bOh\b|\boh\b)(\s*(?:\w{2,}|$))",
        #     strip_space,
        # )
    ]
    replacer, keeper = entity_replacer_keeper(
        pre_rules=pre_rules, entity_rules=entity_rules, post_rules=post_rules
    )
    return keeper


def num_keeper_orig(num_range=10, extra_rules=[]):
    num_int_map_ty = [
        (
            r"\b" + num2words(i) + r"\b",
            " " + str(i) + " ",
        )
        for i in reversed(range(num_range))
    ]
    re_rules = [
        (re.compile(k, re.IGNORECASE), v)
        for (k, v) in [
            # (r"[ ;,.]", " "),
            (r"\bdouble(?: |-)(\w+)\b", "\\1 \\1"),
            (r"\btriple(?: |-)(\w+)\b", "\\1 \\1 \\1"),
            (r"hundred", "00"),
            (r"\boh\b", " 0 "),
            (r"\bo\b", " 0 "),
        ]
        + num_int_map_ty
    ] + [(re.compile(k), v) for (k, v) in extra_rules]

    def merge_intervals(intervals):
        # https://codereview.stackexchange.com/a/69249
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                # test for intersection between lower and higher:
                # we know via sorting that lower[0] <= higher[0]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (
                        lower[0],
                        upper_bound,
                    )  # replace by merged interval
                else:
                    merged.append(higher)
        return merged

    # merging interval tree for optimal # https://www.geeksforgeeks.org/interval-tree/

    def keep_numeric_literals(w2v_out):
        # out = w2v_out.lower()
        out = re.sub(r"[ ;,.]", " ", w2v_out).strip()
        # out = " " + out.strip() + " "
        # out = re.sub(r"double (\w+)", "\\1 \\1", out)
        # out = re.sub(r"triple (\w+)", "\\1 \\1 \\1", out)
        num_spans = []
        for (k, v) in re_rules:  # [94:]:
            matches = k.finditer(out)
            for m in matches:
                # num_spans.append((k, m.span()))
                num_spans.append(m.span())
            # out = re.sub(k, v, out)
        merged = merge_intervals(num_spans)
        num_ents = len(merged)
        keep_out = " ".join((out[s[0] : s[1]] for s in merged))
        return keep_out, num_ents

    return keep_numeric_literals


def infer_num_replacer(num_range=100, condense=True):
    entity_rules, vocab = infer_num_rules_vocab(num_range)
    corrector = vocab_corrector_gen(vocab)
    post_rules = [(r"[^0-9]", "")] if condense else []
    replacer, keeper = entity_replacer_keeper(
        entity_rules=entity_rules, post_rules=post_rules
    )

    def final_replacer(x):
        return replacer(corrector(x))

    return final_replacer


def vocab_corrector_gen(vocab, distance=1, method="spell"):
    spell = spellchecker.SpellChecker(distance=distance)
    words_to_remove = set(spell.word_frequency.words()) - set(vocab)
    spell.word_frequency.remove_words(words_to_remove)
    spell.word_frequency.load_words(vocab)

    if method == "spell":

        def corrector(inp):
            # return " ".join(
            #     [spell.correction(tok) for tok in spell.split_words(inp)]
            # )
            return " ".join(
                [spell.correction(tok) for tok in inp.split()]
            )

    elif method == "edit":
        # editdistance.eval("banana", "bahama")

        def corrector(inp):
            match_dists = sorted(
                [(v, editdistance.eval(inp, v)) for v in vocab],
                key=lambda x: x[1],
            )
            return match_dists[0]

    else:
        raise ValueError(f"unsupported method:{method}")

    return corrector


if __name__ == "__main__":
    repl = infer_num_replacer()
    import pdb

    pdb.set_trace()
