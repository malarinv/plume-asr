import re


def entity_replacer_keeper(pre_rules=[], entity_rules=[], post_rules=[]):
    # def replacer_keeper_gen():
    pre_rules_c = [(re.compile(k), v) for (k, v) in pre_rules]
    entity_rules_c = [(re.compile(k, re.IGNORECASE), v) for (k, v) in entity_rules]
    post_rules_c = [(re.compile(k), v) for (k, v) in post_rules]

    re_rules = pre_rules_c + entity_rules_c + post_rules_c

    def replacer(w2v_out):
        out = w2v_out
        for (k, v) in re_rules:
            out = k.sub(v, out)
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

    # merging interval tree for optimal # https://www.geeksforgeeks.org/interval-tree/

    def keep_literals(w2v_out):
        # out = re.sub(r"[ ;,.]", " ", w2v_out).strip()
        out = w2v_out
        for (k, v) in pre_rules_c:
            out = k.sub(v, out)
        num_spans = []
        for (k, v) in entity_rules_c:  # [94:]:
            matches = k.finditer(out)
            for m in matches:
                # num_spans.append((k, m.span()))
                num_spans.append(m.span())
            # out = re.sub(k, v, out)
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
            ("\\bninety-nine\\b", "99"),
            ("\\bninety-eight\\b", "98"),
            ("\\bninety-seven\\b", "97"),
            ("\\bninety-six\\b", "96"),
            ("\\bninety-five\\b", "95"),
            ("\\bninety-four\\b", "94"),
            ("\\bninety-three\\b", "93"),
            ("\\bninety-two\\b", "92"),
            ("\\bninety-one\\b", "91"),
            ("\\bninety\\b", "90"),
            ("\\beighty-nine\\b", "89"),
            ("\\beighty-eight\\b", "88"),
            ("\\beighty-seven\\b", "87"),
            ("\\beighty-six\\b", "86"),
            ("\\beighty-five\\b", "85"),
            ("\\beighty-four\\b", "84"),
            ("\\beighty-three\\b", "83"),
            ("\\beighty-two\\b", "82"),
            ("\\beighty-one\\b", "81"),
            ("\\beighty\\b", "80"),
            ("\\bseventy-nine\\b", "79"),
            ("\\bseventy-eight\\b", "78"),
            ("\\bseventy-seven\\b", "77"),
            ("\\bseventy-six\\b", "76"),
            ("\\bseventy-five\\b", "75"),
            ("\\bseventy-four\\b", "74"),
            ("\\bseventy-three\\b", "73"),
            ("\\bseventy-two\\b", "72"),
            ("\\bseventy-one\\b", "71"),
            ("\\bseventy\\b", "70"),
            ("\\bsixty-nine\\b", "69"),
            ("\\bsixty-eight\\b", "68"),
            ("\\bsixty-seven\\b", "67"),
            ("\\bsixty-six\\b", "66"),
            ("\\bsixty-five\\b", "65"),
            ("\\bsixty-four\\b", "64"),
            ("\\bsixty-three\\b", "63"),
            ("\\bsixty-two\\b", "62"),
            ("\\bsixty-one\\b", "61"),
            ("\\bsixty\\b", "60"),
            ("\\bfifty-nine\\b", "59"),
            ("\\bfifty-eight\\b", "58"),
            ("\\bfifty-seven\\b", "57"),
            ("\\bfifty-six\\b", "56"),
            ("\\bfifty-five\\b", "55"),
            ("\\bfifty-four\\b", "54"),
            ("\\bfifty-three\\b", "53"),
            ("\\bfifty-two\\b", "52"),
            ("\\bfifty-one\\b", "51"),
            ("\\bfifty\\b", "50"),
            ("\\bforty-nine\\b", "49"),
            ("\\bforty-eight\\b", "48"),
            ("\\bforty-seven\\b", "47"),
            ("\\bforty-six\\b", "46"),
            ("\\bforty-five\\b", "45"),
            ("\\bforty-four\\b", "44"),
            ("\\bforty-three\\b", "43"),
            ("\\bforty-two\\b", "42"),
            ("\\bforty-one\\b", "41"),
            ("\\bforty\\b", "40"),
            ("\\bthirty-nine\\b", "39"),
            ("\\bthirty-eight\\b", "38"),
            ("\\bthirty-seven\\b", "37"),
            ("\\bthirty-six\\b", "36"),
            ("\\bthirty-five\\b", "35"),
            ("\\bthirty-four\\b", "34"),
            ("\\bthirty-three\\b", "33"),
            ("\\bthirty-two\\b", "32"),
            ("\\bthirty-one\\b", "31"),
            ("\\bthirty\\b", "30"),
            ("\\btwenty-nine\\b", "29"),
            ("\\btwenty-eight\\b", "28"),
            ("\\btwenty-seven\\b", "27"),
            ("\\btwenty-six\\b", "26"),
            ("\\btwenty-five\\b", "25"),
            ("\\btwenty-four\\b", "24"),
            ("\\btwenty-three\\b", "23"),
            ("\\btwenty-two\\b", "22"),
            ("\\btwenty-one\\b", "21"),
            ("\\btwenty\\b", "20"),
            ("\\bnineteen\\b", "19"),
            ("\\beighteen\\b", "18"),
            ("\\bseventeen\\b", "17"),
            ("\\bsixteen\\b", "16"),
            ("\\bfifteen\\b", "15"),
            ("\\bfourteen\\b", "14"),
            ("\\bthirteen\\b", "13"),
            ("\\btwelve\\b", "12"),
            ("\\beleven\\b", "11"),
            ("\\bten\\b", "10"),
            ("\\bnine\\b", "9"),
            ("\\beight\\b", "8"),
            ("\\bseven\\b", "7"),
            ("\\bsix\\b", "6"),
            ("\\bfive\\b", "5"),
            ("\\bfour\\b", "4"),
            ("\\bthree\\b", "3"),
            ("\\btwo\\b", "2"),
            ("\\bone\\b", "1"),
            ("\\bzero\\b", "0"),
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
        (r"\boh\b", " 0 "),
        (r"\bo\b", " 0 "),
        (r"\bdouble(?: |-)(\w+|\d+)\b", "\\1 \\1"),
        (r"\btriple(?: |-)(\w+|\d+)\b", "\\1 \\1 \\1"),
    ]
    return entity_rules


def default_alnum_rules(num_range, oh_is_zero):
    oh_is_zero_rules = [
        (r"\boh\b", "0"),
        (r"\bo\b", "0"),
    ]
    entity_rules = (
        default_num_only_rules(num_range)
        + (oh_is_zero_rules if oh_is_zero else [(r"\boh\b", "o")])
        + [
            (r"\bdouble(?: |-)(\w+|\d+)\b", "\\1 \\1"),
            (r"\btriple(?: |-)(\w+|\d+)\b", "\\1 \\1 \\1"),
            (r"\b([a-zA-Z])\b", "\\1"),
        ]
    )
    return entity_rules


def num_replacer(num_range=100, condense=True):
    entity_rules = default_num_rules(num_range)
    post_rules = [(r"[^0-9]", "")] if condense else []
    # post_rules = []
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


def alnum_replacer(num_range=100, oh_is_zero=False, condense=True):
    entity_rules = default_alnum_rules(num_range, oh_is_zero)
    # entity_rules = default_num_rules(num_range)
    pre_rules = [(r"[ ;,.]", " "), (r"[']", "")]

    def upper_case(match_obj):
        char_elem = match_obj.group(0)
        return char_elem.upper()

    post_rules = (
        [
            # (r"\b[a-zA-Z]+\'[a-zA-Z]+\b", ""),
            (r"\b[a-zA-Z]{2,}\b", ""),
            (r"[^a-zA-Z0-9]", ""),
            (r"([a-z].*)", upper_case),
        ]
        if condense
        else []
    )
    replacer, keeper = entity_replacer_keeper(
        pre_rules=pre_rules, entity_rules=entity_rules, post_rules=post_rules
    )
    return replacer


def alnum_keeper(num_range=100, oh_is_zero=False):
    entity_rules = default_alnum_rules(num_range, oh_is_zero)
    pre_rules = [(r"[ ;,.]", " "), (r"[']", "")]
    post_rules = []
    replacer, keeper = entity_replacer_keeper(
        pre_rules=pre_rules, entity_rules=entity_rules, post_rules=post_rules
    )
    return keeper


def test_num():
    num_extractor = num_replacer()
    keeper = num_keeper()
    num_only_replacer = num_replacer(condense=False)
    assert num_extractor("thirty-two") == "32"
    assert num_extractor("not thirty-two fifty-nine") == "3259"
    assert num_extractor(" triPle 5 fifty 3") == "555503"
    assert num_only_replacer(" triPle 5 fifty 3") == " 5 5 5 50 3"
    assert num_extractor("douBle 2 130") == "22130"
    assert num_extractor("It is a One fifty eIght 5 fifty ") == "1508550"
    assert (
        num_only_replacer(" It is  a  One fifty eIght 5 fifty ")
        == " It is  a  1 50 8 5 50 "
    )
    assert num_extractor("One fifty-eight 5 oh o fifty") == "15850050"
    assert keeper(
        "my phone number is One hundred fifty-eight not 5 oh o fifty more"
    ) == ("One hundred fifty-eight 5 oh o fifty", 7)


def test_alnum():
    extractor_oh = alnum_replacer(oh_is_zero=True)
    extractor = alnum_replacer()
    keeper = alnum_keeper()
    only_replacer = alnum_replacer(condense=False)
    assert extractor("I'm thirty-two") == "32"
    assert extractor("a thirty-two") == "A32"
    assert extractor("not a b thirty-two fifty-nine") == "AB3259"
    assert extractor(" triPle 5 fifty 3") == "555503"
    assert only_replacer(" triPle 5 fifty 3") == " 5 5 5 50 3"
    assert extractor("douBle 2 130") == "22130"
    assert extractor("It is a One b fifty eIght A Z 5 fifty ") == "A1B508AZ550"
    assert (
        only_replacer(" It's  a ;  One b fifty eIght A Z 5 fifty ")
        == " Its  a    1 b 50 8 A Z 5 50 "
    )
    assert (
        only_replacer(" I'm is  a  One b fifty eIght A Z 5 fifty ")
        == " Im is  a  1 b 50 8 A Z 5 50 "
    )
    assert extractor("One Z fifty-eight 5 oh o b fifty") == "1Z585OOB50"
    assert extractor_oh("One Z fifty-eight 5 oh o b fifty") == "1Z58500B50"
    assert keeper(
        "I'll phone number One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == ("One hundred n fifty-eight 5 oh o fifty A B", 10)
    assert keeper("I'm One hundred n fifty-eight not 5 oh o fifty A B more") == (
        "One hundred n fifty-eight 5 oh o fifty A B",
        10,
    )

    assert keeper("I am One hundred n fifty-eight not 5 oh o fifty A B more") == (
        "I One hundred n fifty-eight 5 oh o fifty A B",
        11,
    )
