from plume.utils import (
    num_replacer,
    num_keeper,
    alnum_replacer,
    alnum_keeper,
    random_segs,
)
import numpy
import random as rand
import pytest


def test_num_replacer_keeper():
    num_extractor = num_replacer()
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
    keeper = num_keeper()
    assert keeper(
        "my phone number is One hundred fifty-eight not 5 oh o fifty more"
    ) == ("One hundred fifty-eight 5 oh o fifty", 7)


def test_alnum_replacer():
    extractor_oh = alnum_replacer(oh_is_zero=True)
    extractor = alnum_replacer()
    only_replacer = alnum_replacer(condense=False)
    assert extractor("5 oh i c 3") == "5OIC3"
    assert extractor("I am, oh it is 3. I will") == "3"
    assert extractor("I oh o 3") == "IOO3"
    assert extractor("I will 3 I") == "3I"
    assert extractor("I'm thirty-two") == "32"
    assert extractor("I am thirty-two") == "32"
    assert extractor("I j thirty-two") == "IJ32"
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
    assert (
        extractor("I One hundred n fifty-eight not 5 oh o fifty A B more")
        == "I100N585OO50AB"
    )


def test_alnum_keeper():
    keeper = alnum_keeper()
    assert keeper("I One hundred n fifty-eight not 5 oh o fifty A B more") == (
        "I One hundred n fifty-eight 5 oh o fifty A B",
        11,
    )
    assert keeper(
        "I'll phone number One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == ("One hundred n fifty-eight 5 oh o fifty A B", 10)
    assert keeper(
        "I'm One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == (
        "One hundred n fifty-eight 5 oh o fifty A B",
        10,
    )

    assert keeper(
        "I am One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == (
        "One hundred n fifty-eight 5 oh o fifty A B",
        10,
    )


def test_alpha_keeper():
    keeper = alnum_keeper()
    assert keeper("I One hundred n fifty-eight not 5 oh o fifty A B more") == (
        "I One hundred n fifty-eight 5 oh o fifty A B",
        11,
    )
    assert keeper(
        "I'll phone number One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == ("One hundred n fifty-eight 5 oh o fifty A B", 10)
    assert keeper(
        "I'm One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == (
        "One hundred n fifty-eight 5 oh o fifty A B",
        10,
    )

    assert keeper(
        "I am One hundred n fifty-eight not 5 oh o fifty A B more"
    ) == (
        "One hundred n fifty-eight 5 oh o fifty A B",
        10,
    )


@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)


def test_random_segs(random):
    segs = random_segs(100000, 1000, 3000)

    def segs_comply(segs, min, max):
        for (start, end) in segs:
            if end - start < min or end - start > max:
                return False
        return True

    assert segs_comply(segs, 1000, 3000) == True
