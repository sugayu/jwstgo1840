'''Test of jwstgo1840.nirspec.background.
'''

from ...nirspec import background


##
def test_get_amplifier_patterns():
    patterns = background.get_amplifier_patterns()
    for i, p in enumerate(patterns):
        y = int(250 + 500 * i)
        assert p[y, 1000]
