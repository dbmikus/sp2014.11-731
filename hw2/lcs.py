# Longest common substring
# Source code from:
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python


def LCS(s1, s2):
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    string_start = x_longest - longest
    string_end = x_longest
    return (s1[string_start : string_end], string_start, string_end)
