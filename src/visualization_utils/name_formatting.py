import difflib


def get_non_overlapping_parts(next_name, prev_name):
    matches = difflib.SequenceMatcher(None, next_name, prev_name).get_matching_blocks()

    next_name_cut = ''
    next_start = 0
    prev_name_cut = ''
    prev_start = 0

    for (n, p, length) in matches:
        next_name_cut, next_start = get_outer_string_and_next_start(n, length, next_start, next_name_cut, next_name)
        prev_name_cut, prev_start = get_outer_string_and_next_start(p, length, prev_start, prev_name_cut, prev_name)

    return next_name_cut, prev_name_cut


def get_outer_string_and_next_start(n, length, start, cur_cuts, full_string):
    cur_cuts += full_string[start:n]
    next_start = n + length
    return cur_cuts, next_start


def get_overlap(next_name, prev_name, junks=False):
    matches = difflib.SequenceMatcher(None, next_name, prev_name).get_matching_blocks()

    string_matches = [next_name[n:n + length] for (n, p, length) in matches]

    return (' ' if junks else '').join(string_matches)


def remove_overlapping_string_junks_from_split(strings_with_junks, split_char='-'):
    overlapping_junks = get_overlapping_string_junks_from_all_by_split_string(strings_with_junks, split_char)

    def remove_overlapping_junks_from_string(string_junks):
        return split_char.join([junk for junk in string_junks.split(split_char) if junk not in overlapping_junks])

    return [remove_overlapping_junks_from_string(s) for s in strings_with_junks]


def get_overlapping_string_junks_from_all_by_split_string(strings, split_char):
    overlapping_algorithm_names = strings[0].split(split_char)

    for string_with_junks in strings[1:]:
        for prev_overlapping_string in reversed(range(len(overlapping_algorithm_names))):
            if overlapping_algorithm_names[prev_overlapping_string] not in string_with_junks.split(split_char):
                del overlapping_algorithm_names[prev_overlapping_string]

    return overlapping_algorithm_names


def get_last_two_upper_case_letters_except_first(alg_name):
    all_upper_characters = ''.join([c for c in alg_name if c.isupper()])
    return all_upper_characters[1:][-2:].lower()
