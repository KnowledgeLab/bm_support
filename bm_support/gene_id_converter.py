import gzip
import json
from itertools import permutations


class GeneIdConverter(object):

    def __init__(self, fpath, types_list, enforce_int):
        """

        :param fpath:
        :param types_list: can be from
            ['hgnc_id', 'entrez_id', 'symbol', 'cosmic', 'ucsc_id']

        :param enforce_int: list of types for which type int is enforced, e.g. 'entrez_id'
        """

        with gzip.open(fpath, 'rb') as f:
            jsonic = json.loads(f.read().decode('utf-8'))
        self.convs = {}
        types_perms = list(permutations(types_list, 2))

        for u, v in types_perms:
            ff = filter(lambda x: u in x.keys() and v in x.keys(), jsonic['response']['docs'])
            self.convs[u, v] = dict(map(lambda x: (int(x[u]) if u in enforce_int else x[u],
                                                   int(x[v]) if v in enforce_int else x[v]), ff))

        self.types_list = types_list
        self.types_perms = types_perms
        self.enforce_int = enforce_int

        self.u, self.v = types_list[0], types_list[1]

    def choose_converter(self, u, v):
        if u in self.types_list and v in self.types_list:
            self.u, self.v = u, v

    def __getitem__(self, key):
        return self.convs[(self.u, self.v)][key]