import gzip
import json
from numpy import array
from pandas import DataFrame, read_csv
from os.path import expanduser
from itertools import permutations

types = ['hgnc_id', 'entrez_id', 'symbol']
enforce_ints = ['entrez_id']
ttypes = {'hgnc_id': int, 'entrez_id': int, 'symbol': str}


class GeneIdConverter(object):
    def __init__(self, fpath, types_list, enforce_int, lower_case=True):
        """

        :param fpath: file path to gzipped json
            json format is like the one taken from
                ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/json/hgnc_complete_set.json
        :param types_list: can be from
            ['hgnc_id', 'entrez_id', 'symbol', 'cosmic', 'ucsc_id']

        :param enforce_int: list of types for which type int is enforced, e.g. 'entrez_id'
        """

        with gzip.open(fpath, 'rb') as f:
            jsonic = json.loads(f.read().decode('utf-8'))
        self.convs = {}
        self.sets = {}
        types_perms = list(permutations(types_list, 2))

        for u in types_list:
            s = filter(lambda x: u in x.keys(), jsonic['response']['docs'])
            self.sets[u] = set(map(lambda x: int(x[u]) if u in enforce_int else x[u], s))

        for u, v in types_perms:
            ff = filter(lambda x: u in x.keys() and v in x.keys(), jsonic['response']['docs'])
            self.convs[u, v] = dict(map(lambda x: (int(x[u]) if u in enforce_int else x[u],
                                                   int(x[v]) if v in enforce_int else x[v]), ff))
            if lower_case:
                keys = list(self.convs[u, v].keys())
                vs = [self.convs[u, v][k] for k in keys]
                if ttypes[u] == str:
                    keys = [x.lower() for x in keys]
                if ttypes[v] == str:
                    vs = [x.lower() for x in vs]
                self.convs[u, v] = dict(zip(keys, vs))

        self.types_list = types_list
        self.types_perms = types_perms
        self.enforce_int = enforce_int

        self.u, self.v = types_list[0], types_list[1]

    def update_with_broad_symbols(self, verbose=False):
        # load gene df
        fname_gene = '~/data/lincs/GSE92742_Broad_LINCS_gene_info.txt.gz'
        gene_df = read_csv(expanduser(fname_gene), sep='\t')
        gene_df.rename(columns={'pr_gene_symbol': 'symbol_broad',
                                'pr_gene_id': 'entrez_id_broad'}, inplace=True)

        types_perms_update = list(permutations(['symbol', 'entrez_id'], 2))
        gene_df['symbol_broad'] = gene_df['symbol_broad'].apply(lambda x: x.lower())
        for u, v in types_perms_update:
            # find such entrez_broad that are not in gc['entrez'] already
            # forget about the sets for now
            cdict = self.convs[u, v]
            # self.view(self.u, self)
            if verbose:
                print('Size of {0} : {1} dict is {2}'.format(u, v, len(self.convs[u, v])))
            sset = set(self.convs[u, v].keys())
            mask = gene_df[u + '_broad'].apply(lambda x: x not in sset)
            if verbose:
                print('New keys : {0} like {1}'.format(sum(mask),
                                                       gene_df.loc[mask, [u + '_broad', v + '_broad']].head().values))
            extra_dict = dict(gene_df.loc[mask, [u + '_broad', v + '_broad']].values)
            cdict.update(extra_dict)
            if verbose:
                print('Updated size of {0} : {1} dict is {2}'.format(u, v, len(self.convs[u, v])))

    def choose_converter(self, u, v):
        if u in self.types_list and v in self.types_list:
            self.u, self.v = u, v

    def keys(self):
        return self.convs[(self.u, self.v)].keys()

    def change_case(self, key, mode='upper'):
        # there is not check if key field is str
        transform_foo = lambda x: x.upper() if mode == 'upper' else x.lower()
        if key in self.types_list:
            extra_keys = [k for k in self.types_list if k != key]
            for ek in extra_keys:
                self.convs[(key, ek)] = {transform_foo(k): v for k, v in self.convs[(key, ek)].items()}
                self.convs[(ek, key)] = {k: transform_foo(v) for k, v in self.convs[(ek, key)].items()}

    def to_pd_df(self):
        arr = array([(u, v) for u, v in self.convs[(self.u, self.v)].items()])
        return DataFrame(arr, columns=(self.u, self.v))

    def values(self):
        return self.convs[(self.u, self.v)].values()

    def update(self, update_dict):
        self.convs[(self.u, self.v)].update(update_dict)

    def view(self, k=5, u=None, v=None):
        if not u:
            u = self.u
        if not v:
            v = self.v
        return list(self.convs[(u, v)].items())[:k]

    def __getitem__(self, key):
        return self.convs[(self.u, self.v)][key]
