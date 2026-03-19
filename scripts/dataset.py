import os
import pandas as pd
from pathlib import Path

from rdkit import Chem

import torch
import sklearn
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def canonicalize_rxn(rxn):
    canonicalized_smiles = []
    r, p = rxn.split('>>')
    for s in [r, p]:
        mol = Chem.MolFromSmiles(s)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        canonicalized_smiles.append(Chem.MolToSmiles(mol))
    return '>>'.join(canonicalized_smiles)

class USPTODataset(object):
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('%s/labeled_data.csv' % args['data_dir'])
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'val'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        self.smiles = df['Products'].tolist()
        self.masks = df['Mask'].tolist()
        self.labels = [eval(t) for t in df['Labels']]
        saved_graphs_dir = PROJECT_ROOT / 'data' / 'saved_graphs'
        saved_graphs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_path = str(saved_graphs_dir / ('%s_dglgraph.bin' % args['dataset']))
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            print('Processing dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('\rProcessing molecule %d/%d' % (i+1, len(self.smiles)), end='', flush=True)
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            print ()
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.labels[item], self.masks[item]

    def __len__(self):
            return len(self.smiles)
        
class USPTOTestDataset(object):
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv(str(PROJECT_ROOT / 'data' / args['dataset'] / 'raw' / 'raw_test.csv'))
        self.rxns = df['reactants>reagents>production'].tolist()
        self.rxns = [canonicalize_rxn(rxn) for rxn in self.rxns]
        self.smiles = [rxn.split('>>')[-1] for rxn in self.rxns]
        saved_graphs_dir = PROJECT_ROOT / 'data' / 'saved_graphs'
        saved_graphs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file_path = str(saved_graphs_dir / ('%s_test_dglgraph.bin' % args['dataset']))
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            print('Loading previously saved test dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            print('Processing test dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('Processing molecule %d/%d' % (i+1, len(self.smiles)))
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
            return self.smiles[item], self.graphs[item], self.rxns[item]

    def __len__(self):
            return len(self.smiles)