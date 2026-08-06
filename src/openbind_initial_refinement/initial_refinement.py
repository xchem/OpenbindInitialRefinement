import dataclasses

import gemmi

@dataclasses.dataclass
class ResID:
    chain: str
    seqid: str

    def __repr__(self):
        return f'{self.chain}/{self.seqid}'

    def __hash__(self):
        return f'{self.chain}{self.seqid}'

def drop_atoms(st, resids: list[ResID]) -> gemmi.Structure:
    new_st = st.clone()
    for chain in st[0]:
        del new_st[0][chain.name]

    resids_hashable = [(resid.chain, resid.seqid) for resid in resids]
    
    for chain in st[0]:
        new_chain = gemmi.Chain(chain.name)
        for res in chain:
            if (chain.name, str(res.seqid.num)) in resids_hashable:
                    continue
            else:
                new_chain.add_residue(res)
        new_st[0].add_chain(new_chain)
    
    return new_st