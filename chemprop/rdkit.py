from rdkit import Chem

def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.
    
    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize = False)
        Chem.SanitizeMol(mol, sanitizeOps = Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def make_polymer_mol(smiles: str, keep_h: bool, add_h: bool, fragment_weights: list):
    """
    Builds an RDKit molecule from a SMILES string.

    :param smiles: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param fragment_weights: List of monomer fractions for each fragment in s. Only used when input is a polymer.
    :return: RDKit molecule.
    """

    # check input is correct, we need the same number of fragments and their weights
    num_frags = len(smiles.split('.'))
    if len(fragment_weights) != num_frags:
        raise ValueError(f'number of input monomers/fragments ({num_frags}) does not match number of '
                         f'input number of weights ({len(fragment_weights)})')

    # if it all looks good, we create one molecule object per fragment, add the weight as property
    # of each atom, and merge fragments into a single molecule object
    mols = []
    for s, w in zip(smiles.split('.'), fragment_weights):
        m = make_mol(s, keep_h, add_h)
        for a in m.GetAtoms():
            a.SetDoubleProp('w_frag', float(w))
        mols.append(m)

    # combine all mols into single mol object
    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)

    return mol
