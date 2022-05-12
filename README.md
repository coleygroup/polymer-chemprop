# wD-MPNN for Polymer Property Prediction
This repository contains the weighted, directed message passing neural network (wD-MPNN) used for polymer property prediction in the paper [A graph representation of molecular ensembles for polymer property prediction](#). For the reference Chemprop code, please go to https://github.com/chemprop/chemprop.

## Usage
For general Chemprop usage, please refer to its documentation: https://chemprop.readthedocs.io/en/latest/.

When providing polymer structures as input rather than molecules, add the `--polymer` flag to the `chemprop_train` call, e.g.:

```
chemprop_train --data_path input.csv --dataset_type regression --save_dir chemprop_checkpoints --polymer
```

In the example above, the file `input.csv` would contain an extended string representation that includes information on the average repeating unit of the polymer. This is an example of such input:

```
[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1ccc([*:4])c(N)c1|0.25|0.75|<1-3:0.25:0.25<1-4:0.25:0.25<2-3:0.25:0.25<2-4:0.25:0.25<1-2:0.25:0.25<3-4:0.25:0.25<1-1:0.25:0.25<2-2:0.25:0.25<3-3:0.25:0.25<4-4:0.25:0.25
```

* The first part (`[*:1]c1ccc2c(c1)S(=O)(=O)c1cc([*:2])ccc1-2.[*:3]c1ccc([*:4])c(N)c1`) contains the SMILES strings for all monomers in polymer. In this example there are 2 monomers separated by a `.`. 
* The relative abundance of each monomer, reflecting their stoichiometry, is provided in the substring `|0.25|0.75|`, which indicates a ratio of 1:3 for the first and second monomers provided in the SMILES string, respectively. 
* In the SMILES string, numbered wildcards (e.g., `[*:3]`) are used to indicate the location of possible connections within or between monomers. All these possible extra bonds, and their weights (i.e., frequencies) are specified in `<1-3:0.25:0.25<1-4:0.25:0.25< ...`. The information for each possible bond (in the example above we have 10 possible bonds, both within and between monomers) is specified after a `<` symbols, e.g. `<1-3:0.25:0.75`. This would mean that there can be a bond between the atom connecting to `[*:1]` and the atom connecting to `[*:3]`, that the directed edge `1->3` has a weight of 0.25, and that the directed edge `3->1` has a weight of 0.75. If both the forward and backward edge have the same probability of occurring, the same weight is simply repeated while still separating the weights with `:`, e.g., `<1-3:0.5:0.5`.
* The degree of polymerization can also be provided as part of the input, by appending it at the end of the input string after the symbol `~`, e.g. `...<3-4:0.5:0.5~502.5`. This will be internally transformed into the weighting factor `1 + log(Xn)` discussed in the paper.

## Citation
If you use this code, please cite:

```
@misc{wdmpnn,
      title={A graph representation of molecular ensembles for polymer property prediction}, 
      author={Matteo Aldeghi and Connor W. Coley},
      year={2022},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={}
}
```
