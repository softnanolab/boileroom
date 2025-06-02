import os
from typing import Union
from io import StringIO
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile


# TODO: for now, we can keep it in desprot, but long-term it makes sense to
# have it as a default part of boileroom (as a function that can be run locally)
def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    assert isinstance(pdb_path, (str, StringIO)), "pdb_path must be a string or StringIO"
    if isinstance(pdb_path, str):
        assert os.path.exists(pdb_path), "pdb_path must be a valid path"
    return PDBFile.read(pdb_path).get_structure(model=1)


def pdb_string_to_atomarray(pdb_string: str) -> AtomArray:
    assert isinstance(pdb_string, str), "pdb_string must be a string"
    return pdb_file_to_atomarray(StringIO(pdb_string))
