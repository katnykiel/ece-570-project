from pymatgen.core import Structure
from mp_api.client import MPRester
import csv

# Load Materials Project API key
with open(os.path.expanduser("~/.mpkey.txt"), "r") as f:
    key = f.readlines()[0]

# Query structures and stablilty using Materials Project API
with MPRester(key) as m:
    data = m.materials.summary.search(
        num_chunks=10,
        chunk_size=1000,
        num_sites=[6, 12],
        fields=["structure", "is_stable"],
    )

# Pull out the required data
structures = [doc.structure.as_dict() for doc in data]
stables = [doc.is_stable for doc in data]

# save structures and stabilites to file
with open("structures.json", "w") as f:
    json.dump(structures, f)
