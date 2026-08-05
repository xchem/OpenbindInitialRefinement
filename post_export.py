import argparse

def remove_weak_waters():
    ...

def reciprocal_space_refinement():
    ...

def get_density_fit_stats():
    ...

def get_structural_model_stats():
    ...

def main():
    # Get rid of weak waters with findwaters
    new_structure_path = remove_weak_waters()

    # Do an initial reciprocal refinement
    new_structure_path = reciprocal_space_refinement(new_structure_path)

    # Check for bad density with gemmi.blobs
    density_fit_stats = get_density_fit_stats(new_structure_path)

    # Check for bad model with molprobity 
    structural_model_stats = get_structural_model_stats(new_structure_path)

    ...

if __name__ == "__main__":
    ...