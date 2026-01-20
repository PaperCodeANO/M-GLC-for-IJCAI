import pickle

with open('tox_10_motif.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print(len(loaded_dict.keys()))