from sklearn.preprocessing import normalize
import numpy as np

rcp = 'magface100_recomp'
pcp = 'magface100'
fns_recomp = np.load(f'{rcp}/filenames_lfw.npy')
emb_recomp = np.load(f'{rcp}/embeddings_lfw.npy')
fns = np.load(f'{pcp}/filenames_lfw.npy')
emb = np.load(f'{pcp}/embeddings_lfw.npy')
fns_recomp_matched = lambda x: str.replace(x, '_data/images', '.')
fns_recomp = np.array(list(map(fns_recomp_matched, fns_recomp)))

n_emb_recomp = normalize(emb_recomp)
n_emb = normalize(emb)

simis = []
for fn in fns:
    try:
        idx_a = np.where(fn == fns_recomp)[0][0]
        idx_b = np.where(fn == fns)[0][0]
        simi = np.dot(n_emb_recomp[idx_a], n_emb[idx_b])
        simis.append(simi)
    except:
        pass

simis = np.array(simis)
print(f'mean={simis.mean()}')
print(f'min={simis.min()}')