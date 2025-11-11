import numpy as np
import pandas as pd
import rpy2.robjects as robjects

###########################
### R utility functions ###
###########################
robjects.r('''
    read_cross <- function(filename) {
        library(qtl)
        genomap <- jittermap(pull.map(read.cross(format="csvr", file=filename)))
        return(genomap)
    }
    map_snp_order <- function(genomap) {
        mapSNPorder <- names(genomap[[1]])
        for (i in 2:20) {
            mapSNPorder <- c(mapSNPorder, names(genomap[[i]]))
        }
        return(mapSNPorder)
    }
    sim_cross <- function(genomap, n_progeny) {
        library(qtl)
        fake_cross <- sim.cross(map = genomap, n.ind = n_progeny, type = "riself", map.function = "morgan")$geno
        progeny_matrix <- do.call(cbind, lapply(fake_cross, function(chr) chr$data))
        return(progeny_matrix)
    }
''')

read_cross_func = robjects.r['read_cross'] # Input is the path of a genetic map with empty. Returns a genetic map robject.
map_snp_order_func = robjects.r['map_snp_order'] # Input is a genetic map robject. Returns an r array of marker names ordered as the genetic map. Better to recast as python list.
_sim_cross_func = robjects.r['sim_cross'] # Input is a genetic map robject and the number of progeny to simulate. Return an r matrix where each row corresponds to a simulated progeny. Each component is either 1 or 2 indicating the parent.

########################################################
### Simulate progenies genotypes for a cross p1 x p2 ###
########################################################
def sim_cross_with_genos(p1_geno, p2_geno, n_progeny, genmap):
    '''
    # Design choice: supports -1/0/1 encoding only.
    Design choice: supports different encodings. Naively copy values from parents without modification.

    Parameters
    ----------
    p1_geno, p2_geno: pandas series
    genmap: robject genetic map object from qtl package

    Return:
    -------
    # numpy array of shape (n_progeny, len(p1_geno))
    Pandas dataframe of shape (n_progeny, len(p1_geno))
    '''
    progeny_mtx = np.array(_sim_cross_func(genmap, n_progeny))
    # res = np.zeros(progeny_mtx.shape)
    res = [[0] * progeny_mtx.shape[1] for _ in range(progeny_mtx.shape[0])]
    for i in range(progeny_mtx.shape[0]):
        for j in range(progeny_mtx.shape[1]):
            source_parent = progeny_mtx[i, j]
            # val = p1_geno.iloc[j] if source_parent == 1 else p2_geno.iloc[j]
            # res[i,j] = val if val != 0 else np.random.choice(a=[-1, 0, 1], p=[15/32, 1/16, 15/32])
            # res[i,j] = p1_geno.iloc[j] if source_parent == 1 else p2_geno.iloc[j]
            res[i][j] = p1_geno.iloc[j] if source_parent == 1 else p2_geno.iloc[j]
    return pd.DataFrame(columns=p1_geno.index, data=res)
