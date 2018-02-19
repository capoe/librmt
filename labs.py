#! /usr/bin/env python
import librmt as rmt
import numpy as np
import csv
import json
import math
import scipy.cluster.hierarchy as sch
from sklearn.cluster import SpectralClustering

VERBOSE = False

# TODO mpf_kernel: communicate kernel transform through options (mpf + cached descriptor)

# =======================
# HIERARCHICAL CLUSTERING
# =======================

def hierarchical_clustering(
        distance_matrix,
        min_size=2,
        method='centroid',
        log=None, 
        verbose=False):
    # Compute linkage matrix L:
    # L[i] = [ c1, c2, d12, n12 ] 
    # Read this as: cluster i merged from clusters c1 and c2, 
    # with cluster distance d12, and number of member nodes n12
    N = distance_matrix.shape[0]
    ii = np.triu_indices(N, k=1) # upper triangle without diagonal
    distances_compressed = distance_matrix[ii]
    if log: log << "Hierarchical clustering ..." << log.endl
    #distances_compressed = np.random.uniform(size=distances_compressed.shape) # HACK
    #L = sch.linkage(distance_matrix, method='centroid') # NOTE BUG: Distance matrix not interpreted as such!!
    #L = sch.linkage(distances_compressed, method='single') #, method='centroid', metric="*should not be used*")
    L = sch.linkage(distances_compressed, method='centroid') #, method='centroid', metric="*should not be used*")
    if log: log << "... done." << log.endl
    # Unwrap clusters from linkage matrix
    if log: log << "Unwrapping clusters ..." << log.endl
    clusters = []
    # Add leaf nodes as single-node clusters
    for i in range(distance_matrix.shape[0]):
        cluster = {
            "idx": i,
            "nodes": [i],
            "len": 1,
            "parents": [ None, None ],
        }
        clusters.append(cluster)
    for i in range(L.shape[0]):
        cidx = distance_matrix.shape[0]+i
        c1 = int(L[i][0]+0.5)
        c2 = int(L[i][1]+0.5)
        nodes = clusters[c2]["nodes"] + clusters[c1]["nodes"]
        clusters.append({
            "idx": cidx,
            "nodes": nodes,
            "len": len(nodes),
            "parents": [ c1, c2 ]
        })
        if log and verbose:
            log << "Linkage record:" << L[i] << log.endl
            log << "Parent clusters" << log.endl
            p0 = clusters[cidx]["parents"][0]
            p1 = clusters[cidx]["parents"][1]
            log << clusters[p0]["nodes"] << log.endl
            log << clusters[p1]["nodes"] << log.endl
            log << "Cluster" << cidx << len(clusters[cidx]["nodes"]) << clusters[cidx]["nodes"] << log.endl
    if log: log << "... done." << log.endl
    clusters = sorted(clusters, key=lambda c: c["len"])
    if verbose and log:
        for cidx, c in enumerate(clusters):
            log << "Cluster %d |c| %d" % (cidx, len(c["nodes"])) << log.endl
    return filter(lambda c: len(c["nodes"]) >= min_size, clusters)

def dendrogram_plot(D, tags):
    import scipy
    import pylab
    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(16,16))
    ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
    Y = sch.linkage(D, method='centroid')
    Z1 = sch.dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
    Y = sch.linkage(D, method='centroid') # 'single'
    Z2 = sch.dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    #axmatrix.set_xticks([])
    #axmatrix.set_yticks([])
    axmatrix.tick_params(labelsize=5.0)
    axmatrix.set_xticks(np.arange(D.shape[0]))
    axmatrix.set_yticks(np.arange(D.shape[0]))
    axmatrix.set_xticklabels(tags[idx1], rotation=90.)
    axmatrix.set_yticklabels(tags[idx2], rotation=0.)
    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    fig.show()
    fig.savefig('dendrogram.svg')

def fps(D, n_select, i0):
    # Farthest-point sampling of a total of n_select points 
    # with seed indices i0 and distance matrix D
    if type(i0) != list: i0 = [ i0 ]
    mask = np.zeros((D.shape[0],), 'bool')
    idcs_selected = i0
    distances = [ 0.0 for i in range(len(i0)) ]
    for i in range(n_select-len(i0)):
        D_red = D[idcs_selected]
        D_avg = np.average(D_red, axis=0)
        idcs = np.argsort(D_avg)[::-1]
        for j in idcs:
            if j in idcs_selected:
                pass
            else:
                idcs_selected.append(j)
                distances.append(D_avg[j])
                break
    mask[idcs_selected] = True
    return idcs_selected, np.where(mask == False)[0], distances

def split_mpf_target(IX, Y, levels, 
        min_interval, deep_mp=False, mp_options={}, idcs=None, 
        filters=[], this_level=1, split="average", log=None):
    # Split threshold
    if type(split) != str:
        log << "Hierarchical decomposition using" << split << log.endl
        y_split = split[0]
        split = split[1]
    elif split == "average":
        y_split = np.average(Y)
    elif split == "median":
        y_split = np.median(Y)
    else: assert False # 'median' or 'average'
    y_min = np.min(Y)
    y_max = np.max(Y)
    if y_max - y_min < min_interval: return filters
    if type(idcs) == type(None):
        idcs = np.arange(IX.shape[0])
    idcs_0 = np.where(Y <= y_split)
    idcs_1 = np.where(Y > y_split)
    IX_0 = IX[idcs_0]
    IX_1 = IX[idcs_1]
    Y_0 = Y[idcs_0]
    Y_1 = Y[idcs_1]
    if log:
        log << "   "*(this_level-1) + "Level %d: analysing split @ %+1.2f  [%4d, %4d]" % (this_level, y_split, IX_0.shape[0], IX_1.shape[0]) << log.endl
    if IX_0.shape[0] < 2 or IX_1.shape[0] < 2:
        return filters
    IX_0, X_mean_0, X_std_0 = rmt.utils.zscore(IX_0)
    IX_1, X_mean_1, X_std_1 = rmt.utils.zscore(IX_1)
    if mp_options["append_filters_mp"]:
        if deep_mp == False:
            # Lower partition
            L_0, V_0, _X_mean_0, _X_std_0 = mp_transform(IX_0, False, False, log)
            # Upper partition
            L_1, V_1, _X_mean_1, _X_std_1 = mp_transform(IX_1, False, False, log)
        else:
            V_0 = hpcamp(IX_0, mp_options, log=log if VERBOSE else None)
            L_0 = np.array([])
            V_1 = hpcamp(IX_1, mp_options, log=log if VERBOSE else None)
            L_1 = np.array([])
        # NOTE HACK TO CHECK THAT/WHETHER PCs ARE A BETTER CHOICE THAN RANDOM VECTORS
        #V_0 = rmt.utils.generate_orthonormal_vectors(V_0.shape[1], V_0.shape[0]).T
        #V_1 = rmt.utils.generate_orthonormal_vectors(V_0.shape[1], V_0.shape[0]).T
        filters.append({ 
            "idcs": idcs[idcs_0], "L": L_0, "n_filters": V_0.shape[1], 
            "V": V_0, "mu": X_mean_0, "std": X_std_0, "y_min": y_min, "y_max": y_split, 
            "partition": 0, "level": this_level })
        filters.append({ 
            "idcs": idcs[idcs_1], "L": L_1, "n_filters": V_1.shape[1], 
            "V": V_1, "mu": X_mean_1, "std": X_std_1, 
            "y_min": y_split, "y_max": y_max, 
            "partition": 1, "level": this_level })

    if ("append_filters_mp_bimodal" in mp_options) and mp_options["append_filters_mp_bimodal"]:
        if deep_mp: raise NotImplementedError("TODO mp_bimodal=True and deep_mp=True")
        avg_0 = np.average(Y_0)
        avg_1 = np.average(Y_1)
        idcs_00 = np.where(Y < avg_0)[0]
        idcs_11 = np.where(Y > avg_1)[0]
        idcs_01 = np.concatenate([idcs_00, idcs_11])
        IX_01 = IX[idcs_01]
        Y_01 = Y[idcs_01]
        IX_01, X_mean_01, X_std_01 = rmt.utils.zscore(IX_01)
        L_01, V_01, _X_mean_01, _X_std_01 = mp_transform(IX_01, False, False, log)
        filters.append({ 
            "idcs": idcs[idcs_01], "L": L_01, "n_filters": V_01.shape[1], 
            "V": V_01, "mu": X_mean_01, "std": X_std_01, 
            "y_min": y_min, "y_max": y_max, 
            "partition": -1, "level": this_level })

    if mp_options["append_filter_centres"]:
        x_moment_2 = mp_options["x_moment_2"] # np.std(IX, axis=0)**2
        x_moment_4 = mp_options["x_moment_4"] # np.average(IX**4, axis=0)
        msd_threshold_k = mp_options["msd_threshold_coeff"]
        filter_centre_msd_0 = X_mean_0.dot(X_mean_0)
        filter_centre_msd_1 = X_mean_1.dot(X_mean_1)
        log << "Filter centre msd" << filter_centre_msd_0 << " | " << filter_centre_msd_1 << log.endl
        def sample_variance(d, N, s2, s4):
            return np.sum(1./N*s2), np.sum( (s4 - 3*s2*s2)/N**3 + 2*s2*s2/N**2)
        msd_0, msd_var_0 = sample_variance(
            IX_0.shape[1], IX_0.shape[0], x_moment_2, x_moment_4)
        msd_1, msd_var_1 = sample_variance(
            IX_1.shape[1], IX_1.shape[0], x_moment_2, x_moment_4)
        # Add as filter if MSD deemed significant
        filter_centre_msd_threshold_0 = msd_0 + msd_threshold_k*msd_var_0**0.5
        filter_centre_msd_threshold_1 = msd_1 + msd_threshold_k*msd_var_1**0.5

        if mp_options["normalise_filter_centres"]:
            X_mean_0 = X_mean_0/np.dot(X_mean_0, X_mean_0)**0.5
            X_mean_1 = X_mean_1/np.dot(X_mean_1, X_mean_1)**0.5

        if filter_centre_msd_0 >= filter_centre_msd_threshold_0:
            filters.append({
                "idcs": None, "L": None, "n_filters": 1,
                "V": np.array([X_mean_0]).T, "mu": np.zeros(X_mean_0.shape), "std": np.ones(X_mean_0.shape), "y_min": 0, "y_max": 0, 
                "partition": 0, "level": this_level })
        if filter_centre_msd_1 >= filter_centre_msd_threshold_1:
            filters.append({
                "idcs": None, "L": None, "n_filters": 1,
                "V": np.array([X_mean_1]).T, "mu": np.zeros(X_mean_1.shape), "std": np.ones(X_mean_1.shape), "y_min": 0, "y_max": 0, 
                "partition": 1, "level": this_level })

    # Descend into next level(s)
    if this_level < levels:
        filters = split_mpf_target(IX_0, Y_0, deep_mp=deep_mp, split=split,
            mp_options=mp_options, levels=levels, min_interval=min_interval, 
            idcs=idcs[idcs_0], filters=filters, this_level=this_level+1, log=log)
        filters = split_mpf_target(IX_1, Y_1, deep_mp=deep_mp, split=split,
            mp_options=mp_options, levels=levels, min_interval=min_interval, 
            idcs=idcs[idcs_1], filters=filters, this_level=this_level+1, log=log)
    return filters

def hierarchical_clustering_target(Y, levels, idcs=None, 
        clusters=[], this_level=1, parent=-1, split="average", log=None):
    # Split threshold
    if split == "average":
        y_split = np.average(Y)
    elif split == "median":
        y_split = np.median(Y)
    else: assert False # 'median' or 'average'
    y_min = np.min(Y)
    y_max = np.max(Y)
    if type(idcs) == type(None):
        idcs = np.arange(Y.shape[0])
    idcs_0 = np.where(Y <= y_split)[0]
    idcs_1 = np.where(Y > y_split)[0]
    Y_0 = Y[idcs_0]
    Y_1 = Y[idcs_1]
    if log:
        log << "   "*(this_level-1) + "Level %d: split @ %+1.2f  [%4d, %4d]" % (this_level, y_split, Y_0.shape[0], Y_1.shape[0]) << log.endl
    if Y_0.shape[0] < 2 or Y_1.shape[0] < 2:
        return clusters
    cidx_0 = len(clusters)
    cidx_1 = cidx_0+1

    clusters.append({ 
        "y_min": y_min, "y_max": y_split,
        "idx": cidx_0, "nodes": idcs[idcs_0], "len": len(idcs_0), "parent": parent })
    clusters.append({ 
        "y_min": y_split, "y_max": y_max,
        "idx": cidx_1, "nodes": idcs[idcs_1], "len": len(idcs_1), "parent": parent })

    # Descend into next level(s)
    if this_level < levels:
        clusters = hierarchical_clustering_target(
            Y_0, levels=levels, idcs=idcs[idcs_0], clusters=clusters, this_level=this_level+1, log=log, parent=cidx_0)
        clusters = hierarchical_clustering_target(
            Y_1, levels=levels, idcs=idcs[idcs_1], clusters=clusters, this_level=this_level+1, log=log, parent=cidx_1)
    return clusters

# =========
# FILTERING
# =========

def mp_transform(IX, norm_avg, norm_std, log, hist=False):
    # Computes MP thresholded PC space based on cov(IX)
    mp_gamma = float(IX.shape[1])/IX.shape[0]
    IX_norm_pca, IZ, X_mean, X_std, S, L, V = rmt.pca_compute(
        IX=IX,
        log=None,
        norm_div_std=norm_std,
        norm_sub_mean=norm_avg,
        eps=0.,
        ddof=0)
    if log: log << rmt.dist_mp_bounds(mp_gamma) << log.endl
    idcs_eigen_signal = np.where(L.diagonal() > rmt.dist_mp_bounds(mp_gamma)[1])[0]
    n_dim_mp_signal = len(idcs_eigen_signal)
    if log: log << "[mptf] Dim reduction %d -> %d with gamma_mp = %+1.2f" % (
        IX.shape[1], n_dim_mp_signal, mp_gamma) << log.endl
    L_signal = L.diagonal()[idcs_eigen_signal]
    V_signal = V[:,idcs_eigen_signal]
    if hist:
        hist, bin_edges = np.histogram(a=L.diagonal(), bins=100, density=True)
        hist_signal, bin_edges = np.histogram(a=L_signal, bins=bin_edges, density=True)
        bin_centres = np.zeros((bin_edges.shape[0]-1,), dtype='float64')
        for i in range(bin_edges.shape[0]-1):
            bin_centres[i] = 0.5*(bin_edges[i]+bin_edges[i+1])
        mp_sample = rmt.dist_mp_sample(bin_centres, mp_gamma)
        np.savetxt('out.pca_hist.txt', np.array([ bin_centres, hist, mp_sample, hist_signal ]).T)
        np.savetxt('out.pca_eigv.txt', L.diagonal())
    return L_signal, V_signal, X_mean, X_std

def hpcamp(IX, options, log):
    # PARAMETERS
    # - target subspace dimension ("take")
    # - minimum cluster size ("min_cluster_size")
    # TODO
    # - Improve FPS routine, taking into account 
    # ... angle between PCAs
    # ... subspace overlap (angle between subspace directors)
    # ... eigenvalue magnitude (difference?)
    # OPTIONS
    file_debug = False
    verbose = False if log else False
    options = options["hpcamp_transform"]
    if "hc_method" in options:
        hc_method = options["hc_method"]
    else:
        hc_method = "centroid"
    min_cluster_size = options["min_cluster_size"]
    take = options["target_dim"]
    k_c_max = options["cluster_similarity_threshold"]
    sequence_sim_cutoff = options["filter_seq_sim_cutoff"]
    sparsify = options["sparsify"]
    do_fps = options["do_fps"]

    # ==========
    # CLUSTERING
    # ==========

    # Covariance-based clustering
    N = IX.shape[0]
    D = IX.shape[1]
    covmat = np.cov(IX, rowvar=False, ddof=0)
    dmat = (1. - covmat**2)
    clusters_all = hierarchical_clustering(
        distance_matrix=dmat, min_size=1, method=hc_method, log=log, verbose=False)
    clusters = filter(lambda c: len(c["nodes"]) >= min_cluster_size, clusters_all)
    # Calculate cluster subspace similarity matrix
    C = np.zeros((len(clusters), IX.shape[1]), dtype='float64')
    for cidx, cluster in enumerate(clusters):
        C[cidx, cluster["nodes"]] = 1.0
    C = (C.T/np.sqrt(np.sum(C*C, axis=1))).T
    K_C = C.dot(C.T)
    if verbose:
        print "Cluster similarity matrix"
        print K_C

    # =================
    # FILTER EXTRACTION
    # =================

    clusters_analysed = []
    filter_idcs = []
    filter_coeffs = []
    filter_lambdas = []
    filter_largest = []
    filter_cluster = []
    for c in clusters_all:
        c["filters"] = []
        c["skip"] = False
    for cidx, cluster in enumerate(clusters):
        n_fss = cluster["len"]
        fss = float(n_fss)/D
        gamma = float(n_fss)/N
        is_largest = (cidx == len(clusters) - 1)
        # Proceed only if subspace overlap with previous analysed clusters is sufficiently small
        if is_largest: pass
        elif len(clusters_analysed):
            k_c = K_C[cidx, clusters_analysed]
            k_max = np.max(k_c)
            if k_max > k_c_max:
                cluster["skip"] = True
                cluster["filters"] = clusters[clusters_analysed[np.argmax(k_c)]]["filters"]
                if verbose: log << "Cluster %4d    gamma*=%1.3f fss*=%1.2f d*=%4d    Skip with k_max=%+1.4f" % (
                    cidx, gamma, fss, n_fss, k_max) << log.endl
                continue
        else: pass
        clusters_analysed.append(cidx)
        # Subsample fingerprint components to match this cluster
        idcs_ss = sorted(cluster["nodes"])
        if verbose: log << idcs_ss << log.endl
        # Slice descriptor matrix
        IX_ss = IX[:,idcs_ss]
        # MP filtering
        L, V, mu, std = mp_transform(
            IX=IX_ss, norm_avg=False, norm_std=False, log=log if verbose else None)
        if verbose: log << "Projecting onto MP space: shape(v) =" << V.shape << log.endl
        # Itemise filters
        n_filters = V.shape[1]
        off = len(filter_lambdas)
        # ... Keep track of PCs of largest cluster to seed FPS
        if is_largest:
            for j in range(n_filters):
                filter_largest.append(off+j)
        # ... Add to PC pool
        for j in range(n_filters):
            if verbose:
                print "adding", idcs_ss
                print "coeffs", V[:,j]
            filter_idcs.append(idcs_ss)
            filter_coeffs.append(V[:,j])
            filter_lambdas.append(L[j])
            filter_cluster.append(cluster["idx"])
            cluster["filters"].append(off+j)
        if log: log << "Cluster %4d [%4d]    gamma*=%1.3f fss*=%1.2f d*=%4d    %4d filters" % (cidx, cluster["idx"], gamma, fss, n_fss, len(filter_idcs)) << log.endl
        #log << "@fss=%+1.2f <> @gamma=%+1.2f: generated %d filters" % (fss, float(n_fss)/N, len(filter_idcs)) << log.endl
    # ASSEMBLE FILTER MATRIX
    n_filters = len(filter_idcs)
    V_tf = np.zeros((IX.shape[1], n_filters), dtype=IX.dtype)
    P_tf = np.zeros((IX.shape[1], n_filters), dtype=IX.dtype)
    for j in range(n_filters):
        V_tf[filter_idcs[j], j] = filter_coeffs[j]
        P_tf[filter_idcs[j], j] = 1.

    # =================================
    # SUBSELECT FILTERS USING SEQUENCES
    # =================================

    if sparsify == "sequence":
        # TRACE PCs THROUGH CLUSTER HIERARCHY
        map_cluster = { c["idx"]: c for cidx, c in enumerate(clusters) }
        map_cluster_all = { c["idx"]: c for cidx, c in enumerate(clusters_all) }
        filter_origin = {}
        for c in reversed(clusters):
            if c["skip"]: continue # skip cluster for which no decomposition was made
            # Retrieve PCs (=filters) for this cluster via filter indices
            p0 = c["parents"][0]
            p1 = c["parents"][1]
            parent_fidcs = c["filters"]
            child_1_fidcs = map_cluster_all[p0]["filters"]
            child_2_fidcs = map_cluster_all[p1]["filters"]
            child_fidcs = child_1_fidcs + child_2_fidcs
            # Initialise filter-to-filter map for filters of this parent cluster
            for fidx in parent_fidcs:
                filter_origin[fidx] = []
            # TEST ASSIGNMENT BETWEEN PARENT AND CHILD PCs FOR EACH CLUSTER
            V_child_1 = V_tf[:, child_1_fidcs]
            V_child_2 = V_tf[:, child_2_fidcs]
            V_parent = V_tf[:, parent_fidcs]
            V_children = V_tf[:, child_fidcs]
            #if V_child_1.shape[1] > 0 and V_child_2.shape[1] > 0:
            #    k_c1c2 = V_child_1.T.dot(V_child_2)
            if V_children.shape[1] > 0:
                #k_c12c12 = V_children.T.dot(V_children)
                k_par_child = np.abs(V_parent.T.dot(V_children))
                # Find relations using similarity (= vector overlap) cutoff
                rels = np.where(k_par_child > sequence_sim_cutoff)
                for i0, i1 in zip(rels[0], rels[1]):
                    fidx_0 = parent_fidcs[i0]
                    fidx_1 = child_fidcs[i1]
                    filter_origin[fidx_0].append(fidx_1)
        if verbose:
            print "Filter to filter map", filter_origin 
        # Unwrap filter-to-filter map into disjoint sequences
        assigned = {}
        sequences = []
        for fidx in reversed(range(n_filters)):
            if fidx in assigned: continue
            seq = []
            fidx_child = fidx
            while True:
                seq.append(fidx_child)
                fidx_child_list = filter_origin[fidx_child]
                if fidx_child_list == []:
                    break
                elif len(fidx_child_list) > 1:
                    assert False # Single-child policy, similarity cutoff chosen too small?
                else:
                    fidx_child = fidx_child_list[0]
                assert fidx_child not in assigned
                assigned[fidx_child] = True
            if verbose: 
                log << "Add sequence: length %d" % len(seq) << log.endl
                print seq
            sequences.append(seq)
        sequences = sorted(sequences, key=lambda s: len(s))
        if log: 
            log << "Have %d filter sequences" % (len(sequences)) << log.endl
            for idx, seq in enumerate(sequences):
                log << len(seq)
            log << log.endl
        if file_debug:
            ofs = open('sequences.txt', 'w')
            sizes = [ len(f) for f in filter_idcs ]
            for seq in sequences:
                for fidx in seq:
                    ofs.write('%d %+1.7e\n' % (sizes[fidx], filter_lambdas[fidx]))
                ofs.write('\n')
            ofs.close()
        # Select sequence heads (and/or tails depending on options)
        idcs_sel = []
        for s in sequences:
            for sel in options["sequence_select"]:
                if sel < 0 and len(s) < 1-sel:
                    pass
                elif len(s) >= sel+1:
                    idcs_sel.append(s[sel])
        if file_debug:
            ofs = open('seq-heads.txt', 'w')
            for fidx in idcs_sel:
                ofs.write('%d %+1.7e\n' % (sizes[fidx], filter_lambdas[fidx]))
            ofs.close()
        filter_lambdas = np.array(filter_lambdas)[idcs_sel]
        filter_idcs = np.array(filter_idcs)[idcs_sel]
        V_tf = V_tf[:,idcs_sel]

    # ==========================
    # SUBSELECT FILTERS WITH FPS
    # ==========================

    #dim_overlap = P_tf.T.dot(P_tf) # NOTE This matrix can be very large => memory issues
    if V_tf.shape[1] > take and do_fps:
        # Distance matrix for FPS
        K = (V_tf.T.dot(V_tf))**2
        D = (1. - K**2 + 1e-10)**0.5
        if file_debug: 
            import soap
            positions = soap.soapy.dimred_matrix('kernelpca', kmat=K, distmat=D, outfile="") 
            sizes = [ len(f) for f in filter_idcs ]
            ofs = open('mds_filter_size.txt', 'w')
            for s in range(len(sizes)):
                ofs.write("%+1.7e %+1.7e %d %+1.2f\n" % (positions[s][0], positions[s][1], sizes[s], filter_lambdas[s]))
            ofs.close()
        # Determine root components
        if options["fps_root"] == "largest_pc":
            i0 = np.argmax(filter_lambdas)
        elif options["fps_root"] == "largest_cluster":
            i0 = filter_largest # list of PC idcs extracted for largest cluster
        else: raise ValueError(options["fps_root"])
        idcs_sel, idcs_disc, distances = fps(D, take, i0)
        for idx, d in enumerate(distances):
            log << "Filter %2d" % idx << "  Distance %1.2f" % d << "  Subspace dim %3d" % len(filter_idcs[idcs_sel[idx]]) << log.endl
        if file_debug:
            pos_sel = positions[idcs_sel]
            ofs = open('mds-sel.txt', 'w')
            for s in idcs_sel:
                ofs.write("%+1.7e %+1.7e %d %+1.2f\n" % (positions[s][0], positions[s][1], sizes[s], filter_lambdas[s]))
            ofs.close()
        # Slice filter mapping matrix accordingly
        filter_lambdas = np.array(filter_lambdas)[idcs_sel]
        V_tf = V_tf[:,idcs_sel]
        if file_debug:
            amplitudes = np.sum(V_tf*V_tf, axis=1)
            np.savetxt('filter-amplitudes.txt', amplitudes)
        log << "Reducing to %d fps filters" % V_tf.shape[1] << log.endl
    """
    elif take > 0:
        filter_lambdas = np.array(filter_lambdas)
        order = np.argsort(filter_lambdas)[::-1]
        filter_lambdas = filter_lambdas[order]
        if verbose:
            print "Ordered projections"
            print filter_lambdas[0:take]
        V_tf = V_tf[:,order[0:take]]
        amplitudes = np.sum(V_tf*V_tf, axis=1)
        np.savetxt('tmp.txt', amplitudes)
        log << "Reducing to %d ranked filters" % V_tf.shape[1] << log.endl
    """
    if V_tf.shape[1] < 1:
        log << log.my << "ERROR HPCAMP did not return any filters" << log.endl
        raise RuntimeError()
    return V_tf

# ================
# STATE TRANSFORMS
# ================

def hpcamp_transform(state, options, log):
    # Transform
    V_tf = hpcamp(state["IX_train"], options, log)
    # Project
    state["IX_train"] = state["IX_train"].dot(V_tf)
    state["IX_test"] = state["IX_test"].dot(V_tf)
    # Normalise
    norm = np.std(state["IX_train"], axis=0, keepdims=True)
    mean = np.sum(state["IX_train"], axis=0, keepdims=True)/state["IX_train"].shape[0]
    state["IX_train"] = (state["IX_train"]-mean)/norm
    state["IX_test"] = (state["IX_test"]-mean)/norm
    log.prefix = log.prefix[0:-7]
    return state

def bipartite_mp_transform_htree(state, options, log):
    # TODO Handle sparsification
    min_size_cluster_y = options["bipartite_mp_transform_htree"]["min_cluster_size"]
    min_std_cluster_y = options["bipartite_mp_transform_htree"]["min_cluster_std"] # relative to std-dev of all targets
    k_c_max_y = options["bipartite_mp_transform_htree"]["max_cluster_sim"]
    # MP method
    deep_mp = options["bipartite_mp_transform_htree"]["deep_mp"]
    # Cluster origins as filters?
    append_filter_centres = options["bipartite_mp_transform_htree"]["append_filter_centres"]
    msd_threshold_k = options["bipartite_mp_transform_htree"]["msd_threshold_coeff"]

    verbose = False
    np.set_printoptions(precision=2)

    # POST-PROCESS OPTIONS
    IX = state["IX_train"]
    Y = state["T_train"]
    if min_size_cluster_y == None:
        min_size_cluster_y = int(0.1*IX.shape[0]+0.5)
    options_deep_mp = {
        # Clustering
        "min_cluster_size": int(0.1*IX.shape[1]+0.5),
        "cluster_similarity_threshold": 0.9,
        # Sequencing
        "sparsify": "sequence", # "sequence" or "fps"
        "filter_seq_sim_cutoff": 0.71, # 0.71, 0.9
        "sequence_select": [0], # [0] or [ 0, -1 ]
        # FPS
        "do_fps": False,
        "fps_root": "largest_pc", # "largest_cluster"
        "target_dim": 30,
    }

    # DESCRIPTOR DISTRIBUTION MOMENTS FOR LATER USE
    assert np.sum(np.sum(IX, axis=0)**2) < 1e-10 # Need to centre descriptor first before using this transformation
    x_moment_2 = np.std(IX, axis=0)**2
    x_moment_4 = np.average(IX**4, axis=0)
    log << "Moments:   <mu2> = %+1.7e   <mu4> = %+1.7e" % (np.average(x_moment_2), np.average(x_moment_4)) << log.endl

    # TARGET-BASED CLUSTERING
    dmat_y = np.abs(np.subtract.outer(Y, Y))
    clusters_y_all = hierarchical_clustering(
        distance_matrix=dmat_y, min_size=1, log=log, verbose=False)
    log << "Target-space: # clusters [all] =" << len(clusters_y_all) << log.endl
    clusters_y = filter(lambda c: len(c["nodes"]) >= min_size_cluster_y, clusters_y_all)
    log << "Target-space: # clusters [>s0] =" << len(clusters_y) << log.endl

    # Filter clusters based on std dev
    clusters_y_filtered = []
    std_Y = np.std(Y)
    for cidx, c in enumerate(clusters_y):
        Y_c = Y[c["nodes"]]
        std_Y_c = np.std(Y_c)
        c["y_avg"] = np.average(Y_c)
        c["y_std"] = std_Y_c
        if std_Y_c/std_Y > min_std_cluster_y:
            clusters_y_filtered.append(c)
    clusters_y_filtered = sorted(clusters_y_filtered, key=lambda c: -c["len"])
    log << "Target-space: # clusters [>dy] =" << len(clusters_y_filtered) << log.endl

    # Calculate cluster subspace similarity matrix
    C = np.zeros((len(clusters_y_filtered), Y.shape[0]), dtype='float64')
    for cidx, cluster in enumerate(clusters_y_filtered):
        C[cidx, cluster["nodes"]] = 1.0
    C = (C.T/np.sqrt(np.sum(C*C, axis=1))).T
    K_C = C.dot(C.T)

    # Initialise fields
    for c in clusters_y_all:
        c["filters"] = []
        c["filter_mu"] = None
        c["filter_sigma"] = None
        c["skip"] = False
    clusters_y_analysed = []
    clusters_y_analysed_idcs = []
    cluster_centre_filters = []

    # Analyse clusters
    for cidx, cluster in enumerate(clusters_y_filtered):
        is_largest = cluster["len"] == IX.shape[0] #(cidx == len(clusters_y_filtered) - 1)

        # Proceed only if subspace overlap with previous analysed clusters is sufficiently small
        if is_largest: pass
        elif len(clusters_y_analysed_idcs):
            k_c = K_C[cidx, clusters_y_analysed_idcs]
            k_max = np.max(k_c)
            if k_max > k_c_max_y:
                cluster["skip"] = True
                cluster["filter_sigma"] = clusters_y_filtered[clusters_y_analysed_idcs[np.argmax(k_c)]]["filter_sigma"]
                cluster["filter_mu"] = clusters_y_filtered[clusters_y_analysed_idcs[np.argmax(k_c)]]["filter_mu"]
                cluster["filters"] = clusters_y_filtered[clusters_y_analysed_idcs[np.argmax(k_c)]]["filters"]
                continue
        else: pass
        log << "Cluster %3d   size = %3d    <y> = %+1.2f +/- %+1.2f" % (
            cidx, cluster["len"], cluster["y_avg"], cluster["y_std"]) << log.endl
        if verbose:
            print "Cluster", cidx, "size", cluster["len"], "largest?", is_largest, "global idx", cluster["idx"], "parents", cluster["parents"]
            print "Cluster y avg =", cluster["y_avg"], "+/-", cluster["y_std"]
        clusters_y_analysed_idcs.append(cidx)
        clusters_y_analysed.append(cluster)

        # Slice descriptor matrix along sample axis.
        # Then determine mean and std-dev for each component
        idcs_ss = sorted(cluster["nodes"])
        IX_ss = IX[idcs_ss,:]
        filter_mu = np.average(IX_ss, axis=0)
        filter_std = np.std(IX_ss, axis=0)
        IX_ss = (IX_ss - filter_mu) #/filter_std
        IX_ss = rmt.utils.div0(IX_ss, filter_std)
        cluster["filter_mu"] = filter_mu
        cluster["filter_std"] = filter_std

        # TEST DISTANCE FROM ORIGIN
        if append_filter_centres:
            filter_centre_msd = filter_mu.dot(filter_mu)
            log << "Filter centre msd" << filter_centre_msd << log.endl
            def sample_variance(d, N, s2, s4):
                return np.sum(1./N*s2), np.sum( (s4 - 3*s2*s2)/N**3 + 2*s2*s2/N**2)
            msd, msd_var = sample_variance(
                IX_ss.shape[1], IX_ss.shape[0], x_moment_2, x_moment_4)
            # Add as filter if MSD deemed significant
            filter_centre_msd_threshold = msd + msd_threshold_k*msd_var**0.5
            log << "Centre MSD threshold" << filter_centre_msd_threshold << log.endl
            l0, l1 = rmt.dist_mp_bounds(float(IX_ss.shape[1])/IX_ss.shape[0])
            filter_centre_msd_threshold_mp = l1
            log << "Centre MSD threshold (MP)" << filter_centre_msd_threshold_mp << log.endl
            if filter_centre_msd >= filter_centre_msd_threshold:
                cluster_centre_filters.append(filter_mu)

        # EXTRACT FILTERS FROM SLICED DATA MATRIX
        if deep_mp:
            V = hpcamp(
                IX=IX_ss,
                options={ "hpcamp_transform": options_deep_mp },
                log=log)
            cluster["filters"] = V
        else:
            L, V, mu, std = mp_transform(
                IX=IX_ss,
                norm_avg=False, # done above
                norm_std=False, # done above
                log=log if verbose else None,
                hist=False)
            cluster["filters"] = V
        log << "Projecting onto MP space: shape(v) =" << V.shape << log.endl

    # COLLECT FILTERS
    V_tf = [] # NOTE Rows are the filter directions (mp_transform returns the transpose of this)
    mu_tf = []
    std_tf = []
    # From cluster mean
    ones = 1.*np.ones((IX.shape[1],))
    zeros = 1.*np.zeros((IX.shape[1],))
    for filt in cluster_centre_filters:
        filt = filt/np.dot(filt,filt)**0.5
        V_tf.append(filt)
        mu_tf.append(zeros)
        std_tf.append(ones)
    # From cluster variance
    for cluster in clusters_y_analysed:
        for i in range(cluster["filters"].shape[1]):
            V_tf.append(cluster["filters"][:,i])
            mu_tf.append(cluster["filter_mu"])
            std_tf.append(cluster["filter_std"])
    V_tf = np.array(V_tf)
    mu_tf = np.array(mu_tf)
    std_tf = np.array(std_tf)
    log << "Target-space: filter matrix" << V_tf.shape << log.endl

    # NOTE HACK
    #s_hs = []
    #for i in range(len(state["model"].models)):
    #    s_hs.append(state["model"].models[i].s_h)
    #V_tf = np.array(s_hs)
    #mu_tf = np.zeros(V_tf.shape)
    #std_tf = np.ones(V_tf.shape)
    #raw_input('...')

    # SHIFT, SCALE AND PROJECT
    IX_train_out = []
    IX_test_out = []
    for n in range(V_tf.shape[0]):
        #ix_train = ((state["IX_train"]-mu_tf[n])/std_tf[n]).dot(V_tf[n])
        ix_train = rmt.utils.div0(state["IX_train"]-mu_tf[n], std_tf[n]).dot(V_tf[n])
        #ix_test = ((state["IX_test"]-mu_tf[n])/std_tf[n]).dot(V_tf[n])
        ix_test = rmt.utils.div0(state["IX_test"]-mu_tf[n], std_tf[n]).dot(V_tf[n])
        IX_train_out.append(ix_train)
        IX_test_out.append(ix_test)
    state["IX_train"] = np.array(IX_train_out).T
    state["IX_test"] = np.array(IX_test_out).T

    # RENORMALISE
    state = rmt.normalise_descriptor_zscore(state, options, log)
    return state

def legendre_transform(state, options, log):
    log << log.mg << "Applying Legendre polynomial mapping" << log.endl
    # Apply Legendre mapping
    min_degree = options["legendre_transform"]["min_degree"] 
    max_degree = options["legendre_transform"]["max_degree"] 
    basis_type = options["legendre_transform"]["basis_type"]
    log << "Min degree, max degree =" << min_degree << "," << max_degree << log.endl
    # vvv TODO Is there a more elegant solution to this?
    scale = options["legendre_transform"]["scale"] # ix -> ix*scale in units of sigma (= 1.0)
    # ^^^
    d = state["IX_train"].shape[1]
    n_degree = max_degree-min_degree+1
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    IX_train_out = np.zeros(
        (state["IX_train"].shape[0], n_degree*d), 
        dtype=state["IX_train"].dtype)
    IX_test_out = np.zeros(
        (state["IX_test"].shape[0], n_degree*d), 
        dtype=state["IX_test"].dtype)
    # In principle monomials x, x**2, x**3, ... should perform equivalently
    if basis_type == "legendre":
        legendre = [
            lambda x: 1.0,                  # n = 0 (here not used)
            lambda x: x,                    # n = 1
            lambda x: 0.5*(3*x**2 - 1),     # ...
            lambda x: 0.5*(5*x**3 - 3*x),
            lambda x: 0.125*(35*x**4 - 30*x**2 + 3),
            lambda x: 0.125*(63*x**5 - 70*x**3 + 15*x),
            lambda x: 1./16.*(231*x**6 - 315*x**4 + 105*x**2 - 5),
            lambda x: 1./16.*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
        ]
    elif basis_type == "monomial":
        legendre = [
            lambda x: 1.0,                  # n = 0 (here not used)
            lambda x: x,                    # n = 1
            lambda x: x**2,
            lambda x: x**3,
            lambda x: x**4,
            lambda x: x**5,
            lambda x: x**6,
            lambda x: x**7
        ]
    else: assert False
    for n_degree in range(min_degree, max_degree+1):
        n = n_degree - min_degree
        ix_train_n = legendre[n_degree](scale*IX_train)
        ix_test_n = legendre[n_degree](scale*IX_test)
        IX_train_out[:,n*d:(n+1)*d] = ix_train_n
        IX_test_out[:,n*d:(n+1)*d] = ix_test_n
    log << "Input dimension before legendre transform: d =" << d << log.endl 
    log << "Output dimension after legendre transform: d'=" << IX_train_out.shape[1] << log.endl
    std = np.std(IX_train_out, axis=0)
    if len(std) > 0:
        log << "Standard deviation along components: min=%+1.2e avg=%+1.2e max=%+1.2e" % (np.min(std), np.average(std), np.max(std)) << log.endl
    if VERBOSE:
        log << "Standard deviation along components" << std << log.endl
    state["IX_train"] = IX_train_out
    state["IX_test"] = IX_test_out
    return state

def bipartite_mp_transform_btree(state, options, log):
    log << log.mg << "Binary y-tree MP transform" << log.endl
    n_levels = options["bipartite_mp_transform_btree"]["n_levels"]
    std_scale = options["bipartite_mp_transform_btree"]["min_interval"]
    deep_mp = options["bipartite_mp_transform_btree"]["deep_mp"]
    if "t_average" in state:
        t_average = state["t_average"]
    else:
        t_average = 0.0
    split = options["bipartite_mp_transform_btree"]["split"]
    if type(split) != str:
        log << "Correcting split instruction" << split << "for target shift" << t_average << log.endl
        split[0] = split[0] - t_average
    verbose = False
    T_train = state["T_train"]
    T_test = state["T_test"]
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    
    # CALCULATE OPTIONS FOR TREE-BASED MPF
    if deep_mp:
        min_clust_size = int(0.1*IX_train.shape[1]+0.5)
        if min_clust_size < 2:
            min_clust_size = 2
        log << "MP mode '%s':" % ("deep" if deep_mp else "standard") << "setting min_cluster_size=%d" % (min_clust_size) << log.endl
        mp_options = { 
          "hpcamp_transform": {
            # Clustering
            "min_cluster_size": min_clust_size,
            "cluster_similarity_threshold": 0.9,
            # Sequencing
            "sparsify": "sequence", # "sequence" or "fps"
            "filter_seq_sim_cutoff": 0.71, # 0.71,
            "sequence_select": [0], # [0] or [ 0, -1 ]
            # FPS
            "do_fps": False,
            "fps_root": "largest_pc", # "largest_cluster"
            "target_dim": 30 } }
    else:
        mp_options = {}

    mp_options["x_moment_2"] = np.std(IX_train, axis=0)**2
    mp_options["x_moment_4"] = np.average(IX_train**4, axis=0)
    mp_options["append_filters_mp"] = options["bipartite_mp_transform_btree"]["append_filters_mp"]
    mp_options["append_filters_mp_bimodal"] = options["bipartite_mp_transform_btree"]["append_filters_mp_bimodal"]
    mp_options["append_filter_centres"] = options["bipartite_mp_transform_btree"]["append_filter_centres"]
    mp_options["normalise_filter_centres"] = options["bipartite_mp_transform_btree"]["normalise_filter_centres"]
    mp_options["msd_threshold_coeff"] = options["bipartite_mp_transform_btree"]["msd_threshold_coeff"]

    # RECURSIVELY (=HIERARCHICALLY) APPLY MP FILTERING TO DATASET
    filters = split_mpf_target(IX_train, T_train, 
        min_interval=std_scale*np.std(T_train), 
        split=split,
        filters=[],
        levels=n_levels, 
        deep_mp=deep_mp,
        mp_options=mp_options,
        log=log)
    # Log filters if requested
    if "data_logger" in state:
        log << "Logging %d filters" % (len(filters)) << log.endl
        state["data_logger"].append({"tag": "mp_filters", "data": filters})

    # NOTE HACK
    #filters = []
    #for i in range(len(state["model"].models)):
    #    s_h = state["model"].models[i].s_h
    #    filters.append({ "mu": np.zeros(s_h.shape), "std": np.ones(s_h.shape), "V": np.array([s_h]).T, "level": 0, "partition": 0, "n_filters": 1, "y_min": 0, "y_max": 0 })
    #filters = []
    #J = None
    #for i in range(len(state["model"].models)):
    #    s_h = state["model"].models[i].s_h
    #    if type(J) == type(None):
    #        J = np.outer(s_h, s_h)
    #    else:
    #        J  = J + np.outer(s_h, s_h)
    #import numpy.linalg as la
    #w, V = la.eig(J)
    #print w
    ##V = rmt.utils.generate_orthonormal_vectors(V.shape[1], V.shape[0]).T
    #for i in range(V.shape[1]):
    #    s_h = V[:,i].real
    #    filters.append({ "mu": np.zeros(s_h.shape), "std": np.ones(s_h.shape), "V": np.array([s_h]).T, "level": 0, "partition": 0, "n_filters": 1, "y_min": 0, "y_max": 0 })
    #raw_input('...')

    IUs_train = []
    IUs_test = []
    IVs = []
    for filter_ in filters:
        # TODO Add filter centres "mu" as filters?
        log << "level:partition {level:02d}:{partition:02d}   # filters {n_filters:2d}   y-bounds [{y_min:+1.2f}, {y_max:+1.2f}]".format(**filter_) << log.endl
        if filter_["n_filters"] == 0: continue

        IU_train = rmt.utils.div0(IX_train-filter_["mu"], filter_["std"]).dot(filter_["V"])
        IU_test = rmt.utils.div0(IX_test-filter_["mu"], filter_["std"]).dot(filter_["V"])

        #IU_train = (IX_train-filter_["mu"]).dot(filter_["V"])
        #IU_test = (IX_test-filter_["mu"]).dot(filter_["V"])

        #IU_train = (IX_train-filter_["mu"]).dot((filter_["V"].T*filter_["std"]).T)
        #IU_test = (IX_test-filter_["mu"]).dot((filter_["V"].T*filter_["std"]).T)

        #IU_train = (IX_train-filter_["mu"]).dot(rmt.utils.div0(filter_["V"].T, filter_["std"]).T)
        #IU_test = (IX_test-filter_["mu"]).dot(rmt.utils.div0(filter_["V"].T, filter_["std"]).T)

        #IU_train = rmt.utils.div0(IX_train, filter_["std"]).dot(filter_["V"])
        #IU_test = rmt.utils.div0(IX_test, filter_["std"]).dot(filter_["V"])

        IUs_train.append(IU_train)
        IUs_test.append(IU_test)
        IVs.append(filter_["V"])
        if verbose:
            IU_check = ((IX_train[filter_["idcs"]]-filter_["mu"])/filter_["std"]).dot(filter_["V"])
            #IU_check = ((IX_train[filter_["idcs"]])/filter_["std"]).dot(filter_["V"])
            print "L", filter_["L"]
            print "std-total", np.std(IU_train, axis=0)**2
            print "var (should match L from above)", np.std(IU_check, axis=0)**2
            print "avg", np.average(IU_check, axis=0)
            raw_input('...')

    # CONCATENATE PROJECTED DESCRIPTORS
    if len(IUs_train) > 0:
        IU_train = np.concatenate(IUs_train, axis=1)
        IU_test = np.concatenate(IUs_test, axis=1)
        IV = np.concatenate(IVs, axis=1)
        if verbose:
            print "K-IV", IV.T.dot(IV)
            print IU_train.shape
            print IU_test.shape
            raw_input('___')
        state["IX_train"] = IU_train
        state["IX_test"] = IU_test
    else:
        print "WARNING No filters returned"
        state["IX_train"] = np.zeros((IX_train.shape[0],0))
        state["IX_test"] = np.zeros((IX_test.shape[0],0))
    return state

def mpf_kernel(state, options, log):
    ftag = "mpf_kernel"
    # OPTIONS
    normalise = options[ftag]["normalise"]
    tf1 = options[ftag]["tf1"]
    tf2 = options[ftag]["tf2"]
    w1 = options[ftag]["w1"]
    w2 = options[ftag]["w2"]
    #xi = 2 # NOTE xi = 2 works better?
    verbose = False
    # KERNEL ON MP FILTERS
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    K_train = IX_train.dot(IX_train.T)
    K_cross = IX_test.dot(IX_train.T)
    K_test = IX_test.dot(IX_test.T)
    if normalise:
        norm_train = K_train.diagonal()
        norm_test = K_test.diagonal()
        K_train = K_train/(np.outer(norm_train, norm_train)**0.5)
        K_test = K_test/(np.outer(norm_test, norm_test)**0.5)
        K_cross = K_cross/(np.outer(norm_test, norm_train)**0.5)
        if verbose:
            print "norm", norm_train
            print norm_test
    if verbose:
        print "K_train", K_train
        print "K_test", K_test
        print "K_cross", K_cross
        raw_input('...')
    #K_train = (0.5*(1.+K_train))**xi # NOTE
    #K_cross = (0.5*(1.+K_cross))**xi # NOTE
    K_train = tf1(K_train)
    K_cross = tf1(K_cross)
    # KERNEL ON CACHED DESCRIPTOR
    K2_train, K2_cross = rmt.kernel_dot_tf(
        state["IX_train_cached"], 
        state["IX_test_cached"], 
        { "tf": tf2, "normalise": normalise })
    if verbose:
        print "K2_train", K2_train
        print "K2_cross", K2_cross
        raw_input('...')
    # COMBINE KERNELS
    #w1 = 0.5
    #w2 = 0.5
    state["has_K"] = True
    state["K_train"] = w1*K_train+w2*K2_train
    state["K_test"]  = w1*K_cross+w2*K2_cross
    return state

def bipartite_mp_transform_simtree(state, options, log):
    log << log.mg << "Bipartite sim-tree MP transform" << log.endl
    n_levels = options["bipartite_mp_transform_simtree"]["n_levels"]
    std_scale = options["bipartite_mp_transform_simtree"]["min_interval"]
    deep_mp = options["bipartite_mp_transform_simtree"]["deep_mp"]
    split = options["bipartite_mp_transform_simtree"]["split"]
    verbose = False

    T_train = state["T_train"]
    T_test = state["T_test"]
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    x_moment_2 = np.std(IX_train, axis=0)**2
    x_moment_4 = np.average(IX_train**4, axis=0)

    # Kernel (similarity) matrix
    K_sim = IX_train.dot(IX_train.T)**2.0 # NOTE option
    K_sim = K_sim/np.outer(K_sim.diagonal(), K_sim.diagonal())**0.5
    D_sim = (1. - K_sim + 1e-10)**0.5

    K_sim = 0.5*(K_sim+K_sim.T)
    D_sim = 0.5*(D_sim+D_sim.T)

    # SPECTRAL CLUSTERING (TARGET-BASED)
    clusters = hierarchical_clustering_target(
        Y=T_train, 
        levels=n_levels, 
        split=split, 
        log=log)

    #s_hs = []
    #for i in range(len(state["model"].models)):
    #    s_hs.append(state["model"].models[i].s_h)
    #V_tf = np.array(s_hs)
    #mu_tf = np.zeros(V_tf.shape)
    #std_tf = np.ones(V_tf.shape)
    #print T_train[0]
    #x = IX_train[0]
    #print (-0.5*V_tf.dot(x)**2/x.shape[0])

    # EXTRACT FILTERS
    filters = []
    for cluster in clusters:
        #emat = -0.5*IX_train[cluster["nodes"]].dot(V_tf.T)**2/IX_train.shape[1]
        #print T_train[cluster["nodes"]]
        #print np.sum(emat, axis=1)
        #print np.sum(emat, axis=0)

        # Similarity matrix for this cluster
        D_cluster = D_sim[cluster["nodes"]][:,cluster["nodes"]]
        K_cluster = K_sim[cluster["nodes"]][:,cluster["nodes"]]
        print "Cluster", cluster["idx"], "of size", cluster["len"]

        # Subclustering
        n_clusters = 10
        if cluster["len"] < 5*n_clusters: continue
        clf = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = clf.fit_predict(K_cluster)
        
        # Extract filters from sub-clusters
        for i in range(n_clusters):
            idcs = cluster["nodes"][np.where(labels == i)[0]]
            print "Subcluster", i, "of size", len(idcs)
            if len(idcs) < 2: continue

            IX_i = IX_train[idcs]
            IX_i, X_mean_i, X_std_i = rmt.utils.zscore(IX_i)
            L_i, V_i, _X_mean_i, _X_std_i = mp_transform(IX_i, False, False, log=None)

            filters.append({
                "idcs": idcs, "L": L_i, "n_filters": V_i.shape[1],
                "V": V_i, "mu": X_mean_i, "std": X_std_i, "y_min": cluster["y_min"], "y_max": cluster["y_max"],
                "partition": i, "level": cluster["idx"]
            })
            if True: # NOTE Option
                msd_threshold_k = 3 # NOTE Option
                filter_centre_msd_i = X_mean_i.dot(X_mean_i)
                def sample_variance(d, N, s2, s4):
                    return np.sum(1./N*s2), np.sum( (s4 - 3*s2*s2)/N**3 + 2*s2*s2/N**2)
                msd_i, msd_var_i = sample_variance(
                    IX_i.shape[1], IX_i.shape[0], x_moment_2, x_moment_4)
                # Add as filter if MSD deemed significant
                filter_centre_msd_threshold_i = msd_i + msd_threshold_k*msd_var_i**0.5
                if filter_centre_msd_i >= filter_centre_msd_threshold_i:
                    filters.append({
                        "idcs": None, "L": None, "n_filters": 1,
                        "V": np.array([X_mean_i]).T, "mu": np.zeros(X_mean_i.shape), "std": np.ones(X_mean_i.shape), "y_min": 0, "y_max": 0, 
                        "partition": i, "level": cluster["idx"] })

            #emat = -0.5*IX_train[idcs].dot(V_tf.T)**2/IX_train.shape[1]
            #print len(idcs), np.average(emat, axis=0)

        #kclusters = hierarchical_clustering(D_cluster, method='single')
        #for clust in kclusters:
        #    idcs = cluster["nodes"][clust["nodes"]]
        #    emat = -0.5*IX_train[idcs].dot(V_tf.T)**2/IX_train.shape[1]
        #    print clust["len"], np.sum(emat, axis=0)
        #    #print clust["len"]
        #    raw_input('___')

    # NOTE HACK
    #filters = []
    #for i in range(len(state["model"].models)):
    #    s_h = state["model"].models[i].s_h
    #    filters.append({ "mu": np.zeros(s_h.shape), "std": np.ones(s_h.shape), "V": np.array([s_h]).T, "level": 0, "partition": 0, "n_filters": 1, "y_min": 0, "y_max": 0 })
    #raw_input('...')

    IUs_train = []
    IUs_test = []
    IVs = []
    for filter_ in filters:
        # TODO Add filter centres "mu" as filters?
        log << "level:partition {level:02d}:{partition:02d}   # filters {n_filters:2d}   y-bounds [{y_min:+1.2f}, {y_max:+1.2f}]".format(**filter_) << log.endl
        if filter_["n_filters"] == 0: continue
        IU_train = rmt.utils.div0(IX_train-filter_["mu"], filter_["std"]).dot(filter_["V"])
        IU_test = rmt.utils.div0(IX_test-filter_["mu"], filter_["std"]).dot(filter_["V"])
        #IU_train = ((IX_train)/filter_["std"]).dot(filter_["V"])
        #IU_test = ((IX_test)/filter_["std"]).dot(filter_["V"])
        IUs_train.append(IU_train)
        IUs_test.append(IU_test)
        IVs.append(filter_["V"])
        if verbose:
            IU_check = ((IX_train[filter_["idcs"]]-filter_["mu"])/filter_["std"]).dot(filter_["V"])
            #IU_check = ((IX_train[filter_["idcs"]])/filter_["std"]).dot(filter_["V"])
            print "L", filter_["L"]
            print "std-total", np.std(IU_train, axis=0)**2
            print "var (should match L from above)", np.std(IU_check, axis=0)**2
            print "avg", np.average(IU_check, axis=0)
            raw_input('...')

    # CONCATENATE PROJECTED DESCRIPTORS
    IU_train = np.concatenate(IUs_train, axis=1)
    IU_test = np.concatenate(IUs_test, axis=1)
    IV = np.concatenate(IVs, axis=1)
    if verbose:
        print "K-IV", IV.T.dot(IV)
        print IU_train.shape
        print IU_test.shape
        raw_input('___')
    state["IX_train"] = IU_train
    state["IX_test"] = IU_test
    return state

def cache_descriptor(state, options, log):
    log << log.mg << "Caching descriptor" << log.endl
    log << "Descriptor dimension: %d" % state["IX_train"].shape[1] << log.endl
    state["IX_train_cached"] = np.copy(state["IX_train"])
    state["IX_test_cached"] = np.copy(state["IX_test"])
    return state

def uncache_descriptor(state, options, log):
    log << log.mg << "Uncaching descriptor" << log.endl
    log << "Descriptor dimension before uncaching: %d" % state["IX_train"].shape[1] << log.endl
    if "data_logger" in state:
        log << "Logging index assignment" << log.endl
        i0 = 0
        i1 = state["IX_train"].shape[1]
        i2 = i1 + state["IX_train_cached"].shape[1]
        state["data_logger"].append({
            "tag": "uncache",
            "data": { "cached": [i1, i2], "base": [i0, i1] }
        })
    state["IX_train"] = np.concatenate(
        [state["IX_train"], state["IX_train_cached"]], axis=1)
    state["IX_test"] = np.concatenate(
        [state["IX_test"], state["IX_test_cached"]], axis=1)
    log << "Descriptor dimension after uncaching: %d" % state["IX_train"].shape[1] << log.endl
    return state

# =====
# OTHER
# =====

def upconvert_descriptor_matrix(IX, concatenate):
    D = IX.shape[1]
    D2 = D*(D+1)/2
    idcs_upper = np.triu_indices(D)
    IX_up = np.zeros((IX.shape[0], D+D2), dtype=IX.dtype)
    IX_up[:,0:D] = IX
    for i in range(IX.shape[0]):
        IX_up[i,D:] = np.outer(IX[i], IX[i])[idcs_upper]
    return IX_up

