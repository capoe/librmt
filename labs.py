#! /usr/bin/env python
import librmt as rmt
import numpy as np
import csv
import json
import math
import scipy.cluster.hierarchy as sch

def hierarchical_clustering(
        distance_matrix,
        min_size=2,
        log=None, 
        verbose=False):
    # Compute linkage matrix L:
    # L[i] = [ c1, c2, d12, n12 ] 
    # Read this as: cluster i merged from clusters c1 and c2, 
    # with cluster distance d12, and number of member nodes n12
    if log: log << "Hierarchical clustering ..." << log.flush
    L = sch.linkage(distance_matrix, method='centroid')
    if log: log << "done." << log.endl
    # Unwrap clusters from linkage matrix
    if log: log << "Unwrapping clusters ..." << log.flush
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
    if log: log << "done." << log.endl
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

def normalise_ix(state, options, log):
    norm = np.std(state["IX_train"], axis=0, keepdims=True)
    mean = np.sum(state["IX_train"], axis=0, keepdims=True)/state["IX_train"].shape[0]
    state["IX_train"] = (state["IX_train"]-mean)/norm
    state["IX_test"] = (state["IX_test"]-mean)/norm
    #norm = np.std(state["IX_train"], axis=0, keepdims=True)
    #mean = np.sum(state["IX_train"], axis=0, keepdims=True)/state["IX_train"].shape[0]
    #print norm
    #print mean
    return state

def threshold_ix(state, options, log):
    raise NotImplementedError()
    stds = np.std(state["IX_train"], axis=0)
    return state

def mp_transform(IX, norm_avg, norm_std, log):
    # Computes MP thresholded PC space based on cov(IX)
    mp_gamma = float(IX.shape[1])/IX.shape[0]
    IX_norm_pca, IZ, X_mean, X_std, S, L, V = rmt.pca_compute(
        IX=IX,
        log=None,
        norm_div_std=norm_std,
        norm_sub_mean=norm_avg,
        eps=0.,
        ddof=0)
    idcs_eigen_signal = np.where(L.diagonal() > rmt.dist_mp_bounds(mp_gamma)[1])[0]
    n_dim_mp_signal = len(idcs_eigen_signal)
    if log: log << "[mptf] Dim reduction %d -> %d with gamma_mp = %+1.2f" % (
        IX.shape[1], n_dim_mp_signal, mp_gamma) << log.endl
    L_signal = L.diagonal()[idcs_eigen_signal]
    V_signal = V[:,idcs_eigen_signal]
    return L_signal, V_signal, X_mean, X_std

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

def binomial_coeff(n, k):
    b = 1.
    for i in range(1, k+1):
        b *= (n+1-i)/i
    return b

def hpcamp_transform(state, options, log):
    # PARAMETERS
    # - target subspace dimension ("take")
    # - minimum cluster size ("min_cluster_size")
    # TODO
    # - Improve FPS routine, taking into account 
    # ... angle between PCAs
    # ... subspace overlap (angle between subspace directors)
    # ... eigenvalue magnitude (difference?)
    # OPTIONS
    file_debug = True
    verbose = False
    options = options["hpcamp_transform"]
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
    IX = state["IX_train"]
    N = IX.shape[0]
    D = IX.shape[1]
    covmat = np.cov(IX, rowvar=False, ddof=0)
    dmat = (1. - covmat**2)
    clusters_all = hierarchical_clustering(
        distance_matrix=dmat, min_size=1, log=log, verbose=False)
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
        log << "Cluster %4d [%4d]    gamma*=%1.3f fss*=%1.2f d*=%4d    %4d filters" % (cidx, cluster["idx"], gamma, fss, n_fss, len(filter_idcs)) << log.endl
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
            log << "Add sequence: length %d" % seq << log.endl
            if verbose: print seq
            sequences.append(seq)
        if file_debug:
            ofs = open('sequences.txt', 'w')
            sizes = [ len(f) for f in filter_idcs ]
            for seq in sequences:
                for fidx in seq:
                    ofs.write('%d %+1.7e\n' % (sizes[fidx], filter_lambdas[fidx]))
                ofs.write('\n')
            ofs.close()
        log << "Have %d filter sequences" % (len(sequences)) << log.endl
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
    if do_fps:
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

    # =======
    # Project
    # =======

    state["IX_train"] = state["IX_train"].dot(V_tf)
    state["IX_test"] = state["IX_test"].dot(V_tf)
    # Normalise
    norm = np.std(state["IX_train"], axis=0, keepdims=True)
    mean = np.sum(state["IX_train"], axis=0, keepdims=True)/state["IX_train"].shape[0]
    state["IX_train"] = (state["IX_train"]-mean)/norm
    state["IX_test"] = (state["IX_test"]-mean)/norm
    log.prefix = log.prefix[0:-7]
    return state























