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
    N = distance_matrix.shape[0]
    ii = np.triu_indices(N, k=1) # upper triangle without diagonal
    distances_compressed = distance_matrix[ii]
    #L = sch.linkage(distance_matrix, method='centroid') # NOTE BUG: Distance matrix not interpreted as such!!
    L = sch.linkage(distances_compressed, method='centroid', metric="*should not be used*")
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
            if verbose: 
                log << "Add sequence: length %d" % len(seq) << log.endl
                print seq
            sequences.append(seq)
        sequences = sorted(sequences, key=lambda s: len(s))
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

def upconvert_descriptor_matrix(IX, concatenate):
    D = IX.shape[1]
    D2 = D*(D+1)/2
    idcs_upper = np.triu_indices(D)
    IX_up = np.zeros((IX.shape[0], D+D2), dtype=IX.dtype)
    IX_up[:,0:D] = IX
    for i in range(IX.shape[0]):
        IX_up[i,D:] = np.outer(IX[i], IX[i])[idcs_upper]
    return IX_up

def hpcamp_transform_concat(state, options, log):
    # Options
    upconvert = options["hpcamp_transform"]["upconvert"]
    upconvert_concatenate = True
    concatenate = options["hpcamp_transform"]["concatenate"]
    # Transform
    V_tf = hpcamp(state["IX_train"], options, log)
    # Project
    IU_train = state["IX_train"].dot(V_tf)
    IU_test = state["IX_test"].dot(V_tf)
    # Upconvert?
    if upconvert:
        log << "Upconverting transformed descriptor (concatenate: %s)..." % upconvert_concatenate << log.endl
        IU_train = upconvert_descriptor_matrix(IU_train, concatenate=upconvert_concatenate)
        IU_test = upconvert_descriptor_matrix(IU_test, concatenate=upconvert_concatenate)
    # Concatenate?
    if concatenate:
        log << "Concatenating descriptor ..." << log.endl
        IX_train = np.zeros((state["IX_train"].shape[0], state["IX_train"].shape[1]+IU_train.shape[1]), IU_train.dtype)
        IX_test = np.zeros((state["IX_test"].shape[0], state["IX_test"].shape[1]+IU_test.shape[1]), IU_test.dtype)
        IX_train[:,0:state["IX_train"].shape[1]] = state["IX_train"]
        IX_train[:,state["IX_train"].shape[1]:] = IU_train
        IX_test[:,0:state["IX_test"].shape[1]] = state["IX_test"]
        IX_test[:,state["IX_test"].shape[1]:] = IU_test
    else:
        IX_train = IU_train
        IX_test = IU_test
    log << "Transformed descriptor dimension:" << IX_train.shape[1] << log.endl
    # Store
    state["IX_train"] = IX_train
    state["IX_test"] = IX_test
    log.prefix = log.prefix[0:-7]
    return state


def polympf_transform(state, options, log):

    # TODO Concatenate with linear descriptor
    # TODO Set legendre cutoff and options
    # TODO Handle sparsification

    min_size_cluster_y = 10 # TODO Option
    min_std_cluster_y = 0.1 # relative to std-dev of all targets
    k_c_max_y = 0.9
    msd_threshold_k = 3
    deep_mp = False
    options_deep_mp = {
        # Clustering
        "min_cluster_size": 10,
        "cluster_similarity_threshold": 0.9,
        # Sequencing
        "sparsify": "sequence", # "sequence" or "fps"
        "filter_seq_sim_cutoff": 0.71, # 0.71, 0.9
        "sequence_select": [0], # [0] or [ 0, -1 ]
        # FPS
        "do_fps": False,
        "fps_root": "largest_pc", # "largest_cluster"
        "target_dim": 30,
        # Post-processing
        "upconvert": True,
        "concatenate": True
    }
    verbose = False
    np.set_printoptions(precision=2)

    IX = state["IX_train"]
    Y = state["T_train"]

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
        IX_ss = rmt.div0(IX_ss, filter_std)
        cluster["filter_mu"] = filter_mu
        cluster["filter_std"] = filter_std

        # TEST DISTANCE FROM ORIGIN
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

    # SHIFT, SCALE AND PROJECT
    IX_train_out = []
    IX_test_out = []
    for n in range(V_tf.shape[0]):
        ix_train = ((state["IX_train"]-mu_tf[n])/std_tf[n]).dot(V_tf[n])
        ix_test = ((state["IX_test"]-mu_tf[n])/std_tf[n]).dot(V_tf[n])
        IX_train_out.append(ix_train)
        IX_test_out.append(ix_test)
    state["IX_train"] = np.array(IX_train_out).T
    state["IX_test"] = np.array(IX_test_out).T

    # RENORMALISE
    state = rmt.normalise_descriptor_zscore(state, options, log)
    print np.std(state["IX_train"], axis=0)
    print np.average(state["IX_train"], axis=0)
    print np.std(state["IX_test"], axis=0)

    state = legendre_transform(state, options, log)

    raw_input('___')
    return state

def legendre_transform(state, options, log):
    # Apply Legendre mapping
    max_degree = 6 # TODO Option
    scale = 0.25 # ix -> ix*scale in units of sigma (= 1.0) # TODO Option
    d = state["IX_train"].shape[1]
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    IX_train_out = np.zeros(
        (state["IX_train"].shape[0], max_degree*d), 
        dtype=state["IX_train"].dtype)
    IX_test_out = np.zeros(
        (state["IX_test"].shape[0], max_degree*d), 
        dtype=state["IX_test"].dtype)
    # In principle monomials x, x**2, x**3, ... should perform equivalently
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
    for n in range(1, max_degree+1):
        ix_train_n = legendre[n](scale*IX_train)
        ix_test_n = legendre[n](scale*IX_test)
        IX_train_out[:,(n-1)*d:n*d] = ix_train_n
        IX_test_out[:,(n-1)*d:n*d] = ix_test_n
    log << "Input dimension before legendre transform: d =" << d << log.endl 
    log << "Output dimension after legendre transform: d'=" << IX_train_out.shape[1] << log.endl
    log << "Standard deviation along components" << np.std(IX_train_out, axis=0) << log.endl
    state["IX_train"] = IX_train_out
    state["IX_test"] = IX_test_out
    return state


def pca_legendre_transform(state, options, log):
    # Derive and project onto MP signal PCs
    L, V_tf, X_mean, X_std = mp_transform(state["IX_train"], False, False, log, hist=False)
    state["IX_train"] = state["IX_train"].dot(V_tf)
    state["IX_test"] = state["IX_test"].dot(V_tf)
    #print L
    #print np.average(state["IX_train"], axis=0)
    #print np.std(state["IX_train"], axis=0)**2
    #print "Max", np.max(state["IX_train"], axis=0)
    #print "Min", np.min(state["IX_train"], axis=0)
    # Renormalise descriptor as variance along components changed
    state = rmt.normalise_descriptor_zscore(state, options, log)
    #print np.average(state["IX_train"], axis=0)
    #print np.std(state["IX_train"], axis=0)**2
    #print "Max", np.max(state["IX_train"], axis=0)
    #print "Min", np.min(state["IX_train"], axis=0)
    # Apply Legendre mapping
    # TODO Find better way how to deal with large x (those not easily scaled back to interval [-1,1])
    max_degree = 6 # TODO Option
    scale = 0.25 # ix -> ix*scale in units of sigma (= 1.0) # TODO Option
    d = state["IX_train"].shape[1]
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    IX_train_out = np.zeros(
        (state["IX_train"].shape[0], max_degree*d), 
        dtype=state["IX_train"].dtype)
    IX_test_out = np.zeros(
        (state["IX_test"].shape[0], max_degree*d), 
        dtype=state["IX_test"].dtype)
    # In principle monomials x, x**2, x**3, ... should perform equivalently
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
    for n in range(1, max_degree+1):
        ix_train_n = legendre[n](scale*IX_train)
        ix_test_n = legendre[n](scale*IX_test)
        IX_train_out[:,(n-1)*d:n*d] = ix_train_n
        IX_test_out[:,(n-1)*d:n*d] = ix_test_n
    log << "Input dimension before legendre transform: d =" << d << log.endl 
    log << "Output dimension after legendre transform: d'=" << IX_train_out.shape[1] << log.endl
    log << "Standard deviation along components" << np.std(IX_train_out, axis=0) << log.endl
    state["IX_train"] = IX_train_out
    state["IX_test"] = IX_test_out
    return state

def hpcamp_moments_transform(state, options, log):
    #R = np.random.randint(0, 2, size=state["IX_train"].shape[0]*state["IX_train"].shape[1])
    #R = 1.*R.reshape(state["IX_train"].shape)
    #norm = np.std(R, axis=0, keepdims=True)
    #mean = np.sum(R, axis=0, keepdims=True)/R.shape[0]
    #R = (R-mean)/norm
    #print R
    #print np.std(R, axis=0)
    #print np.average(R, axis=0)
    #L, V_tf, X_mean, X_std = mp_transform(R, False, False, log, hist=True)

    #V_tf = hpcamp(state["IX_train"], options, log)
    #L, V_tf, X_mean, X_std = mp_transform(state["IX_train"], False, False, log, hist=True)
    #print "Mean, std", X_mean, X_std

    T_train = state["T_train"]
    T_test = state["T_test"]
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]

    np.set_printoptions(precision=2)

    def split_mpf_target(IX, Y, levels, idcs=None, filters=[], this_level=1):
        # Split threshold
        y_split = np.average(Y)
        y_min = np.min(Y)
        y_max = np.max(Y)
        if type(idcs) == type(None):
            idcs = np.arange(IX.shape[0])
        idcs_0 = np.where(Y <= y_split)
        idcs_1 = np.where(Y > y_split)
        IX_0 = IX[idcs_0]
        IX_1 = IX[idcs_1]
        Y_0 = Y[idcs_0]
        Y_1 = Y[idcs_1]
        # Lower partition
        L_0, V_0, X_mean_0, X_std_0 = mp_transform(IX_0, True, True, log)
        filters.append({ 
            "idcs": idcs[idcs_0], "L": L_0, "n_filters": V_0.shape[1], 
            "V": V_0, "mu": X_mean_0, "std": X_std_0, "y_min": y_min, "y_max": y_split, 
            "partition": 0, "level": this_level })
        # Upper partition
        L_1, V_1, X_mean_1, X_std_1 = mp_transform(IX_1, True, True, log)
        filters.append({ 
            "idcs": idcs[idcs_1], "L": L_1, "n_filters": V_1.shape[1], 
            "V": V_1, "mu": X_mean_1, "std": X_std_1, 
            "y_min": y_split, "y_max": y_max, 
            "partition": 1, "level": this_level })
        # Descend into next level(s)
        if this_level < levels:
            filters = split_mpf_target(IX_0, Y_0, levels=levels, idcs=idcs[idcs_0], filters=filters, this_level=this_level+1)
            filters = split_mpf_target(IX_1, Y_1, levels=levels, idcs=idcs[idcs_1], filters=filters, this_level=this_level+1)
        return filters

    filters = split_mpf_target(IX_train, T_train, levels=2)
    for filter_ in filters:
        print "level {level:d} #filters {n_filters:d} y in [{y_min:+1.4e} ,{y_max:+1.4e}]".format(**filter_)
        if filter_["n_filters"] == 0: continue
        IU = ((IX_train-filter_["mu"])/filter_["std"]).dot(filter_["V"])
        print "Moments 1", IU.T.dot(T_train)/T_train.shape[0]/np.std(T_train)
        print "Moments 2", (IU**2).T.dot(T_train)/T_train.shape[0]/np.std(T_train)
        print "L", filter_["L"]
        print "std-total", np.std(IU, axis=0)**2
        IU = ((IX_train[filter_["idcs"]]-filter_["mu"])/filter_["std"]).dot(filter_["V"])
        print "std", np.std(IU, axis=0)**2
        print "avg", np.average(IU, axis=0)
        raw_input('...')

    assert False

    IU_low = IX_low.dot(V_low)
    IU_high = IX_high.dot(V_tf)
    IU_train = state["IX_train"].dot(V_tf)
    IU_test = state["IX_test"].dot(V_tf)

    IU2_train = IU_train**2
    IU2_test = IU_test**2

    ofs = open('moments.txt', 'w')
    for i in range(IU_train.shape[0]):
        ofs.write('%s %+1.7e ' % (
            "+" if T_train[i] > 0.0 else "-" , T_train[i]))
        for j in range(IU_train.shape[1]):
            ofs.write('%+1.7e ' % IU_train[i][j])
        ofs.write('\n')
    ofs.close()

    print L
    print np.std(IU_train, axis=0)**2
    print np.std(IU_low, axis=0)**2
    raw_input('...')

    print np.average(T_train)
    print np.std(T_train)
    print np.average(T_test)
    print np.std(T_test)
    print state["t_average"]

    t_moment_1 = IU_train.T.dot(T_train)/T_train.shape[0]
    t_moment_2 = IU2_train.T.dot(T_train)/T_train.shape[0]

    print "t_moments_1", t_moment_1
    print "t_moments_2", t_moment_2

    raw_input('MOMO')





















