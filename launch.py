#! /usr/bin/env python
import librmt as rmt
import numpy as np
import os
import pickle
import json
import sys

# TODO opt_hyper api

def evaluate(
    options_models, # Options relating to execution of models
    options_meta,   # Options relating to evaluation of all models
    options_state,  # Options relating to construction of state
    options_build,  # Options relating to construction of models
    load_state,     # Function f(arg: options_state) called to create state
    build_models,   # Function f(arg: options_build) called to create models
    log,            # Logger
    check_options = (lambda opt: True),     # Function f(arg: options_models) used to check option validity
    update_options = (lambda opt, i: opt)): # function f(arg: options_models, arg: options_meta, i_rep) called to update options for repetition i_rep

    """
    # Example for options_meta
    options_meta = {
        "n_reps": 10,
        "seed_base": 791623,
        "opt_hyper": False,
        "filelog": False,
        "filelog_name": "run-models-log.json",
        "fileout_name": "run-models-out.json",
        "models": [
            "upca-rr",
            "tree2-mpf-leg-rr", 
            "treehc-mpf-leg-rr", 
            "linear-rr", 
            "kernel-rr", 
        ],
        "sweeps": [
            ["split_test_train/f_train", [ 0.2 ]],
            ["split_test_train/f_lowest_split", [ 0.2 ]]
            #["split_test_train/f_train", [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]],
            #["split_test_train/f_lowest_split", [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]]
        ],
        "store": ["n_train", "n_test", "rmse_train", "rmse_test", "std_data_train", "std_data_test"]
    }
    """

    # BUILD MODELS
    model_dict = build_models(options_build)

    # BUILD OPTIONS SWEEPS
    options_set = rmt.utils.dict_compile(options_models, options_meta["sweeps"], "combinatorial")

    # LOAD RECORDS
    if options_meta["filelog"] and options_meta["filelog_name"] in os.listdir('./'):
        records = json.load(open(options_meta["filelog_name"]))
        tags = {}
        for r in records:
            t = r["tag"]
            if t in tags:
                tags[t] += 1
            else: 
                tags[t] = 1
    else:
        records = []
        tags = {}

    for options in options_set:
        # VALIDATE OPTIONS
        valid = check_options(options)
        fields = []
        opt_special = {}
        for s in options_meta["sweeps"]:
            fields.append("{}={}".format(s[0], rmt.utils.dict_lookup_path(options, s[0])))
            opt_special[s[0]] = rmt.utils.dict_lookup_path(options, s[0])
        opt_tag = "__".join(fields)
        if not valid:
            log << "Skipping invalid options set: %s" % opt_tag << log.endl
            continue
        # RUN MODELS
        for model in options_meta["models"]:
            record_tag = "model=%s__%s" % (model, opt_tag)
            if record_tag in tags:
                i0_rep = tags[record_tag]
                if i0_rep == options_meta["n_reps"]: 
                    log << "Skip existing: %s" % record_tag << log.endl
                    continue
                else:
                    log << "Completing existing (%d/%d): %s" % (options_meta["n_reps"]-i0_rep, options_meta["n_reps"], record_tag) << log.endl
            else:
                i0_rep = 0
            log << log.endl
            log << log.my << record_tag << log.endl
            log << log.endl
            # CROSS-VALIDATE
            for i in range(i0_rep, options_meta["n_reps"]):
                # UPDATE OPTIONS FOR THIS REPETITION
                options = update_options(options, options_meta, i)
                # Optimise hyperparameters
                if options_meta["opt_hyper"] and i == 0 and model == "hpcamp-rr":
                    log << "Start subspace dimension optimisation" << log.endl
                    options["split_test_train"]["seed"] = seed_base - 1
                    rmses_hyper = []
                    take_hyper = [ 2, 3, 5, 10, 15, 20, 25, 30 ]
                    for take in take_hyper:
                        options["hpcamp_transform"]["target_dim"] = take
                        options["split_test_train"]["seed"] = seed_base + i
                        log.silent = True
                        state_clone = load_state(options_state)
                        out = model_dict[model].apply(state_clone, options, log)
                        log.silent = False
                        rmses_hyper.append(out["out"]["rmse_test"])
                        log << "Take %d : RMSE_test %1.4f" % (take, out["out"]["rmse_test"]) << log.endl
                    idx_take = np.argmin(rmses_hyper)
                    log << "RMSE min" << rmses_hyper[idx_take] << "@" << take_hyper[idx_take] << log.endl
                    options["hpcamp_transform"]["target_dim"] = take_hyper[idx_take]
                    model_hyper = "subspace_size=%d" % take_hyper[idx_take]
                else:
                    model_hyper = ""
                # Load state
                state_clone = load_state(options_state)
                # Apply model
                out = model_dict[model].apply(state_clone, options, log)
                t_test_pred = out["out"]["T_test_pred"]
                t_test = out["out"]["T_test"]
                np.savetxt('out-test-%s.txt' % model, np.array([t_test_pred, t_test]).T)
                record = {
                    "tag": record_tag,
                    "model": model,
                    "model_hyper": model_hyper
                }
                for field in options_meta["store"]: 
                    record[field] = out["out"][field]
                records.append(record)
                if options_meta["filelog"] and len(records) % 10 == 0:
                    json.dump(records, open(options_meta["filelog_name"], "w"), indent=1)

    if options_meta["filelog"]:
        json.dump(records, open(options_meta["filelog_name"], "w"), indent=1)
        json.dump(records, open(options_meta["fileout_name"], "w"), indent=1)

    return

def parse_records(
        records,
        keys_avg, 
        keys_export, 
        map_group = lambda r: r["model"],
        map_average = lambda r: r["tag"],
        transform=lambda r: r, 
        prefix='eval-', 
        postfix='.out'):
    """
    # Example records list
    records = [
         {
          "tag": "model=upca-rr__split_test_train/f_train=0.2__split_test_train/f_lowest_split=0.2", 
          "model": "upca-rr", 
          "rmse_test": 0.20441501531775563, 
         }, 
         {
          "tag": "model=tree2-mpf-leg-rr__split_test_train/f_train=0.2__split_test_train/f_lowest_split=0.2", 
          "model": "tree2-mpf-leg-rr", 
          "rmse_test": 0.1838654286458345, 
         }, 
    ]
    """

    def perform_average(group, keys):
        avgs = { k: 0.0 for k in keys }
        for r in group:
            for k in keys:
                avgs[k] += r[k]
        for k in keys:
            avgs[k] /= len(group)
        record_out = group[0]
        for k in keys:
            record_out[k] = avgs[k]
        return record_out

    def write(tag, group, keys):
        ofs = open('%s%s%s' % (prefix, tag, postfix), 'w')
        ofs.write("# %s\n" % " ".join(keys))
        for g in group:
            for k in keys:
                ofs.write("{} ".format(g[k]))
            ofs.write("\n")
        ofs.close()

    # TRANSFORM
    print "Transform"
    for i in range(len(records)):
        records[i] = transform(records[i])

    # GROUP
    print "Group"
    groups = {}
    for r in records:
        g = map_group(r)
        if g in groups:
            groups[g].append(r)
        else:
            groups[g] = [ r ]
    print "Groups:", groups.keys()

    # AVERAGE/REDUCE
    groups_out = {}
    for g in groups:
        groups_out[g] = []
        group = groups[g]
        group_avgs = {}
        for r in group:
            a = map_average(r)
            if a in group_avgs:
                group_avgs[a].append(r)
            else:
                group_avgs[a] = [ r ]
        
        for k in group_avgs:
            gavg = perform_average(group_avgs[k], keys_avg)
            groups_out[g].append(gavg)
        groups_out[g] = sorted(groups_out[g], key=lambda g: g["tag"])

    # WRITE
    for g in groups_out:
        write(g, groups_out[g], keys_export+keys_avg)
   
    return groups_out

