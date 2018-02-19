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
    silent = False,
    hyper_every = 0,
    return_state = False,
    check_options = (lambda opt: True),     # Function f(arg: options_models) used to check option validity
    update_options = (lambda opt, i: opt)): # function f(arg: options_models, arg: options_meta, i_rep) called to update options for repetition i_rep

    """
    # Example for options_meta
    options_meta = {
        "n_reps": 10,
        "seed_base": 791623,
        "opt_hyper": True,
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
            hyper_is_optimized = False
            model_hyper = ""
            # CROSS-VALIDATE
            for i in range(i0_rep, options_meta["n_reps"]):
                log << log.endl << "Repetition %d" % (i+1) << log.endl << log.endl
                # UPDATE OPTIONS FOR THIS REPETITION
                options = update_options(options, options_meta, i)
                # Optimise hyperparameters
                do_hyper = (options_meta["opt_hyper"] and model_dict[model].hyper != None and hyper_is_optimized == False)
                if hyper_every > 0 and (i % hyper_every) == 0 and options_meta["opt_hyper"]:
                    do_hyper = True
                if do_hyper:
                    hyper = model_dict[model].hyper
                    options_hyper_set = rmt.utils.dict_compile(
                        options, 
                        hyper["sweeps"], 
                        "combinatorial")
                    state_clone = load_state(options_state)
                    target_out = []
                    monitor_out = []
                    tags_out = []
                    results_out = []
                    if hyper["break_at"]:
                        if silent: log.silent = True
                        out = model_dict[model].apply(state_clone, options, log, stop_at=hyper["break_at"])
                        log.silent = False
                        for options_hyper in options_hyper_set:
                            out = model_dict[model].apply(state_clone, options_hyper, log, start_at=hyper["break_at"], prefix=False)
                            target = hyper["target"](out)
                            target_out.append(target[1])
                            monitor = hyper["monitor"](out)
                            monitor_out.append(monitor[1])
                            opt_hyper_tag = ""
                            for s in hyper["sweeps"]:
                                v = rmt.utils.dict_lookup_path(options_hyper, s[0])
                                v = v if type(v) not in [float, int] else "%+1.4e" % v
                                opt_hyper_tag += "{}={} ".format(s[0], v)
                            tags_out.append(opt_hyper_tag)
                            results_out.append(out)
                            log << "[hyper] {} {} {:+1.7e}   {} {:+1.7e}".format(opt_hyper_tag, monitor[0], monitor[1], target[0], target[1]) << log.endl
                    else:
                        raise NotImplementedError()
                    idx = np.argmin(target_out)
                    options = options_hyper_set[idx]
                    log << "[hyper] Optimum @ parameters: {} target: {}".format(tags_out[idx], target_out[idx]) << log.endl
                    model_hyper = tags_out[idx]
                    out = results_out[idx]
                    hyper_is_optimized = True
                else:
                    log << "[hyper] Using parameter set from hyper-optimization:" << model_hyper << log.endl
                    # Load state
                    state_clone = load_state(options_state)
                    # Apply model
                    if silent: log.silent = True
                    out = model_dict[model].apply(state_clone, options, log)
                    log.silent = False
                #t_test_pred = out["out"]["T_test_pred"]
                #t_test = out["out"]["T_test"]
                #np.savetxt('out-test-%s.txt' % model, np.array([t_test_pred, t_test]).T)
                record = {
                    "tag": record_tag,
                    "model": model,
                    "model_hyper": model_hyper
                }
                if return_state:
                    log << "Recording output state" << log.endl
                    record["state"] = state_clone
                for field in options_meta["store"]: 
                    record[field] = out["out"][field]
                    log << "%20s = %-20s" % (field, repr(record[field])) << log.endl
                records.append(record)
                if options_meta["filelog"] and len(records) % 10 == 0:
                    json.dump(records, open(options_meta["filelog_name"], "w"), indent=1)

    if options_meta["filelog"]:
        json.dump(records, open(options_meta["filelog_name"], "w"), indent=1)
        json.dump(records, open(options_meta["fileout_name"], "w"), indent=1)

    return records

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

