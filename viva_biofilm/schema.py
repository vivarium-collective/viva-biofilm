from viva_biofilm import biofilm_core

def load_world(spec: dict):
    w = biofilm_core.World()
    d = spec["domain"]
    w.set_domain(d["nx"], d["ny"], d["dx"], d["layer_thickness"])
    w.set_detachment_rate(spec.get("detachment_rate", 0.0))
    index = {}
    for s in spec["solutes"]:
        index[s["name"]] = w.add_solute(
            s["name"], s["init"], s["diff_liquid"], s["diff_biofilm"], s["bulk"]
        )

    strategies = spec.get("strategies")
    if strategies:
        # Multi-species/competition path: each strategy gets its own
        # species slot (add_species), its own kinetics (add_reaction_for),
        # and its own distributed seeding (spawn_distributed). The pyo3
        # wrapper routes the first add_species() call onto CoreWorld's
        # pre-populated species[0] placeholder, so strategies[0] lands at
        # species index 0, strategies[1] at index 1, etc. (see
        # crates/biofilm-py/src/lib.rs and tests/competition.rs).
        strategy_indices = []
        for strat in strategies:
            idx = w.add_species(strat["density"], strat["division_mass"])
            monod = [(index[n], ks) for n, ks in strat["monod"]]
            yields = [(index[n], c) for n, c in strat["yields"]]
            w.add_reaction_for(idx, strat["mu_max"], monod, yields)
            strategy_indices.append(idx)

        pde = spec.get("pde")
        if pde:
            w.set_pde_params(
                pde.get("tol", 1e-4), pde.get("max_iter", 2000), pde.get("omega", 1.8)
            )

        for i, (idx, strat) in enumerate(zip(strategy_indices, strategies)):
            w.spawn_distributed(
                idx,
                strat["spawn_n"],
                strat.get("band_height", 1.0),
                strat.get("seed_offset", i),
            )
    else:
        for r in spec["reactions"]:
            monod = [(index[n], ks) for n, ks in r["monod"]]
            yields = [(index[n], c) for n, c in r["yields"]]
            w.add_reaction(r["mu_max"], monod, yields)
        sp = spec["species"]
        w.set_species(sp["density"], sp["division_mass"])
        pde = spec.get("pde")
        if pde:
            w.set_pde_params(
                pde.get("tol", 1e-4), pde.get("max_iter", 2000), pde.get("omega", 1.8)
            )
        sw = spec["spawn"]
        w.spawn_agents(sw["n"], sw["band_height"], sw.get("seed_offset", 0))

    w.finalize(int(spec.get("seed", 0)))
    return w

def load_chemostat(spec: dict):
    index = {s["name"]: i for i, s in enumerate(spec["solutes"])}
    c = biofilm_core.ChemostatWorld([s["init"] for s in spec["solutes"]])
    for r in spec["reactions"]:
        sub = r["substrate"] if isinstance(r["substrate"], int) else index[r["substrate"]]
        stoich = [(index[n] if isinstance(n, str) else n, coeff) for n, coeff in r["stoich"]]
        c.add_linear_reaction(sub, r["k"], stoich)
    return c
