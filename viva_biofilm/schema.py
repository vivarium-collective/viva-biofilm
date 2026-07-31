from viva_biofilm import biofilm_core

def load_world(spec: dict):
    w = biofilm_core.World()
    d = spec["domain"]
    w.set_domain(d["nx"], d["ny"], d["dx"], d["layer_thickness"])
    index = {}
    for s in spec["solutes"]:
        index[s["name"]] = w.add_solute(
            s["name"], s["init"], s["diff_liquid"], s["diff_biofilm"], s["bulk"]
        )
    for r in spec["reactions"]:
        monod = [(index[n], ks) for n, ks in r["monod"]]
        yields = [(index[n], c) for n, c in r["yields"]]
        w.add_reaction(r["mu_max"], monod, yields)
    sp = spec["species"]
    w.set_species(sp["density"], sp["division_mass"])
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
