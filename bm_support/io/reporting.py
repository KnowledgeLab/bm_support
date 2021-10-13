def get_freq(report):
    return report["freq"][0]


def get_point_est_state(report):
    root = report["posterior_info"]["point"]
    return {k.split("_")[-1]: root[k][0] for k in root.keys() if k.startswith("pi")}


def get_point_est_beta(report):
    root = report["posterior_info"]["point"]
    return {k: root[k][0] for k in root.keys() if not k.startswith("pi")}


def get_false_point_est_state(report):
    root = report["posterior_info"]["point"]
    return {k: root[k] for k in root.keys() if k.startswith("pi") and not root[k][1][0]}
