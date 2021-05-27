CONFIG = {
    "dbas": {
        "url": "https://localhost/"
    },
    "hyperparameter_combinations": {
        "deliberate": {
            "top_n": [5, 10, 20, 30, 40, 50, 100, 500],
            "alpha": [.1, .2, .3, .4, .5, .6, .7, .8, .9],
            "max_depth": [1, 2],
            "weighting_strategy": ["raw", "normalized", None]
        },
        # "deliberate": {"top_n": [20], "alpha": [.5], "max_depth": [2], "weighting_strategy": ["raw"]},
        # "deliberate": {"top_n": [100], "alpha": [.5], "max_depth": [1], "weighting_strategy": ["raw"]},
        # "deliberate": {"top_n": [10], "alpha": [.5], "max_depth": [2], "weighting_strategy": ["raw"]},
    }
}
