atom_info = {
    "O": {
        'color': (13, 13, 255),
        'radius': 0.74
    },
    
    "H": {
        'color': (255, 255, 255),
        'radius': 0.46
    }
}

bound_dict = {
    ("O", "O"): {"color": (80, 160, 122), "lower": 2.6, "upper": 3.4, "thickness": 3},  # 3.2
    ("O", "H"): {"color": (255, 160, 122), "lower": 1.5, "upper": 2.4, "thickness": 1},
    ("H", "O"): {"color": (255, 160, 122), "lower": 1.5, "upper": 2.4, "thickness": 1},
    ("H", "H"): {"color": None, "lower": 1.8, "upper": 3, "thickness": 0},
}

findPeakResolution = (128,128)