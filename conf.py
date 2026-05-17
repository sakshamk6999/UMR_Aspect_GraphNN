label_to_node = {
    "process": 0,
    "performance": 1,
    "endeavor": 2,
    "habitual": 3,
    "state": 4,
    "activity": 5,
    "none": 6,
    "aspect": 7,
    "imperfective": 8,
    "perfective": 9,
    "atelic": 10 
}

node_to_label = {
    0: "process",
    1: "performance",
    2: "endeavor",
    3: "habitual",
    4: "state",
    5: "activity",
    6: "none",
    7: "aspect",
    8: "imperfective",
    9: "perfective",
    10: "atelic"
}

parent_to_child = {
    "none": None,
    "aspect": ["process", "habitual", "imperfective"],
    "process": ["perfective", "atelic"],
    "imperfective": ["atelic", "state"],
    "perfective": ["performance", "endeavor"],
    "atelic": ["endeavor", "activity"],
    "state": None,
    "performance": None,
    "endeavor": None,
    "habitual": None,
    "activity": None
}

child_to_parent = {
    "performance": ["perfective"],
    "endeavor": ["perfective", "atelic"],
    "activity": ["atelic"],
    "perfective": ["process"],
    "atelic": ["process", "imperfective"],
    "state": ["imperfective"],
    "imperfective": ["aspect"],
    "process": ["aspect"],
    "habitual": ["aspect"],
    "aspect": None,
    "none": None
}