# Data Directory (`data/`)

This directory stores sample leaf specimens and local dataset partitions.

## Dataset Reference
- **Mendeley Dataset Archive:** [https://data.mendeley.com/datasets/x995snz7p3/1](https://data.mendeley.com/datasets/x995snz7p3/1)
- **Associated Research Paper (PMC):** [https://pmc.ncbi.nlm.nih.gov/articles/PMC12175232/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175232/)

## Dataset Counts
The verified dataset distribution is:
- **Total Images:** 2,661
- **Healthy:** 934 images (35.1%)
- **Insect Hole:** 639 images (24.0%)
- **Black Rot:** 1,088 images (40.9%)
- **Total Diseased (Insect Hole + Black Rot):** 1,727 images (64.9%)

## Directory Structure
```text
data/
├── README.md
├── samples/
│   ├── healthy_sample.jpg
│   ├── insect_hole_sample.jpg
│   └── black_rot_sample.jpg
└── [Optional full dataset category subfolders]
    ├── Healthy/
    ├── Insect Hole/
    └── Black Rot/
```

The sample gallery on the Home page automatically scans `data/samples/` and any category subdirectories.
