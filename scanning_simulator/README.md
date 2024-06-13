# Creating ShapeNet and Kortx Datasets
This directory contains code for simulating laser-scanning on ShapeNet meshes. By calling this code, one can generate
the preprocessed curve data needed for the ShapeNet and Kortx evaluations.

## ShapeNet

### Downloads
* Download and unzip the ShapeNet Segmentation Benchmark dataset [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).
* Download the ShapeNetCore V1 dataset. Follow instructions from [here](https://shapenet.org/download/shapenetcore). Please be sure to
 to download version 1! Version 2 has signficant changes that will NOT work!

### Preprocessing ShapeNet for Curve Segmentation
In the main code directory, please run the following to simulate laser scanning on ShapeNet meshes:
```
PYTHONPATH=. python scanning_simulator/shapenet_seg/generate_shapenet.py --seg-data-path <path-to-shapenet-partseg> --mesh-data-path <path-to-ShapeNetCore.v1> --outdir <output-location>
```

For example, one possible run could be:
```angular2html
PYTHONPATH=. python scanning_simulator/shapenet_seg/generate_shapenet.py --seg-data-path ./shapenetcore_partanno_segmentation_benchmark_v0_normal --mesh-data-path ./ShapeNetCore.v1 --outdir ./processed-shapenet
```

Additional useful arguments input include:
* `--category`: Lets you specify a single ShapeNet object category to process
* `--npoints`: How many points to sample. Defaults to 4096
* `--split`: Whether to iterate through the official "train", "validation", or "test" split on Shapenet
* `--raster-res`: Image resolution to "draw" curves on. Defaults to 2048, and very likely does not need to be changed
* `--scanline-density`: Pixel density of "drawn" curves. Defaults to 0.25, and very likely does not need to be changed
* `--scan-direction`: Choose to sample scan lines in a "parallel", "grid", or "random" manner

Furthermore, you can visualize the preprocessing pipeline by specifying the following arguments:
* `--viz`: flag indicating to visualize each generated mesh
* `--mesh_viz_outdir`: if specified, will also save mesh renderings into this directory
* `--viz_mitsuba`: flag indicating to use slower (but better-looking) mitsuba rendering instead of plotly

### Output Data Directory
After running the above script, the specified output directory will contain formatted data for training and evaluating CurveCloudNet. 

## Kortx

### Downloads
* Download and unzip the ShapeNet Segmentation Benchmark dataset [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).
* Download the ShapeNetCore V1 dataset. Follow instructions from [here](https://shapenet.org/download/shapenetcore). Please be sure to
* Download and unzip the raw Kortx scanned data [here](https://drive.google.com/file/d/1z5tiQYELfevh9J26N6huGRDPgES3cf2m/view?usp=drive_link).

### Preprocessing ShapeNet Training Set
We use simulated laser scanning on ShapeNet meshes as the training set for the Kortx data. Notably, because the orientations of
 scanned Kortx objects varies, the ShapeNet meshes are randomly rotated during simulated scanning.

To generate the ShapeNet training and validation splits, run the following command:
```angular2html
PYTHONPATH=. python scanning_simulator/kortx/generate_shapenet.py --seg-data-path <path-to-shapenet-partseg> --mesh-data-path <path-to-ShapeNetCore.v1> --outdir <output-location>
```
Please refer to the "Preprocessing ShapeNet for Curve Segmentation" section for optional additional arguments. Furthermore,
because we randomly rotate each mesh, we provide the additional argument:

* `--num-dataset-repeats`: Number of times to sample from each mesh (with different random orientation) in the training set


### Preprocessing the Kortx Test Set
We provide the raw (dense and aggregate) Kortx captures [here](https://drive.google.com/file/d/1z5tiQYELfevh9J26N6huGRDPgES3cf2m/view?usp=drive_link). 
To preprocess into our evaluation data format, run the following:
```angular2html
PYTHONPATH=. python scanning_simulator/kortx/generate_kortx.py --data-path <path-to-raw-kortx> --outdir <output-location>
```

Additional useful arguments include:
* `--npoints`: number of points to include in each ``frame"
* `--samples_per_scan`: How many test samples to generate from each dense scan
* `--viz`: Whether to visualize the processed data

### Output Data Directory
After running the above scripts, the specified output directory will contain training, validation, and testing data for 
the Kortx experiment.