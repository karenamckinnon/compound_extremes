#!/bin/bash

id_start=0
n_id=2
boot_start=0
nboot=2
datadir=/home/mckinnon/bucket/gsod/9748592161109009547
gjson_fname=/home/mckinnon/projects/compound_extremes/compound_extremes/shapefiles/interior_west.json

python est_dewp_quantiles.py $id_start $n_id $boot_start $nboot $datadir $gjson_fname
