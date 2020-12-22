=================
Code for "Hot extremes have become drier in the US Southwest"
=================

This repository contains research-level code to reproduce the analysis and figures for "Hot extremes have become drier in the US Southwest" by Karen A. McKinnon, Andrew Poppick, and Isla R. Simpson. The manuscript is currently in review at Nature Climate Change, and the preprint is posted on `Research Square <https://www.researchsquare.com/article/rs-102766/v1>`_.

Please the requirements.txt file for needed packages. In particular, you will need my packages `humidity_variability <https://github.com/karenamckinnon/humidity_variability>`_ and `helpful_utilities <https://github.com/karenamckinnon/helpful_utilities>`_. 

There are two primary components of the analysis: (1) estimating the non-crossing quantile smoothing splines for each dataset, and (2) calculating and interpreting the amplification index. Part (1) can take a while, because the model is being fit many times (5 quantiles + a large number of stations or gridboxes).

**(1) ISD**

Download the ISD data from NOAA using::

    python ./scripts/download_isd.py $tmpdir $savedir
    
Then fit the model using::

    python ./scripts/est_q_quantiles.py $yr_start $yr_end $month_start $month_end $id_start $n_id $datadir $gjson_fname $predictor

Run::

    python ./scripts/est_q_quantiles.py -h 
    
for more information on the args. To reproduce the results in the published paper,::
    
    yr_start=1973
    yr_end=2019
    month_start=7
    month_end=9
    gjson_fname=./shapefiles/interior_west.json
    predictor=GMT
    
**(1) ERA5 and JRA-55**

ERA5 hourly data for pressure, 2m temperature, and 2m dewpoint can be downloaded at the `Copernicus Climate Data Store <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_. An example script to download 2m temperature from ERA5 is at::

    ./scripts/download_ERA5_T.py

Note that you will need to install the Climate Data Store API, and set up authentication.

In order to run the quantile smoothing splines model, dewpoint and pressure are first used to calculate specific humidity, and then each gridbox must be formatted like a "station". An example of how to do so is at::

    ./scripts/preprocess_ERA.py 
    
but note that this code will not run "out of the box".

JRA-55 six-hourly 2m air temperature and specific humidity are available via the NCAR Research Data Archive `here <https://rda.ucar.edu/datasets/ds628.0/>`_. We rely on the analysis, not forecast, version, which incorporates screen-level measurements. Similar to ERA5, the JRA-55 data is formatted to look like a station before fitting the model. An exampleof how to do so is at::

    ./scripts/preprocess_JRA55.py 
    
but note that this code will not run "out of the box".

Finally, the quantile smoothing splines model can be run using::

    python ./est_q_quantiles_reanalysis.py $start_year $end_year $start_month $end_month $id_start $n_id $datadir $reanalysis_name

The two :code:`est_q*.py` scripts are very similar, but the ISD-based on includes some additional QC not relevant for reanalysis output.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
