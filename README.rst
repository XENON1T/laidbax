Likelihood- And Interpolated Density Based Analysis for Xenon
=============================================================
Jelle Aalbers, 2016

Statistical XENON1T Analysis for the lazy analyst.

Source code: `https://github.com/XENON1T/laidbax`

Documentation:

- `This note <https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:statspackage_architecture>`_ on the XENON wiki.
- `These example notebooks <https://github.com/XENON1T/laidbax/tree/master/notebooks>`_.
- You can of course also look at the docstrings and code comments.


About
=====

Did you spend years of your life in a dark cave below the mountain building that awesome TPC? Do you want to spend the next few years behind a dimly lit computer screen making sense of the data that comes out? Then this package is not for you.

This package allows you to do parametric inference using Monte-Carlo derived extended unbinned likelihood functions. It lets you make likelihood functions which measure agreement between data and Monte Carlos with different settings: you choose which settings to vary (which parameters the likelihood functions has) and in which space the agreement is measured. For more information, please see `the documentation in this note
<https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:statspackage_architecture>`_ and the `examples in the notebooks folder <https://github.com/XENON1T/wimpy/tree/master/notebooks>`_.

All of the hard work is done by `blueice <https://github.com/JelleAalbers/blueice>`_; this only contains the code necessary to make blueice work with a simple Monte Carlo of the XENON1T experiment. Much of this is derived or shamelessly "borrowed" from other sources:

- Andrew's maximum-gap limit setting code, used for the XENON100 max-gap cross checks.
- Chris' wimpstat repository, used for the XENON100 S2-only limit setting (but no longer available?)
- NEST: not directly, but since this is currently the best xenon TPC code out there, the physics model used here 
- Several of the Monte Carlo group's excellent material on this topic, in particular the `XENON1T Monte Carlo paper <http://arxiv.org/abs/1512.07501>`_ and the `notes linked here <https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:org:papers:xe1t_sensitivity>`_. 

The default model includes lots of information I obtained by curve-tracing plots or even just constructing some function that looked like what I saw in an image. Just saying...


Installation
============
First install the major dependencies: `blueice` and `pax`. We need pax for the common unit system and XENON1T.ini.

Then run `python setup.py develop` or `python setup.py install`
