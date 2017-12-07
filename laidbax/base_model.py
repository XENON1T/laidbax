"""
A XENON1T model

This is a bit off a mess because I don't yet know how to make a nice interface for specifying this.
Maybe INI files or something...
"""
import os
import inspect
import numpy as np
from pax import units

from pax.configuration import load_configuration
pax_config = load_configuration('XENON1T')

from .sources import PolynomialXENONSource, PickledHistogramSource, NRGlobalFitSource

# Store the directory of this file
THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Ignore these if you're an NR source:
nr_ignore_settings = ['er_reference_energy', 'er_max_response_energy', 'er_poly_order',
                      'p_er_electron_fluctuation', 'function_of_log_energy']
nr_ignore_settings += ['er_qy_%d' % i for i in range(10)]

# Ignore these if you're an ER source:
er_ignore_settings = 'lindhard_k drift_field p_nr_electron_fluctuation'.split() + [
    'nr_' + greek for greek in 'alpha beta gamma delta zeta eta lambda'.split()
]

config = dict(
    default_source_class=NRGlobalFitSource,
    data_dirs=[os.path.join(THIS_DIR, 'data'), '.'],
    analysis_space=(('cs1', tuple(np.linspace(3, 70, 68))),
                    ('cs2', tuple(np.logspace(*np.log10([50, 8000]), num=71)))),
    livetime_days=34.2,
    require_s1=True,
    require_s2=True,
    pdf_sampling_multiplier=1,
    pdf_sampling_batch_size=int(1e6),

    # Basic model info
    sources=[
        {'energy_distribution': 'er_bg.csv',
         'class': PolynomialXENONSource,
         'events_per_day': 620/34.2,
         'rate_multiplier': 1.157,  # Scale to SR0 expectation. Too lazy to modify the er_bg.csv.. will
         'recoil_type': 'er',
         'extra_dont_hash_settings': er_ignore_settings,
         'n_events_for_pdf': 1e7,
         'color': 'blue',
         'name': 'er',
         'label': 'ER Background'},

        {'energy_distribution': 'cnns.csv',
         'recoil_type': 'nr',
         'extra_dont_hash_settings': nr_ignore_settings,
         'n_events_for_pdf': 5e6,
         'color': 'orange',
         'name': 'cnns',
         'label': 'CNNS'},
        {'energy_distribution': 'radiogenic_neutrons.csv',
         'recoil_type': 'nr',
         'extra_dont_hash_settings': nr_ignore_settings,
         'n_events_for_pdf': 55e6,
         'color': 'purple',
         'name': 'radiogenics',
         'label': 'Radiogenic neutrons'},

        {'class': PickledHistogramSource,
         'source_data': 'sr0_nonparam_models.pkl',
         'key': 'ac',
         'color': 'green',
         'name': 'ac',
         'label': 'Accidental coincidences'},
        {'class': PickledHistogramSource,
         'source_data': 'sr0_nonparam_models.pkl',
         'key': 'wall',
         'color': 'brown',
         'name': 'wall',
         'label': 'Wall leakage'},
        {'class': PickledHistogramSource,
         'source_data': 'sr0_nonparam_models.pkl',
         'key': 'anomalous',
         'color': 'pink',
         'name': 'anomalous',
         'label': 'Anomalous flat component'},

        {'energy_distribution': 'wimp_50gev_1e-45cm2.csv',
         'recoil_type': 'nr',
         'extra_dont_hash_settings': nr_ignore_settings,
         'n_events_for_pdf': 5e6,
         'color': 'red',
         'name': 'wimp',
         'label': '50 GeV WIMP'}
    ],

    # Area thresholds on uncorrected S1/S2
    s1_area_threshold=0,    # Efficiency operates on coincidence, not area
    s2_area_threshold=150,

    # Bias (primarily due to self-trigger)
    s1_bias='x1t_s1_bias_Feb26.csv',
    s2_bias='x1t_s2_bias_Feb26.csv',

    # Detector parameters
    fiducial_mass=1042,  # kg. np.pi * rmax**2 * (zmax - zmin) * density?
    e_lifetime=452 * units.us,
    v_drift=1.44 * units.m/units.ms,

    # Probability for a photon detected by a PMT to produce two photoelectrons.
    # From https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:double_pe_emission
    double_pe_emission_probability=0.15,

    # g2
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:subgroup:energyscale:g1g2firstresult_summarynote
    # 1.15 to correct for dpe emission
    electron_extraction_efficiency=1,
    s2_gain=11.52/1.15,

    # g1
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:subgroup:energyscale:g1g2firstresult_summarynote
    # 1.15 to correct for dpe emmission
    ph_detection_efficiency= 0.1442 / 1.15,

    # Width (in photoelectrons) of the single-photoelectron area spectrum. Median(gain_sigma)/Median(gains) of TPC PMTs
    pmt_gain_width=0.4181,

    # For sampling of light and charge yield in space
    n_location_samples=int(1e5),  # Number of samples to take for the source positions (for light yield etc, temporary?)
    fiducial_volume_radius= 39.85,
    # Note z is negative, so the maximum z is actually the z of the top boundary of the fiducial volume
    fiducial_volume_zmax= -83.45,
    fiducial_volume_zmin= -13.45,
    s1_relative_ly_map='s1_lce_rz_precomputed_kr83m_sep29_doublez.pkl',     # TODO: Update for new LCE map

    # S1/S2 generation parameters
    base_quanta_yield=73,  # NEST's basic quanta yield, xenon:xenon1t:sim:notes:marco:conversion-ed-to-s1-s2

    # Fano factor for smearing of the base quanta yield
    # xenon:xenon1t:sim:notes:marco:conversion-ed-to-s1-s2 and xenon:xenon1t:sim:notes:marco:t2-script-description,
    # ultimately taken from the NEST code
    base_quanta_fano_factor=0.03,

    # S1 peak detection efficiency (index = number of detected photons, starting from 0)
    # See=xenon:xenon1t:analysis:firstresults:daqtriggerpaxefficiency
    s1_peak_detection_efficiency = ( 0.    ,  0.    ,  0.    ,  0.5135,  0.787 ,  0.851 ,  0.9165,
                                     0.9255,  0.9415,  0.948 ,  0.9545,  0.961 ,  0.972 ,  0.987 ,
                                     0.9865,  0.9845,  0.9865,  0.993 ,  0.992 ,  0.9905,  1),

    er_reference_energy=5,          # keV
    er_max_response_energy=12,
    er_poly_order=3,                # CAUTION: order of n means n terms (so polynomial order n-1...)
    er_qy_0=33.4,
    er_qy_1=-28.6,
    er_qy_2=12,
    p_er_electron_fluctuation=0.045,
    function_of_log_energy=True,

    # NR Global fit paramaters (Lenardo et al. 2015)
    nr_alpha=1.24,
    nr_beta=239,
    nr_gamma=0.01385,
    nr_delta=0.0620,
    nr_zeta=0.0472,
    nr_eta=3.3,
    nr_lambda=1.14,
    lindhard_k=0.1394,

    drift_field=117,                # V/cm
    p_nr_electron_fluctuation=0.02,   # I dunno...

)

# Nonphysical sources with a fixed (cs1, cs2b) histogram ignore all settings except livetime_days
# (not sure exactly how we handled that one, don't want to mess with it)
nonparametric_ignore_settings = [c
                                 for c in config.keys()
                                 if c not in 'sources livetime_days source_data key'.split()]
for s in config['sources']:
    if s.get('class') == PickledHistogramSource:
        s['extra_dont_hash_settings'] = nonparametric_ignore_settings