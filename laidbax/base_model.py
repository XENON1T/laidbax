"""
A XENON1T model

This is a bit off a mess because I don't yet know how to make a nice interface for specifying this.
Maybe INI files or something...
"""
from copy import deepcopy
import os
import inspect
import numpy as np

from pax import units
from pax.configuration import load_configuration
pax_config = load_configuration('XENON1T')

from .XENONSource import RegularXENONSource, SimplifiedXENONSource, PolynomialXENONSource

# Store the directory of this file
THIS_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Ignore these if you're an NR source:
nr_ignore_settings = ['er_photon_yield', 'recombination_fluctuation',
                      'er_poly_order',
                      'er_p_electron_a', 'er_p_electron_b', 'p_er_electron_fluctuation']
nr_ignore_settings += ['er_p_electron_%d' % i for i in range(10)]

# Ignore these if you're an ER source:
er_ignore_settings = ['leff', 'qy', 'nr_photon_yield_field_quenching',
                      'nr_poly_order',
                      'p_nr_electron_fluctuation',
                      'nr_p_electron_a', 'nr_p_electron_b',
                      'nr_p_detectable_a', 'nr_p_detectable_b']
er_ignore_settings += ['nr_p_electron_%d' % i for i in range(10)]
er_ignore_settings += ['nr_p_detectable_%d' % i for i in range(10)]


config = dict(
    default_source_class=PolynomialXENONSource,
    data_dirs=[os.path.join(THIS_DIR, 'data'), '.'],
    analysis_space=(('cs1', tuple(np.linspace(0, 70, 70))),
                    ('cs2', tuple(np.linspace(0, 7000, 70)))),
    livetime_days=2 * 365.25,
    require_s1=True,
    require_s2=True,
    pdf_sampling_multiplier=1,
    pdf_sampling_batch_size=int(1e6),
    # Basic model info
    sources=[
        {'energy_distribution': 'er_bg.csv',
         'color': 'blue',
         'recoil_type': 'er',
         'name': 'er_bg',
         'n_events_for_pdf': 2e7,
         'extra_dont_hash_settings': er_ignore_settings,
         'label': 'ER Background'},
        {'energy_distribution': 'cnns.csv',
         'color': 'orange',
         'recoil_type': 'nr',
         'name': 'cnns',
         'n_events_for_pdf': 5e6,
         'extra_dont_hash_settings': nr_ignore_settings,
         'label': 'CNNS'},
        {'energy_distribution': 'radiogenic_neutrons.csv',
         'color': 'purple',
         'recoil_type': 'nr',
         'name': 'radiogenics',
         'n_events_for_pdf': 5e6,
         'extra_dont_hash_settings': nr_ignore_settings,
         'label': 'Radiogenic neutrons'},
        {'energy_distribution': 'wimp_50gev_1e-45cm2.csv',
         'color': 'red',
         'name': 'wimp_50gev',
         'n_events_for_pdf': 5e6,
         'analysis_target': True,
         'recoil_type': 'nr',
         'extra_dont_hash_settings': nr_ignore_settings,
         'label': '50 GeV WIMP'}
    ],

    # Areatthresholds on uncorrected S1/S2
    s1_area_threshold=0,    # Efficiency operates on coincidence, not area
    s2_area_threshold=150,

    # Bias (primarily due to self-trigger)
    s1_bias = 'x1t_s1_bias_Feb26.csv',
    s2_bias = 'x1t_s2_bias_Feb26.csv',

    # Detector parameters
    fiducial_mass=1000,  # kg. np.pi * rmax**2 * (zmax - zmin) * density?
    e_lifetime=pax_config['DEFAULT']['electron_lifetime_liquid'],
    v_drift=pax_config['DEFAULT']['drift_velocity_liquid'],

    # 31.44 from xenon:xenon1t:analysis:subgroup:energyscale:g1g2firstresult_summarynote_2#g2_total
    # 1.15 for double-pe emission
    #  (10.70/11.46) for non-linearity (the note above computes only the bottom S2 gain).
    s2_gain= 31.44 / 1.15 * (10.70/11.46),

    ph_detection_efficiency= 0.152 / 1.15,       # 0.152 from note referenced above (with nonlinearity correction), 1.15 for double pe emission.

    pmt_gain_width=0.4181,   # Width (in photoelectrons) of the single-photoelectron area spectrum. Median(gain_sigma)/Median(gains) of TPC PMTs

    double_pe_emission_probability=0.15,  # Probability for a photon detected by a PMT to produce two photoelectrons. From https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:aalbers:double_pe_emission

    electron_extraction_efficiency = 0.907,     # From https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:subgroup:energyscale:g1g2firstresult_summarynote_2#non-linearity_correction

    # For sampling of light and charge yield in space
    n_location_samples=int(1e5),  # Number of samples to take for the source positions (for light yield etc, temporary?)
    fiducial_volume_radius=pax_config['DEFAULT']['tpc_radius'] * 0.9,
    # Note z is negative, so the maximum z is actually the z of the top boundary of the fiducial volume
    fiducial_volume_zmax=- 0.05 * pax_config['DEFAULT']['tpc_length'],
    fiducial_volume_zmin=- 0.95 * pax_config['DEFAULT']['tpc_length'],
    s1_relative_ly_map='s1_lce_rz_precomputed_kr83m_sep29_doublez.pkl',

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
                                     0.9865,  0.9845,  0.9865,  0.993 ,  0.992 ,  0.9905, 1),

    # Fit to XENON1T Rn220 data - February 28
    er_reference_energy=10,          # keV
    er_poly_order=3,                 # CAUTION: order of n means n terms (so polynomial order n-1...)
    er_p_electron_0=0.3839,          # Value at reference energy
    er_p_electron_1=-0.3,            # Slope " " "
    er_p_electron_2=0.0897,          # Second derivative " " "
    er_max_response_energy=11.62766,
    p_er_electron_fluctuation=0.04,

    # Fit to XENON1T AmBe data - March 1, 14:24
    nr_reference_energy=100,        # keV
    nr_poly_order=3,                # CAUTION Order of n means n + 1 terms
    nr_p_electron_0=0.1609,         # Value at reference energy
    nr_p_electron_1=-0.2625,        # Slope " " "
    nr_p_electron_2=-0.0156,        # Second derivative " ' "

    # These are not fitted, just something that matches the LUX tritium lines
    nr_p_detectable_0=0.21,
    nr_p_detectable_1=0.02,
    nr_p_detectable_2=0,

    p_nr_electron_fluctuation=0.02,
    function_of_log_energy = True,

)
