import numpy as np


def simulate_signals(config, n_photons, n_electrons, energies=None):
    """Simulate results due to n_photons and n_electrons produced. Returns numpy structured array.
    energies is not used, but will be included in the results for your convenience if you pass it.

    config must be a config dictionary containing at least:
     - s1_relative_ly_map
     - ph_detection_efficiency
     - double_pe_emission_probability
     - e_lifetime
     - v_drift
     - s2_gain
     - fiducial_volume_radius, fiducial_volume_zmin, fiducial_volume_zmax
    and optionally
     - require_s1, require_s2 (booleans, default true)
     - s1_area_threshold, s2_area_threshold (minimum areas in pe for S1 and S2)
     - spatial_distribution (currently only 'uniform' supported)
    """
    c = config
    n_events = len(n_photons)

    # Store everything in a structured array:
    d = np.zeros(n_events, dtype=[
        ('source', np.int),
        # Set here:
        ('energy', np.float),
        ('r2', np.float),
        ('theta', np.float),
        ('z', np.float),
        ('p_photon_detected', np.float),
        ('p_electron_detected', np.float),
        ('electrons_produced', np.int),
        ('photons_produced', np.int),
        ('electrons_detected', np.int),
        ('s1_photons_detected', np.int),
        ('s1_photoelectrons_produced', np.int),
        ('s1', np.float),
        ('s2', np.float),
        ('cs1', np.float),
        ('cs2', np.float),
    ])
    #if n = 0, do not procede:

    if energies is not None:
        d['energy'] = energies

    # If we get asked to simulate 0 events, return the empty array immediately
    if not len(d):
        return d

    # Fill in the inputs from the higher-level code
    d['photons_produced'] = n_photons
    d['electrons_produced'] = n_electrons

    # Sample the positions and relative light yields
    if c.get('spatial_distribution', 'uniform') == 'uniform':
        d['r2'] = np.random.uniform(0, c['fiducial_volume_radius'] ** 2, n_events)
        d['theta'] = np.random.uniform(0, 2 * np.pi, n_events)
        d['z'] = np.random.uniform(c['fiducial_volume_zmin'], c['fiducial_volume_zmax'], size=n_events)
        rel_lys = c['s1_relative_ly_map'].lookup(d['r2'], d['z'])
    else:
        raise NotImplementedError("Only uniform sources supported for now...")

    # Get the light & charge collection efficiency
    d['p_photon_detected'] = c['ph_detection_efficiency'] * rel_lys
    d['p_electron_detected'] = c.get('electron_extraction_efficiency', 1) * \
                               np.exp(d['z'] / c['v_drift'] / c['e_lifetime'])  # No minus: z is negative

    # Detection efficiency
    d['s1_photons_detected'] = np.random.binomial(d['photons_produced'], d['p_photon_detected'])
    d['electrons_detected'] = np.random.binomial(d['electrons_produced'], d['p_electron_detected'])

    # S2 amplification
    d['s2'] = np.random.poisson(d['electrons_detected'] * c['s2_gain'])

    # S1 response
    d['s1_photoelectrons_produced'] = d['s1_photons_detected'] + np.random.binomial(d['s1_photons_detected'],
                                                                                    c['double_pe_emission_probability'])
    d['s1'] = np.random.normal(d['s1_photoelectrons_produced'],
                               np.clip(c['pmt_gain_width'] * np.sqrt(d['s1_photoelectrons_produced']),
                                       1e-9,   # Normal freaks out if sigma is 0...
                                       float('inf')))

    # Get the corrected S1 and S2, assuming our posrec + correction map is perfect
    # Note this does NOT assume the analyst knows the absolute photon detection efficiency:
    # photon detection efficiency / p_photon_detected is just the relative light yield at the position.
    # p_electron_detected is known exactly (since it only depends on the electron lifetime)
    s1_correction = c['ph_detection_efficiency'] / d['p_photon_detected']
    d['cs1'] = d['s1'] * s1_correction
    # S2 correction doesn't include extraction efficiency
    d['cs2'] = d['s2'] / (d['p_electron_detected'] / c.get('electron_extraction_efficiency', 1))

    # Remove events without an S1 or S1
    if c.get('require_s1', True):
        # One photons detected doesn't count as an S1 (since it isn't distinguishable from a dark count)
        d = d[d['s1_photons_detected'] >= 2]
        d = d[d['s1'] > c.get('s1_area_threshold', 0)]

    if c.get('require_s2', True):
        d = d[d['electrons_detected'] >= 1]
        d = d[d['s2'] > c.get('s2_area_threshold', 0)]

    return d
