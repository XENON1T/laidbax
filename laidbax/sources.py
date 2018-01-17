import numpy as np

from scipy.interpolate import interp1d
from blueice.source import MonteCarloSource, HistogramPdfSource
from multihist import Hist1d

from .signals import simulate_signals, sim_events_dtype


class XENONSource(MonteCarloSource):
    """A Source in a XENON-style experiment

    For ER, use yields specified by a polynomial fit
    For NR, use yields specified by the parametrization of the Lenardo et al 2015 global fit
    """
    energy_distribution = None  # Histdd of rate /kg /keV /day.
    s1_bias = None      # Interpolator for S1 bias as a function of detected photons
    s2_bias = None      # Interpolator for S2 bias as a function of detected photons

    def __init__(self, config, *args, **kwargs):
        # Defaults for config settings
        config.setdefault('spatial_distribution', 'uniform')
        config['cache_attributes'] = config.get('cache_attributes', []) + ['energy_distribution', 'recoil_type']
        super().__init__(config, *args, **kwargs)

    def compute_pdf(self):
        self.set_e_spectrum_and_recoil_type()
        h = self.energy_distribution

        # Compute the integrated event rate (in events / day)
        # This includes all events that produce a recoil; many will probably be out of range of the analysis space.
        self.events_per_day = np.sum(h.histogram * h.bin_volumes()) * self.config['fiducial_mass']

        # Nonlinearity on S1 and S2, derived from simulated data
        if 's1_bias_file' in self.config:
            self.s1_bias = interp1d(*self.config['s1_bias_file'], bounds_error='extrapolate', kind='nearest')
        if 's2_bias_file' in self.config:
            self.s2_bias = interp1d(*self.config['s2_bias_file'], bounds_error='extrapolate', kind='nearest')

        super().compute_pdf()

    def set_e_spectrum_and_recoil_type(self):
        """Sets self.energy_distribution to a Hist1d containing rates per (kg kev day), self.recoil_type
        """
        ed_format = self.config.get('energy_distribution_format', 'old')
        if ed_format == 'old':
            self.energy_distribution = self._e_spectrum_from_rates(*self.config['energy_distribution'])
        elif ed_format == 'hist1d':
            self.energy_distribution = self.config['energy_distribution']
        else:
            raise NotImplementedError("Dude, what's %s for an energy spectrum format?" % ed_format)
        self.recoil_type = self.config['recoil_type']

    @staticmethod
    def _e_spectrum_from_rates(es, rates):
        """Turn an energy spectrum specified by differential rates at es into a Hist1d histogram, so we can sample it.
        We average the rates in between the points provided
        """
        rates = np.array(rates)
        h = Hist1d(bins=es)
        h.histogram = 0.5 * (rates[1:] + rates[:-1])
        return h

    def simulate(self, n_events):
        """Simulate n_events from this source."""
        n_events = int(n_events)
        c = self.config
        if n_events==0:
            return simulate_signals(c,[],[],None)

        # Sample energies.
        # get_random assumes an 'events per bin' type histogram, not a PDF, so we must first convert it
        # (for uniform bins there is no distinction)
        events_per_energy_bin = self.energy_distribution * self.energy_distribution.bin_volumes()
        energies = events_per_energy_bin.get_random(n_events)

        photons_produced, electrons_produced = self.yields(energies)

        d = simulate_signals(c, photons_produced, electrons_produced, energies,
                             s1_bias=self.s1_bias, s2_bias=self.s2_bias)

        return d

    def yields(self, energies):
        """Sample arrays of (photons produced, electrons produced) for events at energies
        :param energies: array of energies of events to sample
        """
        if not isinstance(energies, np.ndarray):
            energies = np.ones(1) * energies
        n_events = len(energies)

        # Get the mean number of "base quanta" produced
        c = self.config
        n_quanta = c['base_quanta_yield'] * energies
        nonzero = n_quanta > 0
        n_quanta[nonzero] = np.random.normal(n_quanta[nonzero],
                                             np.sqrt(c['base_quanta_fano_factor'] * n_quanta[nonzero]))

        # 0 or negative numbers of quanta give trouble with the later formulas.
        # Store which events are bad, set them to 1 quanta for now, then zero the yields for these events later.
        bad_events = n_quanta < 1
        n_quanta = np.clip(n_quanta, 1, float('inf'))
        n_quanta = np.round(n_quanta).astype(np.int32)

        c = self.config
        if self.recoil_type == 'nr':
            # Account for quanta getting lost as heat
            p_detectable = self.p_detectable(energies)
            n_quanta = np.random.binomial(n_quanta, p_detectable)

        # Simple lin-log model of probability of becoming an electron
        p_becomes_electron = self.p_electron(energies)

        # Extra fluctuation (according to LUX, due to fluctuation in recombination probability)
        fluctuation = c['p_%s_electron_fluctuation' % self.recoil_type]
        if fluctuation != 0:
            p_becomes_electron = np.random.normal(p_becomes_electron, fluctuation)
        p_becomes_electron = np.clip(p_becomes_electron, 0, 1)

        # Sample the actual numbers binomially
        # The size argument is explicitly needed to always get an array back (even when simulating one event)
        electrons_produced = np.random.binomial(np.clip(n_quanta, 0, None),
                                                p=p_becomes_electron,
                                                size=len(energies))
        photons_produced = n_quanta - electrons_produced

        # "Remove" bad events (see above); actual removal happens at the end of simulate_signals
        photons_produced[bad_events] *= 0
        electrons_produced[bad_events] *= 0

        return photons_produced, electrons_produced

    def mean_signal(self, energy):
        """Utility function which returns the mean location in (cs1, cs2) at a given energy"""
        c = self.config
        rt = self.recoil_type
        nq_mean = c['base_quanta_yield'] * energy
        if rt == 'nr':
            nq_mean *= self.p_detectable(energy)
        ne_mean = nq_mean * self.p_electron(energy)
        nph_mean = nq_mean - ne_mean
        p_dpe = c['double_pe_emission_probability']
        cs2_mean = ne_mean * c['s2_gain'] * c.get('electron_extraction_efficiency', 1) * (1 + p_dpe)
        cs1_mean = nph_mean * c['ph_detection_efficiency'] * (1 + p_dpe)
        return cs1_mean, cs2_mean

    def p_electron(self, energy):
        """Probability a produced quantum becomes an electron at a given energy"""
        if self.recoil_type == 'nr':
            n_ph, n_el = self.nest_yields(energy)
            return n_el / (n_ph + n_el)
        return self.poly_function('qy', energy, minimum=0, maximum=self.config['base_quanta_yield']
                                  ) / self.config['base_quanta_yield']

    def p_detectable(self, energy):
        """Probability a produced quantum becomes a photon at a given energy"""
        assert self.recoil_type == 'nr'
        n_ph, n_el = self.nest_yields(energy)
        return (n_el + n_ph) / (energy * self.config['base_quanta_yield'])

    def poly_function(self, key, energy, minimum=0, maximum=1):
        """Evaluate polynomial function of energy from configuration variables
        :param key: Key in configuration. Will look for rt_key_xxx variables (except min and max energy)
        :param energy: Energies to evaluate function at
        :param minimum: Minimum output of function (clips result)
        :param maximum: Maximum output of function (clips result)
        :returns: array of same length as energy: result of polynomial function
        """
        c = self.config
        rt = self.recoil_type
        ref_e = c['%s_reference_energy' % rt]
        min_energy = c.get('%s_min_response_energy' % rt, 0)
        max_energy = c.get('%s_max_response_energy' % rt, float('inf'))
        energy = np.clip(energy, min_energy, max_energy)

        # Get polynomial order and coefficients
        order = c['%s_poly_order' % rt]
        coefs = np.array([c['%s_%s_%d' % (rt, key, i)] for i in range(order)])

        if '%s_%s_pca_components' % (rt, key) in c:
            # The coefficients specified are not ordinary polynomial coefficients, but normalized PCA factors.
            # We must recover the poly coeffs by un-applying PCA and the normalization before and after PCA
            # Note 'pre' scale was applied before PCA, 'post' after, so to recover original coefs,
            # we must un-apply post-scale first
            coefs = coefs * np.array(c['%s_%s_pca_post_scale' % (rt, key)]) + \
                    np.array(c['%s_%s_pca_post_mean' % (rt, key)])
            coefs = np.dot(coefs, np.array(c['%s_%s_pca_components' % (rt, key)]))
            coefs = coefs * np.array(c['%s_%s_pca_pre_scale' % (rt, key)]) +\
                    np.array(c['%s_%s_pca_pre_mean' % (rt, key)])

        result = 0
        for i in range(order):
            if self.config.get('function_of_log_energy', False):
                x = np.log10(energy / ref_e)
            else:
                x = (energy - ref_e)
            result += coefs[i] * x**i

        return np.clip(result, minimum, maximum)

    def nest_yields(self, energy):
        """NR yields according to the Lenardo et al 2015 global fit.
        Returns (array of produced photons, array of produced electrons)
        """
        c = self.config
        assert self.recoil_type == 'nr'

        Z = 54  # charge of Xenon
        W = 1/c['base_quanta_yield']   # 13.7e-3
        F = c['drift_field']
        E = energy

        # Lindhard quenching factor
        eps = 11.5 * E * Z**(-7/3)
        g = 3 * eps**0.15 + 0.7 * eps**0.6 + eps
        L = c['lindhard_k'] * g/(1 + c['lindhard_k'] * g)

        # Number of quanta
        nq = E * L / W

        # Number of excitons and ions among initial quanta
        nexni = c['nr_alpha'] * F **-c['nr_zeta'] * (1 - np.exp(-c['nr_beta'] * eps))
        ni = nq * 1/(1 + nexni)
        nex = nq - ni

        # Fraction of ions NOT participating in recombination
        squiggle = c['nr_gamma'] * F**-c['nr_delta']
        fnotr = np.log(1 + ni * squiggle)/(ni * squiggle)

        # Fraction of excitons NOT participating in Penning quenching
        fnotp = 1/(1 + c['nr_eta'] * eps**c['nr_lambda'])

        # Finally, number of electrons and photons produced..
        n_el = ni * fnotr
        n_ph = fnotp * (nex + ni * (1 - fnotr))
        return n_ph, n_el


class WIMPSource(XENONSource):
    """Source of signals of WIMP-nucleus recoils using the wimprates package from
    https://github.com/JelleAalbers/wimprates

    Be careful when specifying settings to ignore: depending on the detection mechanism, this could be ER or NR...
    """

    def set_e_spectrum_and_recoil_type(self):
        self.recoil_type = dict(elastic_nr='nr', bremsstrahlung='er')[self.config['wimp_detection_mechanism']]

        try:
            from wimprates import rate_wimp
        except ImportError:
            print("Please install the wimprates library from https://github.com/JelleAalbers/wimprates")
            raise
        import numericalunits as nu

        es = np.array(self.config['wimp_energies'])

        rates = rate_wimp(es * nu.keV,
                          mw=self.config['wimp_mass'] * nu.GeV/nu.c0**2,
                          sigma_nucleon=self.config['wimp_sigma_nucleon'] * nu.cm**2,
                          detection_mechanism=self.config['wimp_detection_mechanism'],
                          interaction=self.config['wimp_interaction'])
        rates /= (nu.kg**-1 * nu.day**-1 * nu.keV**-1)

        # Cutoff spectrum at low energies
        rates[es <= self.config['wimp_%s_response_cutoff' % self.recoil_type]] = 0

        self.energy_distribution = self._e_spectrum_from_rates(es, rates)


class PickledHistogramSource(HistogramPdfSource):
    """Load source from a file with pickled histograms
    Ad-hoc thing constructed to load XENON SR0 data-driven models in a ROOT-free way
    """

    def build_histogram(self):
        # with open(self.config['source_data'], mode='rb') as infile:
        #     stuff = pickle.load(infile)[self.config['key']]
        stuff = self.config['source_data'][self.config['key']]
        self._pdf_histogram = h = stuff['pdf_histogram']
        self._bin_volumes = h.bin_volumes()
        for dim_i, bin_edges in enumerate(h.bin_edges):
            aspace_bin_edges = np.array(self.config['analysis_space'][dim_i][1])
            if len(aspace_bin_edges) != len(bin_edges) or np.any(np.abs(aspace_bin_edges - bin_edges)/bin_edges > 1e-5):
                raise ValueError("Bin edges along dimension %d of histogram source %s do not match analysis space. \n"
                                 "Histogram has %d bin edges: %s\n"
                                 "Analysis space has %d bin edges: %s" % (dim_i, self.name,
                                                                     len(bin_edges), bin_edges,
                                                                     len(aspace_bin_edges), aspace_bin_edges))

        self.events_per_day = stuff['events_per_day']

    def simulate(self, n_events):
        d_simple = HistogramPdfSource.simulate(self, n_events)

        # Add the metadata fields, so we can concatenate these events with those from other sources
        d = np.zeros(len(d_simple), dtype=sim_events_dtype)
        for field_name in d_simple.dtype.fields.keys():
            d[field_name] = d_simple[field_name]

        # Put random z to ensure we can study effects of having wrong electron lifetime
        c = self.config
        d['z'] = np.random.uniform(c['fiducial_volume_zmin'], c['fiducial_volume_zmax'], size=n_events)

        return d
