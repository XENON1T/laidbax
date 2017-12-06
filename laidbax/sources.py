import numpy as np

from scipy.interpolate import interp1d
from blueice.source import MonteCarloSource, HistogramPdfSource
from multihist import Hist1d

from .signals import simulate_signals, sim_events_dtype


class XENONSource(MonteCarloSource):
    """A Source in a XENON-style experiment"""
    energy_distribution = None  # Histdd of rate /kg /keV /day.
    s1_bias = None      # Interpolator for S1 bias as a function of detected photons
    s2_bias = None      # Interpolator for S2 bias as a function of detected photons

    def __init__(self, config, *args, **kwargs):
        # Defaults for config settings
        config.setdefault('spatial_distribution', 'uniform')
        config['cache_attributes'] = config.get('cache_attributes', []) + ['energy_distribution']
        super().__init__(config, *args, **kwargs)

    def compute_pdf(self):
        # Turn the energy spectrum from two arrays into a histogram, so we can sample it.
        # We average the rates in between the points provided
        ed_format = self.config.get('energy_distribution_format', 'old')
        if ed_format == 'old':
            es, rates = self.config['energy_distribution']
            rates = np.array(rates)
            h = self.energy_distribution = Hist1d(bins=es)
            self.energy_distribution.histogram = 0.5 * (rates[1:] + rates[:-1])
        elif ed_format == 'hist1d':
            self.energy_distribution = h = self.config['energy_distribution']
        else:
            raise NotImplementedError("Dude, what's %s for an energy spectrum format?" % ed_format)

        # Compute the integrated event rate (in events / day)
        # This includes all events that produce a recoil; many will probably be out of range of the analysis space.
        self.events_per_day = h.histogram.sum() * self.config['fiducial_mass'] * (h.bin_edges[1] - h.bin_edges[0])

        if 's1_bias_file' in self.config:
            self.s1_bias = interp1d(*self.config['s1_bias_file'], bounds_error='extrapolate', kind='nearest')
        if 's2_bias_file' in self.config:
            self.s2_bias = interp1d(*self.config['s2_bias_file'], bounds_error='extrapolate', kind='nearest')

        super().compute_pdf()

    def simulate(self, n_events):
        """Simulate n_events from this source."""
        n_events = int(n_events)
        c = self.config
        if n_events==0:
            return simulate_signals(c,[],[],None)

        energies = self.energy_distribution.get_random(n_events)

        # Get the mean number of "base quanta" produced
        n_quanta = c['base_quanta_yield'] * energies
        n_quanta = np.random.normal(n_quanta,
                                    np.sqrt(c['base_quanta_fano_factor'] * n_quanta),
                                    size=n_events)

        # 0 or negative numbers of quanta give trouble with the later formulas.
        # Store which events are bad, set them to 1 quanta for now, then zero these events later.
        bad_events = n_quanta < 1
        n_quanta = np.clip(n_quanta, 1, float('inf'))

        n_quanta = np.round(n_quanta).astype(np.int32)
        photons_produced, electrons_produced = self.quanta_to_photons_electrons(energies, n_quanta)

        # "Remove" bad events (see above); actual removal happens at the very end of the function
        photons_produced[bad_events] *= 0
        electrons_produced[bad_events] *= 0

        d = simulate_signals(c, photons_produced, electrons_produced, energies,
                             s1_bias=self.s1_bias, s2_bias=self.s2_bias)

        return d

    def mean_signal(self, energy):
        """Utility function which returns the mean location in (cs1, cs2) at a given energy"""
        c = self.config
        rt = c['recoil_type']
        nq_mean = c['base_quanta_yield'] * energy
        if rt == 'nr':
            nq_mean *= self.p_detectable(energy)
        ne_mean = nq_mean * self.p_electron(energy)
        nph_mean = nq_mean - ne_mean
        p_dpe = c['double_pe_emission_probability']
        cs2_mean = ne_mean * c['s2_gain'] * c.get('electron_extraction_efficiency', 1) * (1 + p_dpe)
        cs1_mean = nph_mean * c['ph_detection_efficiency'] * (1 + p_dpe)
        return cs1_mean, cs2_mean

    def quanta_to_photons_electrons(self, energies, n_quanta):
        if not isinstance(energies, np.ndarray):
            energies = np.ones(1) * energies
            n_quanta = np.ones(1) * n_quanta

        c = self.config
        rt = c['recoil_type']
        if rt == 'nr':
            # Account for quanta getting lost as heat
            p_detectable = self.p_detectable(energies)
            n_quanta = np.random.binomial(n_quanta, p_detectable)

        # Simple lin-log model of probability of becoming an electron
        p_becomes_electron = self.p_electron(energies)

        # Extra fluctuation (according to LUX due to fluctuation in recombination probability)
        fluctuation = c['p_%s_electron_fluctuation' % rt]
        if fluctuation != 0:
            p_becomes_electron = np.random.normal(p_becomes_electron, fluctuation)
        p_becomes_electron = np.clip(p_becomes_electron, 0, 1)

        # Sample the actual numbers binomially
        # The size argument is explicitly needed to always get an array back (even when simulating one event)
        electrons_produced = np.random.binomial(n_quanta, p=p_becomes_electron, size=len(energies))
        return n_quanta - electrons_produced, electrons_produced


class PolynomialXENONSource(XENONSource):

    def p_electron(self, energy):
        return self.poly_function('qy', energy, minimum=0, maximum=self.config['base_quanta_yield']
                                  ) / self.config['base_quanta_yield']

    def p_detectable(self, energy):
        assert self.config['recoil_type'] == 'nr'
        return self.poly_function('ty', energy, minimum=0, maximum=self.config['base_quanta_yield']
                                  ) / self.config['base_quanta_yield']

    def poly_function(self, key, energy, minimum=0, maximum=1):
        c = self.config
        rt = c['recoil_type']
        ref_e = c['%s_reference_energy' % rt]
        result = 0
        max_energy = c.get('%s_max_response_energy' % rt, float('inf'))
        min_energy = c.get('%s_min_response_energy' % rt, 0)
        energy = np.clip(energy, min_energy, max_energy)
        for i in range(c['%s_poly_order' % rt]):
            if self.config.get('function_of_log_energy', False):
                result += c['%s_%s_%d' % (rt, key, i)] * (np.log10(energy / ref_e))**i
            else:
                result += c['%s_%s_%d' % (rt, key, i)] * (energy - ref_e)**i
        return np.clip(result, minimum, maximum)


class NRGlobalFitSource(XENONSource):
    """ER and NR yields according to the NEST global fit

    Specify drift field in V/cm
    """

    def p_electron(self, energy):
        n_ph, n_el = self.nest_yields(energy)
        return n_el / (n_ph + n_el)

    def p_detectable(self, energy):
        n_ph, n_el = self.nest_yields(energy)
        return (n_el + n_ph) / (energy * self.config['base_quanta_yield'])

    def nest_yields(self, energy):
        c = self.config
        assert c['recoil_type'] == 'nr'

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


class PickledHistogramSource(HistogramPdfSource):

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

        # Add zeros for the metadata fields, so we can concatenate these events with those from other sources
        d = np.zeros(len(d_simple), dtype=sim_events_dtype)
        for field_name in d_simple.dtype.fields.keys():
            d[field_name] = d_simple[field_name]

        return d
