import numpy as np

from blueice.source import MonteCarloSource
from blueice.utils import InterpolateAndExtrapolate1D
from multihist import Hist1d

from .signals import simulate_signals


class XENONSource(MonteCarloSource):
    """A Source in a XENON-style experiment"""
    energy_distribution = None  # Histdd of rate /kg /keV /day.

    def __init__(self, config, *args, **kwargs):
        # Defaults for config settings
        config.setdefault('spatial_distribution', 'uniform')
        config['cache_attributes'] = config.get('cache_attributes', []) + ['energy_distribution']
        super().__init__(config, *args, **kwargs)

    def compute_pdf(self):
        # Turn the energy spectrum from two arrays into a histogram, so we can sample it.
        # We average the rates in between the points provided
        es, rates = self.config['energy_distribution']
        h = self.energy_distribution = Hist1d(bins=es)
        self.energy_distribution.histogram = 0.5 * (rates[1:] + rates[:-1])

        # Compute the integrated event rate (in events / day)
        # This includes all events that produce a recoil; many will probably be out of range of the analysis space.
        self.events_per_day = h.histogram.sum() * self.config['fiducial_mass'] * (h.bin_edges[1] - h.bin_edges[0])

        super().compute_pdf()

    def simulate(self, n_events):
        """Simulate n_events from this source."""
        c = self.config

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

        d = simulate_signals(c, photons_produced, electrons_produced, energies)

        return d

    def quanta_to_photons_electrons(self, energies, n_quanta):
        raise NotImplementedError


def _f(e, a, b, reference_energy, min_y=0):
    return np.clip(a * np.log10(e / reference_energy) + b, min_y, 1)


class SimplifiedXENONSource(XENONSource):

    def quanta_to_photons_electrons(self, energies, n_quanta):
        c = self.config
        rt = c['recoil_type']
        if rt == 'nr':
            # Account for quanta getting lost as heat
            p_detectable = _f(energies, c['nr_p_detectable_a'], c['nr_p_detectable_b'], c['reference_energy'])
            n_quanta = np.random.binomial(n_quanta, p_detectable)

        # Simple lin-log model of probability of becoming an electron
        p_becomes_electron = _f(energies,
                                c[rt + '_p_electron_a'],
                                c[rt + '_p_electron_b'],
                                c['reference_energy'],
                                c.get(rt + '_p_electron_min', 0))

        # Extra fluctuation (according to LUX due to fluctuation in recombination probability)
        fluctuation = c['p_%s_electron_fluctuation' % rt]
        if fluctuation != 0:
            p_becomes_electron = np.random.normal(p_becomes_electron, fluctuation)
        p_becomes_electron = np.clip(p_becomes_electron, 0, 1)

        # Sample the actual numbers binomially
        electrons_produced = np.random.binomial(n_quanta, p=p_becomes_electron)
        return n_quanta - electrons_produced, electrons_produced

    def mean_signal(self, energy):
        """Utility function which returns the mean location in (cs1, cs2) at a given energy"""
        # TODO: remove code duplication!
        c = self.config
        rt = c['recoil_type']
        nq_mean = c['base_quanta_yield'] * energy
        if rt == 'nr':
            nq_mean *= _f(energy, c['nr_p_detectable_a'], c['nr_p_detectable_b'], c['reference_energy'])
        ne_mean = nq_mean * _f(energy,
                               c[rt + '_p_electron_a'],
                               c[rt + '_p_electron_b'],
                               c['reference_energy'],
                               c.get(rt + '_p_electron_min', 0))
        nph_mean = nq_mean - ne_mean
        cs2_mean = ne_mean * c['s2_gain'] * c.get('electron_extraction_efficiency', 1)
        cs1_mean = nph_mean * c['ph_detection_efficiency']
        return cs1_mean, cs2_mean



class RegularXENONSource(XENONSource):
    """Old-style MC which constructs photon and electron yields from LEff, QY, etc"""

    def quanta_to_photons_electrons(self, energies, n_quanta):
        c = self.config
        n_events = len(energies)

        p_becomes_photon = energies * self.yield_at(energies, c['recoil_type'], 'photon') / n_quanta
        p_becomes_electron = energies * self.yield_at(energies, c['recoil_type'], 'electron') / n_quanta

        if c['recoil_type'] == 'er':
            # Apply extra recombination fluctuation (NEST tritium paper / Atilla Dobii's thesis)
            p_becomes_electron = np.random.normal(p_becomes_electron,
                                                  p_becomes_electron * c['recombination_fluctuation'],
                                                  size=n_events)
            p_becomes_electron = np.clip(p_becomes_electron, 0, 1)
            p_becomes_photon = 1 - p_becomes_electron
            n_quanta = np.round(n_quanta).astype(np.int)

        elif c['recoil_type'] == 'nr':
            # For NR some quanta get lost in heat.
            # Remove them and rescale the p's so we can use the same code as for ERs after this.
            p_becomes_detectable = p_becomes_photon + p_becomes_electron
            if p_becomes_detectable.max() > 1:
                raise ValueError("p_detected max is %s??!" % p_becomes_detectable.max())
            p_becomes_photon /= p_becomes_detectable
            n_quanta = np.round(n_quanta).astype(np.int)
            n_quanta = np.random.binomial(n_quanta, p_becomes_detectable)

        else:
            raise ValueError('Bad recoil type %s' % c['recoil_type'])

        photons_produced = np.random.binomial(n_quanta, p_becomes_photon)
        electrons_produced = n_quanta - photons_produced

        return photons_produced, electrons_produced

    def yield_at(self, energies, recoil_type, quantum_type):
        """Return the yield in quanta/kev for the given energies (numpy array, in keV),
        recoil type (string, 'er' or 'nr') and quantum type (string, 'photon' or 'electron')"""
        c = self.config

        # The yield functions are all interpolated in log10(energy) space,
        # Since that's where they are usually plotted in... and curve traced from.
        # The yield points are clipped to 0: a few negative values may have slipped in while curve tracing.
        if not hasattr(self, 'yield_functions'):
            self.yield_functions = {k: InterpolateAndExtrapolate1D(np.log10(self.config[k][0]),
                                                                   np.clip(self.config[k][1], 0, float('inf')))
                                    for k in ('leff', 'qy', 'er_photon_yield')}

        log10e = np.log10(energies)
        if quantum_type not in ('electron', 'photon'):
            raise ValueError("Invalid quantum type %s" % quantum_type)

        if recoil_type == 'er':
            """
            In NEST the electronic recoil yield is calculated separately for each event,
            based on details of the GEANT4 track structure (in particular the linear energy transfer).
            Here I use an approximation, which is the "old approach" from the MC group, see
                xenon:xenon1t:sim:notes:marco:t2-script-description#generation_of_light_and_charge
            A fixed number of quanta (base_quanta_yield) is assumed to be generated. We get the photon yield in quanta,
            then assume the rest turns into electrons.
            """
            if quantum_type == 'photon':
                return self.yield_functions['er_photon_yield'](log10e)
            else:
                return c['base_quanta_yield'] - self.yield_functions['er_photon_yield'](log10e)

        elif recoil_type == 'nr':
            """
            The NR electron yield is called Qy.
            It is here assumed to be field-independent (but NEST 2013 fig 2 shows this is wrong...).

            The NR photon yield is described by several empirical factors:
                reference_gamma_photon_yield * efield_light_quenching_nr * leff

            The first is just a reference scale, the second contains the electric field dependence,
            the third (leff) the energy dependence.
            In the future we may want to simplify this to just a single function
            """
            if quantum_type == 'photon':
                return self.yield_functions['leff'](log10e) * \
                       c['reference_gamma_photon_yield'] * c['nr_photon_yield_field_quenching']
            else:
                return self.yield_functions['qy'](log10e)

        else:
            raise RuntimeError("invalid recoil type %s" % recoil_type)
