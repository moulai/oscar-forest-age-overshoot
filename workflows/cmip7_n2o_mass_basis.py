"""Normalize the identified OSCAR v4-beta2.1 CMIP7 N2O input mass basis.

This is a data-source conversion, not a change to OSCAR equations. Apply it
only to the identified preaggregated archive; gridded N2O uses 28/44 directly.
"""
import hashlib
from pathlib import Path
import numpy as np
import xarray as xr

BASIS = 'TgN_two_nitrogen_atoms_v1'
SCHEMA = 'regional-secondary-harvest-carbon-future-n2o-v2'
DYNAMIC = ('D_Eant_N2O', 'D_Ebb2_N2O')
STATIC = ('Eant_N2O_pi', 'Ebb2_N2O_pi')
LEGACY_FIELDS = DYNAMIC + STATIC


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest.hexdigest())


def pi_identity(parameters):
    """Fingerprint the common regional PI references, independent of broadcast."""
    digest = hashlib.sha256()
    digest.update('|'.join(map(str, parameters.reg_land.values)).encode())
    for name in STATIC:
        value = parameters[name]
        if 'config' in value.dims:
            first = value.isel(config=0, drop=True)
            if not bool((value == first).all()):
                raise ValueError('N2O source PI must be common to the original members')
            value = first
        if value.dims != ('reg_land',) or value.attrs.get('units') != 'TgN yr-1':
            raise ValueError('Invalid regional N2O PI reference: ' + name)
        digest.update(name.encode())
        digest.update(np.asarray(value, dtype='<f8').tobytes())
    return digest.hexdigest()


def correct_legacy_fields(dataset, fields=LEGACY_FIELDS):
    """Convert the four known archive fields once; retain signed anomalies."""
    if dataset.attrs.get('n2o_mass_basis'):
        raise ValueError('Refusing a second N2O mass-basis conversion')
    if not set(fields) <= set(LEGACY_FIELDS):
        raise ValueError('Only the four identified source fields may be converted')
    out = dataset.copy(deep=True)
    for name in fields:
        value = dataset[name]
        if value.attrs.get('units') != 'TgN yr-1' or not bool(np.isfinite(value).all()):
            raise ValueError('Unexpected source units or missing values: ' + name)
        out[name] = 2. * value.astype(np.float64)
        out[name].attrs = dict(value.attrs, mass_basis_correction='Twice the archived v4-beta2.1 value; two nitrogen atoms per N2O molecule')
    out.attrs.update(n2o_mass_basis=BASIS,
        n2o_source_correction='OSCAR v4-beta2.1 CMIP7 preaggregated archive; four-field whitelist; source raw retained')
    if set(STATIC) <= set(out):
        out.attrs['n2o_pi_sha256'] = pi_identity(out)
    return out


def correct_legacy_parameters(parameters):
    """Correct source PI only and regenerate the affected native derived PI."""
    out = correct_legacy_fields(parameters, STATIC)
    if 'Ebb_N2O_pi' in out:
        from oscar._core._model._modules.OSCAR_fire import Eq__Ebb_N2O_pi
        value = Eq__Ebb_N2O_pi(out)
        if value is None:
            raise ValueError('Cannot regenerate the native biomass-burning PI')
        out['Ebb_N2O_pi'] = value
        out.Ebb_N2O_pi.attrs = dict(parameters.Ebb_N2O_pi.attrs)
    return out


def label_forcing(forcing, parameters):
    forcing.attrs.update(n2o_mass_basis=BASIS, n2o_pi_sha256=pi_identity(parameters))
    return forcing


def load_continuation_parameters(identities, original, state, forcing):
    """Read the explicitly bound prepared PI revision, never modify history.

    Legacy executions have no revision and retain their original parameters.
    A corrected forcing must carry a matching revision; mixed inputs fail.
    """
    binding = identities.get('n2o_continuation')
    if binding is None:
        if forcing.attrs.get('n2o_mass_basis') == BASIS:
            raise ValueError('Corrected N2O forcing requires corrected continuation parameters')
        return original
    if binding['basis'] != BASIS or forcing.attrs.get('n2o_mass_basis') != BASIS:
        raise ValueError('N2O continuation and forcing mass bases differ')
    record = binding['parameters']
    if fingerprint(record['path']) != record:
        raise ValueError('Prepared N2O continuation parameters changed')
    with xr.open_dataset(record['path']) as opened:
        out = opened.sel(config=original.config).load()
    source_id = binding['history_run_identity']
    if (original.attrs.get('run_identity') != source_id
            or state.attrs.get('run_identity') != source_id
            or out.attrs.get('source_run_identity') != source_id
            or out.attrs.get('n2o_mass_basis') != BASIS):
        raise ValueError('N2O parameter revision is not bound to this historical state')
    for name in ('config', 'source_config', 'reg_land'):
        np.testing.assert_array_equal(out[name], original[name])
    if set(out) != set(original):
        raise ValueError('Continuation parameter schema changed')
    expected = correct_legacy_parameters(original)
    for name in out:
        xr.testing.assert_equal(out[name], expected[name])
    if (pi_identity(out) != binding['pi_sha256']
            or forcing.attrs.get('n2o_pi_sha256') != binding['pi_sha256']):
        raise ValueError('N2O forcing and continuation PI references differ')
    return out
