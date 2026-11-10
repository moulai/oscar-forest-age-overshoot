"""Lossless assembly of ordered member batches in NetCDF files."""
from __future__ import annotations

import numpy as np
from netCDF4 import Dataset


def merge_batch_chunks(paths, output, ids, attrs):
    """Merge 500 ordered members using the source variables and storage chunks."""
    with Dataset(paths[0]) as template, Dataset(output, "w", format="NETCDF4") as destination:
        for name, dim in template.dimensions.items():
            destination.createDimension(name, 500 if name == "config" else len(dim))
        destination.setncatts({**{k: template.getncattr(k) for k in template.ncattrs()}, **attrs})
        for name, variable in template.variables.items():
            options = {}
            if "_FillValue" in variable.ncattrs():
                options["fill_value"] = variable.getncattr("_FillValue")
            if variable.dtype is not str and np.dtype(variable.dtype).kind not in ("O", "S", "U"):
                options.update(zlib=True, complevel=2)
                chunks = variable.chunking()
                if isinstance(chunks, list):
                    options["chunksizes"] = tuple(chunks)
            target = destination.createVariable(name, variable.datatype, variable.dimensions, **options)
            target.setncatts({k: variable.getncattr(k) for k in variable.ncattrs() if k != "_FillValue"})
        position = 0
        for path in paths:
            with Dataset(path) as chunk:
                count = len(chunk.dimensions["config"])
                if chunk.variables["source_config"][:].tolist() != ids[position:position + count]:
                    raise RuntimeError("Batch identity/order changed during publication")
                for name, variable in chunk.variables.items():
                    target = destination.variables[name]
                    if "config" in variable.dimensions:
                        index = [slice(None)] * variable.ndim
                        index[variable.dimensions.index("config")] = slice(position, position + count)
                        target[tuple(index)] = variable[:]
                    elif position == 0:
                        target[...] = variable[...]
                    elif not np.array_equal(target[...], variable[...], equal_nan=(
                            np.dtype(variable.dtype).kind in ("f", "c") if variable.dtype is not str else False)):
                        raise RuntimeError("A non-member quantity changed between batches: " + name)
                position += count
        if position != 500:
            raise RuntimeError("Diagnostic batches do not contain all 500 ordered members")
    # Read the closed container header and its small member coordinate once.
    # Numeric fields above are copied directly at unchanged dtype with lossless
    # compression; storage chunks are the only changed container setting.
    with Dataset(paths[0]) as template, Dataset(output) as written:
        if set(template.variables) != set(written.variables):
            raise RuntimeError("Published diagnostic variable names changed")
        if written.variables["source_config"][:].tolist() != ids:
            raise RuntimeError("Published diagnostic member order changed")
        for name, source in template.variables.items():
            target = written.variables[name]
            if source.dtype != target.dtype or source.dimensions != target.dimensions:
                raise RuntimeError("Published diagnostic dtype/dimensions changed: " + name)
            if set(source.ncattrs()) != set(target.ncattrs()):
                raise RuntimeError("Published diagnostic attributes changed: " + name)
            for attribute in source.ncattrs():
                expected, actual = source.getncattr(attribute), target.getncattr(attribute)
                numeric = np.asarray(expected).dtype.kind in ("f", "c")
                if not np.array_equal(expected, actual, equal_nan=numeric):
                    raise RuntimeError("Published diagnostic attribute changed: " + name + "." + attribute)
