from emulator.data.datasets.common import *
from emulator.data.datasets.climate_model import ClimateModelDataset

def _year(ds):
    """Return the array of calendar years from a time‑indexed xr.Dataset."""
    return ds.time.dt.year.values


def _slice_years(ds: xr.Dataset, first: int, last: int) -> xr.Dataset:
    """Inclusive slice from *first* to *last* calendar year."""
    return ds.sel(time=slice(f"{first}", f"{last}"))


def _slice_decades(ds: xr.Dataset, first: int, n_dec: int = 10) -> xr.Dataset:
    """
    Return *n_dec* consecutive entries from a decadal‑resolution dataset,
    starting with the decade whose LAST year == ``first + 9``.
    """
    years = _year(ds)
    idx0 = np.where(years == first + 9)[0][0]
    indices = np.arange(idx0, idx0 + n_dec * 10, 10)
    return ds.isel(time=indices)


def _interpolate(part1, part2):
    # 1. Squeeze experiment dimension if necessary
    if "experiment" in part1.dims:
        part1 = part1.squeeze("experiment", drop=True)

    # 2. Extract the last year of part1 and first year of part2
    end_part1 = part1.isel(time=-1)
    start_part2 = part2.isel(time=0)

    # 3. Create intermediate time range
    interp_times = pd.date_range(
        start=end_part1["time"].values + pd.DateOffset(years=1),
        end=start_part2["time"].values - pd.DateOffset(years=1),
        freq="12MS"
    )

    # 4. Linearly interpolate each variable
    interp_vars = {}
    for var in part1.data_vars:
        diff = (start_part2[var] - end_part1[var])
        n_years = len(interp_times) + 1
        interpolated = [end_part1[var] + diff * (i / n_years) for i in range(1, n_years)]
        interp_vars[var] = xr.concat(interpolated, dim="time")
        interp_vars[var]["time"] = interp_times

    # 5. Create the interpolated dataset
    interp_ds = xr.Dataset(interp_vars)
    for coord in ["lat", "lon"]:
        interp_ds[coord] = part1[coord]
    interp_ds["time"] = interp_times
    interp_ds["member"] = part1["member"]
    interp_ds["scenario"] = part1["scenario"]
    return interp_ds


def _combine(scn_ds: xr.Dataset, hist_ds: xr.Dataset, base_ds: xr.Dataset = None) -> xr.Dataset:
    # Only combine if hist_ds has earlier years than scn_ds
    if hist_ds.time.dt.year.min() <= scn_ds.time.dt.year.min():
        cutoff = scn_ds.time.dt.year.min()
        hist_part = hist_ds.sel(time=hist_ds.time.dt.year < cutoff)
        scn_part  = scn_ds .sel(time=scn_ds .time.dt.year >= cutoff)
        hist_part['scenario'] = scn_part['scenario']
        combined = xr.concat([hist_part, scn_part], dim="time")
        # Enforce time is continuous (no gaps)
        if not (combined.time.dt.year.diff('time') == 1).all():
            base_part = base_ds.sel(time=(base_ds.time.dt.year > hist_part.time.dt.year.max()) & (base_ds.time.dt.year < scn_part.time.dt.year.min()))
            if base_part.time.size == 0:
                interp_part = _interpolate(hist_part, scn_part)
                combined = xr.concat([hist_part, interp_part, scn_part], dim="time")
            else:
                base_part['scenario'] = scn_part['scenario']
                interp_part = _interpolate(hist_part, base_part)
                combined = xr.concat([hist_part, interp_part, base_part, scn_part], dim="time")
        assert (combined.time.dt.year.diff('time') == 1).all(), \
            "Time dimension is not continuous after combining historical and scenario data."
        return combined
    else:
        # Scenario data already contains all historical data
        return scn_ds

def _combine_safe(scn_ds: xr.Dataset, hist_ds: xr.Dataset, base_ds: xr.Dataset = None) -> xr.Dataset:
    def _squeeze1(ds):
        if "experiment" in ds.dims and ds.sizes["experiment"] == 1:
            ds = ds.squeeze("experiment", drop=True)
        return ds

    cutoff = scn_ds.time.dt.year.min().item()
    hist_part = hist_ds.sel(time=hist_ds.time.dt.year <  cutoff)
    scn_part  = scn_ds .sel(time=scn_ds .time.dt.year >= cutoff)

    # normalize singleton dims so xarray concat works
    hist_part = _squeeze1(hist_part)
    scn_part  = _squeeze1(scn_part)

    # Give hist the target scenario coord for consistent metadata
    hist_part = hist_part.assign_coords(scenario=scn_part["scenario"])

    combined = xr.concat([hist_part, scn_part], dim="time")

    if not (combined.time.dt.year.diff("time") == 1).all():
        # There's a gap; try base if provided
        base_part = None
        if base_ds is not None:
            base_part = base_ds.sel(
                time=(base_ds.time.dt.year > hist_part.time.dt.year.max()) &
                      (base_ds.time.dt.year < scn_part.time.dt.year.min())
            )
            if base_part.sizes.get("time", 0) > 0:
                base_part = _squeeze1(base_part).assign_coords(scenario=scn_part["scenario"])

        pieces = [hist_part]

        # hist → base
        if base_part is not None and base_part.sizes["time"] > 0:
            # If there’s a gap between hist end and base start ≥ 2, interpolate just those years
            y_end_hist  = hist_part.time.dt.year.max().item()
            y_start_base = base_part.time.dt.year.min().item()
            if (y_start_base - y_end_hist) > 1:
                pieces.append(_interpolate(hist_part, base_part))
            pieces.append(base_part)

            # base → scn
            y_end_base  = base_part.time.dt.year.max().item()
            y_start_scn = scn_part.time.dt.year.min().item()
            if (y_start_scn - y_end_base) > 1:
                pieces.append(_interpolate(base_part, scn_part))
        else:
            # No usable base_part; interpolate directly hist → scn only if needed
            y_end_hist  = hist_part.time.dt.year.max().item()
            y_start_scn = scn_part.time.dt.year.min().item()
            if (y_start_scn - y_end_hist) > 1:
                pieces.append(_interpolate(hist_part, scn_part))

        pieces.append(scn_part)
        # filter out empty interpolation pieces
        pieces = [p for p in pieces if p is not None and p.sizes.get("time", 0) > 0]
        combined = xr.concat(pieces, dim="time")

    # Final sanity check
    assert (combined.time.dt.year.diff("time") == 1).all(), \
        "Time is not continuous after combining."

    return combined


class TimescaleAlignedClimateModelDataset(ClimateModelDataset):
    """
    Dataset that returns aligned decadal, yearly, and monthly samples.

    **Triplet definition**

    * ``dec_year0`` – first calendar year of the 100‑year decadal block
    * ``yr_year0``  – first calendar year of the 10‑year yearly block
    * ``mo_year``   – calendar year whose 12 monthly targets are returned

    The three integers satisfy

        dec_year0 ≤ yr_year0 ≤ mo_year ≤ yr_year0+9 ≤ dec_year0+99
    """
    # first scenario year (inclusive)
    _HIST_CUTOFF = 2015  

    # fixed window lengths (years / months)
    _LEN_DEC_YRS = 100   # input
    _LEN_YR_YRS  = 10
    _LEN_MO_MON  = 12    # output; X is still one *year*

    def __init__(
        self,
        *args,
        timescales=("decadal", "yearly", "monthly"),
        align_to_decade: bool = False,
        min_start_year: int = None,
        max_start_year: int = None,
        **kw,
    ):
        num_ensemble_member = kw.get("num_ensemble_member", -1)
        kw["num_ensemble_member"] = -1
        super().__init__(*args, timescales=timescales, **kw)

        if set(timescales) != {"decadal", "yearly", "monthly"}:
            raise ValueError("timescales must be exactly ('decadal','yearly','monthly')")

        self.align_to_decade = align_to_decade
        self.min_start_year = 0 if min_start_year is None else min_start_year
        self.max_start_year = 3000 if max_start_year is None else max_start_year

        # quick look-ups for the base ClimateModelDataset record indices
        self._rec_lut = {
            rec: idx for idx, rec in enumerate(self._records)
        }  # (ts, scn, mem, start) -> idx in ClimateModelDataset

        self._yearly_in  = {}  # (scn,mem) -> xr.Dataset
        self._yearly_out = {}
        self._dec_out    = {}  # combined decadal outputs

        self._min_yearly_out = {}
        self._min_dec_out = {}
        self._aligned_members: dict[str, list[str]] = {}

        for scn in self.scenarios:
            self._aligned_members[scn] = [
                mem
                for mem in self.get_unique_members()
                if ("yearly", self.historical_key, mem) in self._cache
                and ("yearly", scn, mem) in self._cache
                and ("decadal", self.historical_key, mem) in self._cache
                and ("decadal", scn, mem) in self._cache
            ]
            for mem in self._aligned_members[scn]:
                # yearly inputs --------------------------------------------
                hist_inp, hist_out = self._cache[("yearly", self.historical_key, mem)]
                scn_inp , scn_out  = self._cache[("yearly", scn,          mem)]
                if self.IB:
                    # For IB data, we need to fill in the gap between historical and scenario data
                    if scn == self.historical_key:
                        self._yearly_in [(scn, mem)] = scn_inp
                    else:
                        if self.climate_model == "UKESM1-0-LL":
                            fire = OPENBURNING_MODEL_MAPPING[self.climate_model][1]
                            base_path = self.store.input_store("yearly", fire, "ssp245", self.historical_key, False)
                            base_inp = xr.open_zarr(base_path, consolidated=True)
                            if self.regrid_input:
                                base_inp = self.regrid_xarray_dataset(base_inp, self.grid)
                            base_inp = _to_month_start(base_inp)
                            # make sure vars/dims line up
                            base_inp = base_inp[self.ivars]
                            if "member" in base_inp.dims and mem in base_inp.member.values:
                                base_inp = base_inp.sel(member=mem)
                            self._yearly_in [(scn, mem)] = _combine_safe(scn_inp, hist_inp, base_inp)
                        else:
                            base_inp, base_out = self._cache[("yearly", "MA-BASELINE", mem)]
                            self._yearly_in [(scn, mem)] = _combine(scn_inp, hist_inp, base_inp)
                else:
                    self._yearly_in [(scn, mem)] = _combine(scn_inp, hist_inp)
                self._yearly_out[(scn, mem)] = scn_out

                # Decadal history is already concatenated in the underlying data
                self._dec_out[(scn, mem)] = self._cache[("decadal", scn, mem)][1]

                self._min_yearly_out[(scn, mem)] = self._yearly_out[(scn, mem)].time.dt.year.min().item()
                self._min_dec_out[(scn, mem)] = self._dec_out[(scn, mem)].time.dt.year.min().item()

        self._triplets: list[tuple[str, str, int, int, int]] = []
        for scn in self.scenarios:
            for mem in self._aligned_members.get(scn, []):
                if (scn, mem) not in self._yearly_in:
                    continue
                yr_ds = self._yearly_in[(scn, mem)]
                years = _year(yr_ds)                       # ascending, consecutive

                # guard: need 100 consecutive years
                if len(years) < self._LEN_DEC_YRS:
                    continue

                max_dec_start = len(years) - self._LEN_DEC_YRS
                dec_start_indices = range(max_dec_start + 1)
                if self.align_to_decade:
                    dec_start_indices = [
                        i for i in dec_start_indices
                        if int(years[i]) % 10 == 0
                    ]

                for dec_idx in dec_start_indices:
                    dec_year0 = int(years[dec_idx])

                    yr_start_range = range(dec_year0, dec_year0 + 91)
                    if self.align_to_decade:
                        yr_start_range = [
                            y for y in yr_start_range if y % 10 == 0
                        ]
                    for yr_year0 in yr_start_range:
                        # monthly sample year •inside• the 10‑yr block
                        for mo_year in range(yr_year0, yr_year0 + self._LEN_YR_YRS):
                            
                            if (
                                (
                                    scn != self.historical_key and
                                    (
                                    mo_year < self._HIST_CUTOFF or
                                    yr_year0 < self._HIST_CUTOFF
                                    )
                                ) or
                                    (
                                        mo_year < max(self.min_start_year, self._min_yearly_out[(scn, mem)]) or
                                        yr_year0 < max(self.min_start_year, self._min_yearly_out[(scn, mem)]) or
                                        dec_year0 < max(self.min_start_year, self._min_dec_out[(scn, mem)]) or 
                                        mo_year > self.max_start_year or
                                        yr_year0 > self.max_start_year or
                                        dec_year0 > self.max_start_year
                                    )
                            ):
                                # Never sample historical months or years unless this is the historical scenario
                                # and never sample before the minimum start year or before the minimum year in the outputs
                                continue  
                            self._triplets.append(
                                (scn, mem, dec_year0, yr_year0, mo_year)
                            )

        # The parent’s records are no longer needed – free a bit of RAM
        self._records = []
        # Filter by ensemble members here if needed
        if num_ensemble_member > 0:
            self.filter_by_number_of_members(num_ensemble_member)
            self.num_ensemble_member = num_ensemble_member
        print("Total triplets:", len(self._triplets))
        self._original_triplets = self._triplets.copy()

    def _build_tensor(self, xr_sel: xr.Dataset, stats: dict) -> torch.Tensor:
        """
        xr.Dataset -> torch.Tensor with shape (time, var, lat, lon), or the
        swapped (time, var, lon, lat) layout for checkpoint reproduction.
        """
        xr_sel = xr_sel.fillna(0.0)
        xr_sel = xr_sel.transpose("time", ..., "lat", "lon")
        xr_sel = normalize(xr_sel, stats, self.log_vars, self.min_max_norm)
        t = (
            torch.as_tensor(xr_sel.to_array().values, dtype=torch.float32)
            .movedim(0, 1)        # [time, var, lat, lon]
            .squeeze(2)           # drop 'member'
        )
        # Keep swap_lat_lon for reproducing checkpoints trained with swapped spatial axes.
        if self.spatial_layout == "swap_lat_lon":
            t = t.movedim(-1, -2) # [time, var, lat, lon] -> [time, var, lon, lat]
        return t

    def filter_by_member(self, member: str):
        """Filter the dataset by a specific ensemble member."""
        filtered_triplets = [
            triplet for triplet in self._triplets if triplet[1] == member
        ]
        self._triplets = filtered_triplets

    def filter_by_number_of_members(self, n_members: int):
        """Filter the dataset to only include a specific number of ensemble members."""
        unique_members = self.get_unique_members()
        selected_members = set(unique_members[:n_members])
        filtered_triplets = [
            triplet for triplet in self._triplets if triplet[1] in selected_members
        ]
        self._triplets = filtered_triplets

    def filter_by_year_range(self, start_year: int, end_year: int) -> None:
        if start_year > end_year:
            raise ValueError("start_year must be <= end_year")

        def overlaps(a0, a1, b0, b1):  # inclusive
            return not (a1 < b0 or b1 < a0)

        kept = []
        for scn, mem, dec_year0, yr_year0, mo_year in self._triplets:
            dec_ok = overlaps(dec_year0, dec_year0 + self._LEN_DEC_YRS - 1, start_year, end_year)
            yr_ok  = overlaps(yr_year0,  yr_year0  + self._LEN_YR_YRS  - 1, start_year, end_year)
            mo_ok  = (start_year <= mo_year <= end_year)
            if dec_ok and yr_ok and mo_ok:   # require all three to overlap
                kept.append((scn, mem, dec_year0, yr_year0, mo_year))
        self._triplets = kept

    def get_unique_members(self) -> List[str]:
        """Get unique ensemble members in the dataset."""
        if len(self._records) == 0:
            return sorted(set(triplet[1] for triplet in self._triplets))
        else:
            return super().get_unique_members()

    def get_unique_scenarios(self) -> List[str]:
        """Get unique scenarios in the dataset."""
        if len(self._records) == 0:
            return sorted(set(triplet[0] for triplet in self._triplets))
        else:
            return super().get_unique_scenarios()

    def filter_duplicates(
        self,
        timescale: str,
        keep: str = "first",      # "first" or "random"
        seed: int | None = None,  # for deterministic "random"
        min_keep: int | None = None,
    ) -> int:
        """
        Remove duplicate samples for the chosen timescale by keeping one representative
        per unique target-window key. If min_keep is provided, ensure the resulting
        dataset has at least that many items by retaining extra (duplicate) entries.

        timescale:
            - "decadal": duplicates share (scenario, member, dec_year0)
            - "yearly" : duplicates share (scenario, member, yr_year0)
            - "monthly": duplicates share (scenario, member, mo_year)
        keep:
            - "first"  : keep the first occurrence per group (stable)
            - "random" : keep one at random per group (optionally seeded)
        min_keep:
            - If given, guarantee the final len(self) >= min_keep by adding back
            additional indices chosen consistently with `keep`. Never exceeds the
            original size.

        Returns:
            The number of triplets removed (original_len - new_len).
        """
        valid = {"decadal", "yearly", "monthly"}
        if timescale not in valid:
            raise ValueError(f"timescale must be one of {valid}, got {timescale!r}")
        if keep not in {"first", "random"}:
            raise ValueError("keep must be 'first' or 'random'")
        if min_keep is not None and min_keep <= 0:
            raise ValueError("min_keep must be a positive integer if provided.")

        import random
        rng = random.Random(seed) if (keep == "random" and seed is not None) else random

        def key_for(triplet: tuple[str, str, int, int, int]):
            scn, mem, dec_year0, yr_year0, mo_year = triplet
            if timescale == "decadal":
                return (scn, mem, dec_year0)
            elif timescale == "yearly":
                return (scn, mem, yr_year0)
            else:  # "monthly"
                return (scn, mem, mo_year)

        # Group indices by deduplication key
        groups: dict[tuple, list[int]] = {}
        for idx, trip in enumerate(self._triplets):
            groups.setdefault(key_for(trip), []).append(idx)

        original_len = len(self._triplets)

        # Primary selection: pick one index per group
        kept_indices = []
        leftover_indices = []
        for _, idxs in groups.items():
            if keep == "first":
                kept_indices.append(idxs[0])
                leftover_indices.extend(idxs[1:])  # deterministic order
            else:  # random
                choice = rng.choice(idxs)
                kept_indices.append(choice)
                leftover_indices.extend(i for i in idxs if i != choice)

        # If we need to keep more to satisfy min_keep, add from leftovers
        if min_keep is not None and len(kept_indices) < min_keep:
            need = min(min_keep, original_len) - len(kept_indices)
            if need > 0 and leftover_indices:
                if keep == "random":
                    # sample without replacement from leftovers
                    # if need > len(leftovers), sample all (handled by min above)
                    extra = rng.sample(leftover_indices, k=min(need, len(leftover_indices)))
                else:
                    # stable fill from leftover order
                    extra = leftover_indices[:need]
                kept_indices.extend(extra)

        # Build the new triplet list
        keep_set = set(kept_indices)
        self._triplets = [t for i, t in enumerate(self._triplets) if i in keep_set]

        removed = original_len - len(self._triplets)
        return removed

    def buffer(self, buffer_size: int):
        """Buffer the dataset by a specific size."""
        if buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0.")
        first_triplet = self._triplets[0]
        self._triplets += [first_triplet] * buffer_size

    def reset(self):
        """Reset the dataset to its original state."""
        self._triplets = self._original_triplets.copy()

    def __len__(self) -> int:
        return len(self._triplets)

    def __getitem__(self, idx: int):
        scn, mem, dec_year0, yr_year0, mo_year = self._triplets[idx]

        # handy references -------------------------------------------------
        yr_in  = self._yearly_in [(scn, mem)][self.ivars]
        yr_out = self._yearly_out[(scn, mem)][self.ovars]
        dec_out = self._dec_out[(scn, mem)][self.ovars]

        # ------------- decadal -------------------------------------------
        X_dec_in  = _slice_years(yr_in,  dec_year0, dec_year0 + 99)
        Y_dec_out = _slice_decades(dec_out, dec_year0)

        dec_sample = {
            "X": self._build_tensor(X_dec_in,  self._in_stats["yearly"]),
            "Y": self._build_tensor(Y_dec_out, self._out_stats["decadal"]),
            "metadata": {
                "input_stats": self._in_stats["yearly"],
                "stats":       self._out_stats["decadal"],
                "X_time": (dec_year0, dec_year0 + 99),
                "Y_time": (_year(Y_dec_out := Y_dec_out).min() - 9, _year(Y_dec_out).max()),
            },
        }

        # ------------- yearly --------------------------------------------
        X_yr_in  = _slice_years(yr_in,  yr_year0, yr_year0 + 9)
        Y_yr_out = _slice_years(yr_out, yr_year0, yr_year0 + 9)

        yr_sample = {
            "X": self._build_tensor(X_yr_in,  self._in_stats["yearly"]),
            "Y": self._build_tensor(Y_yr_out, self._out_stats["yearly"]),
            "metadata": {
                "input_stats": self._in_stats["yearly"],
                "stats":       self._out_stats["yearly"],
                "X_time": (yr_year0, yr_year0 + 9),
                "Y_time": (yr_year0, yr_year0 + 9),
            },
        }

        # ------------- monthly -------------------------------------------
        # X_mo_in = _slice_years(yr_in, mo_year, mo_year)           # one‑year slice

        # choose the correct monthly output store (hist vs scn)
        mo_key = (self.historical_key if mo_year < self._HIST_CUTOFF else scn, mem)
        mo_out_ds = self._cache[("monthly", mo_key[0], mo_key[1])][1][self.ovars]
        Y_mo_out = mo_out_ds.sel(time=mo_out_ds.time.dt.year == mo_year)

        X_mo_in_ds = _slice_years(yr_in, mo_year, mo_year)
        X_mo = self._build_tensor(X_mo_in_ds, self._in_stats["yearly"])
        in_stats = self._in_stats["yearly"]

        mo_sample = {
            # "X": self._build_tensor(X_mo_in, self._in_stats["yearly"]),
            "X": X_mo,
            "Y": self._build_tensor(Y_mo_out, self._out_stats["monthly"]),
            "metadata": {
                # "input_stats": self._in_stats["yearly"],
                "input_stats": in_stats,
                "stats":       self._out_stats["monthly"],
                "X_time": (mo_year, mo_year),
                "Y_time": (mo_year, mo_year),
            },
        }

        # ------------- wrap & return -------------------------------------
        outer_meta = {
            "climate_model": self.climate_model,
            "scenario":      scn,
            "member":        mem,
            "latitude":      self.grid.lat_bnds.values[0].mean(axis=1),
            "longitude":     self.grid.lon_bnds.values[0].mean(axis=1),
            "input_latitude": self.input_grid.lat.values,
            "input_longitude": self.input_grid.lon.values,
            "spatial_layout": self.spatial_layout,
        }

        return {
            "decadal": dec_sample,
            "yearly":  yr_sample,
            "monthly": mo_sample,
            "metadata": outer_meta,
        }
