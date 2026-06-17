from emulator.data.datasets.common import *

class ClimateModelDataset(torch.utils.data.Dataset):
    """Single-climate-model dataset for one or more independent timescales."""

    def __init__(
        self,
        output_data_dir: str | Path,
        climate_model: str,
        input_variables: Sequence[str],
        output_variables: Sequence[str],
        input_seq_len: Dict[str, int],
        output_seq_len: Dict[str, int],
        scenarios: Sequence[str] | None = None,
        num_ensemble_member: int = -1,
        timescales: Sequence[str] = ("yearly",),
        regrid_input: bool = False,
        IB: bool = False,
        min_max_norm: bool = False,
        log_vars: list[str] = [],
        min_start_year: int = 0,
        max_start_year: int = 3000,
        align_to_decade: bool = False,
        spatial_layout: str = "swap_lat_lon",
        member_variant_labels: Sequence[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        if spatial_layout not in SPATIAL_LAYOUTS:
            raise ValueError(f"spatial_layout must be one of {SPATIAL_LAYOUTS}, got {spatial_layout!r}.")
        self.root = Path(output_data_dir)
        self.climate_model = climate_model
        self.ivars = list(input_variables)
        self.ovars = list(output_variables)
        self.in_len = input_seq_len
        self.out_len = output_seq_len
        self.scenarios = scenarios
        self.num_ensemble_member = num_ensemble_member
        self.timescales = list(timescales)
        # Overlapping stride for decades and years to create more examples and allow more flexibility
        self.stride = {
            "monthly": 12,
            "yearly": 1,
            "decadal": 1,
        }
        # For monthly, use a stride of output length when the output length is not 12
        if "monthly" in self.out_len and self.out_len["monthly"] != 12:
            self.stride["monthly"] = self.out_len["monthly"]
        self.IB = IB
        self.min_max_norm = min_max_norm
        self.log_vars = log_vars

        self.min_start_year = min_start_year
        self.max_start_year = max_start_year
        self.align_to_decade = align_to_decade
        self.spatial_layout = spatial_layout
        self.member_variant_labels = list(member_variant_labels) if member_variant_labels is not None else None
        self.store = ClimateSuiteDataStore(self.root)

        self.grid = xr.open_dataset(self.store.grid_path, chunks="auto", engine="h5netcdf")
        self.input_grid = None

        self._records: List[Tuple] = []  # (ts, scn, mem, t_start)
        self._cache: Dict[Tuple[str, str], Tuple[xr.Dataset, xr.Dataset]] = {}
        self._in_stats: Dict[str, dict] = {}   # keyed by timescale only
        self._out_stats: Dict[str, dict] = {}
        self.regrid_input = regrid_input
        self.aod = None

        hist_fire, scen_fire = OPENBURNING_MODEL_MAPPING.get(
            climate_model, OPENBURNING_MODEL_MAPPING["other"]
        )

        # ------------------------------------------------------------------
        # iterate timescales and scenarios, build sample index
        # ------------------------------------------------------------------
        # If the original config doesn't specify historical scenario, we need to add it manually for __init__ only
        # as __getitem__ would need historical data to concate
        self.historical_key = "MA-HISTORICAL" if self.IB else "historical"
        if self.historical_key not in scenarios:
            scenarios.append(self.historical_key)

        for ts in self.timescales:
            print(f"Processing timescale: {ts} for {climate_model} with scenarios: {','.join(scenarios)}")
            stride_ts = self.stride[ts]
            in_span = self.in_len[ts]
            out_span = self.out_len[ts]

            # list scenarios
            if scenarios is None:
                base = self.root / "outputs" / climate_model / ts
                scn_names = sorted(p.stem for p in base.glob("*.zarr"))
            else:
                scn_names = list(scenarios)

            # open all datasets *once* (needed for stats + record length)
            inp_datasets, out_datasets = [], []
            for scn in tqdm(scn_names):
                fire = hist_fire if scn == self.historical_key else scen_fire
                inp_store = self.store.input_store(ts, fire, scn, self.historical_key, self.IB)
                out_store = self.store.output_store(climate_model, ts, scn)
                base_in = xr.open_zarr(inp_store, consolidated=True)  # Defaults to chunks='auto'
                base_out = xr.open_zarr(out_store, consolidated=True)  # Defaults to chunks='auto'

                if self.regrid_input:
                    base_in = self.regrid_xarray_dataset(base_in, self.grid)

                if self.input_grid is None:
                    self.input_grid = base_in

                base_in = _to_month_start(base_in)
                base_out = _to_month_start(base_out)

                # limit ensemble members
                member_labels = list(base_out.member.values)
                if not self.IB and self.member_variant_labels is not None:
                    member_labels = self.filter_member_variants(member_labels)

                if "member" in base_in.dims:
                    member_labels = [
                        m for m in member_labels if m in base_in.member.values
                    ]

                member_labels = member_labels[: self.num_ensemble_member] if self.num_ensemble_member > -1 else member_labels

                # loop over members -----------------------------------------------
                for mem in member_labels:
                    inp = base_in
                    out = base_out.sel(member=mem)

                    if "member" in inp.dims and mem in inp.member.values:
                        inp = inp.sel(member=mem)

                    # prune all-NaN timesteps FOR THIS MEMBER
                    inp, out = _align_and_prune(inp, out)
                    if ts == "monthly" and (inp.sizes["time"] < in_span or out.sizes["time"] < out_span):
                        print(f"Skipping {ts} {scn} {mem} due to insufficient data length: {inp.sizes['time']} vs {in_span}, {out.sizes['time']} vs {out_span}")
                        continue
                    if inp.sizes["time"] == 0 or out.sizes["time"] == 0:
                        print(f"Skipping {ts} {scn} {mem} due to empty dataset.")
                        continue

                    # cache pruned datasets  --------------------------------------
                    self._cache[(ts, scn, mem)] = (inp, out)

                    # build sample index ------------------------------------------
                    times = out.time.values
                    span = max(in_span, out_span)
                    for start in range(0, len(times) - span + 1, stride_ts):
                        # NOTE: We can return shorter windows for yearly and decadal data
                        if ts == "monthly" and start + span > len(times):
                            break

                        # Filter out records that start before the minimum start year
                        start_year = times[start].astype("datetime64[Y]").astype(int) + 1970
                        if start_year < self.min_start_year or start_year > self.max_start_year:
                            continue
                        # Newly added
                        if self.align_to_decade:
                            if ts == "monthly":
                                # Require January of a decade year (e.g., 1990-01, 2000-01, …)
                                this_time = pd.to_datetime(times[start])
                                if not (this_time.year % 10 == 0 and this_time.month == 1):
                                    continue
                            elif ts == "yearly":
                                # Require the first year of a decade (… 1990, 2000, 2010, …)
                                if start_year % 10 != 0:
                                    continue
                            elif ts == "decadal":
                                # NOTE: decadal time is labeled by the LAST year of the decade
                                # (e.g., 2039 for 2030–2039), so decade-aligned starts have year % 10 == 9.
                                if start_year % 10 != 9:
                                    continue
                        self._records.append((ts, scn, mem, start))

                    inp_datasets.append(inp[self.ivars])
                    out_datasets.append(out[self.ovars])

            # compute stats per timescale across all scenarios
            if ts not in self._in_stats:
                self._in_stats[ts]  = _get_or_make_stats(inp_datasets,  self.climate_model, self.IB, ts,
                                                         "inputs", self.min_max_norm, self.log_vars)
                self._out_stats[ts] = _get_or_make_stats(out_datasets, self.climate_model, self.IB, ts,
                                                         "outputs", self.min_max_norm, self.log_vars)

        self._original_records = self._records.copy()  # keep original records for filtering


    def regrid_xarray_dataset(self, dataset: xr.Dataset, target_grid: xr.Dataset) -> xr.Dataset:
        """Regrids an xarray dataset by a given factor. Currently uses
        xesmf Regridder.
        Args:
            dataset (xr.Dataset): Dataset to be downsampled.
            target_grid (xr.Dataset): Target Dataset.
        Returns:
            xr.Dataset: Upsampled dataset.
        """
        import xesmf as xe

        current_lat = dataset.coords['lat']
        current_lon = dataset.coords['lon']
        source_grid = {'lon': current_lon, 'lat': current_lat}

        regridder = xe.Regridder(source_grid, target_grid, 'bilinear', periodic=True)

        regridded_dataset = regridder(dataset)

        return regridded_dataset

    def filter_member_variants(self, member: list[str]):
        return filter_member_variant_labels(member, self.member_variant_labels)

    def _open_pair(self, ts, scn, mem):
        return self._cache[(ts, scn, mem)]

    def __len__(self):
        return len(self._records)

    def get_unique_members(self):
        """Get unique ensemble members in the dataset."""
        unique_members = set()
        for record in self._records:
            ts, scn, mem, start = record
            unique_members.add(mem)
        return sorted(list(unique_members))

    def get_unique_scenarios(self):
        """Get unique scenarios in the dataset."""
        unique_scenarios = set()
        for record in self._records:
            ts, scn, mem, start = record
            unique_scenarios.add(scn)
        return sorted(list(unique_scenarios))

    def _record_label_year(
        self, ts: str, scn: str, mem: str, start: int
    ) -> tuple[int, int]:
        """
        Return a (label_year, span_years) for the *target* window of a record.
        For decadal: label_year = LAST year of the decade (e.g., 2039 for 2030–2039).
        For monthly: label_year = first year of the monthly target window (typically == the calendar year when out_len=12).
        For yearly:  label_year = first year of the yearly target window.
        span_years is the number of calendar years covered by the target window (useful for sanity checks when monthly spans year boundaries).
        """
        inp_ds, out_ds = self._open_pair(ts, scn, mem)

        in_len  = self.in_len[ts]
        out_len = self.out_len[ts]
        start_o = start
        end_o   = start + out_len
        if ts != "monthly":
            end_o = min(end_o, len(out_ds.time))

        sel_out = out_ds.isel(time=slice(start_o, end_o))

        if ts == "decadal":
            # One step at decadal resolution → time labels are "last year of each decade"
            y0 = int(sel_out["time"].dt.year.values[0])     # e.g., 2039
            return y0, 10

        # monthly / yearly → use the first target timestamp as the label
        years = sel_out["time"].dt.year.values
        label = int(years[0])
        span  = int(years[-1]) - int(years[0]) + 1
        return label, span

    def filter_by_year_condition(
        self,
        condition: Callable[[int], bool],
        decade_label: Literal["end", "start"] = "end",
    ):
        """
        Keep only records of timescale `ts` whose label year satisfies `condition`.

        Semantics:
        - yearly:  label_year = first target year
        - monthly: label_year = first target year (works whether out_len=12 or ≠12)
        - decadal: label_year = last year of the decade (e.g., 2039). If decade_label="start",
                    we map the passed 'year' to (year+9) internally before applying `condition`.
        """
        filtered = []
        for (ts, scn, mem, start) in self._records:

            label_year, _ = self._record_label_year(ts, scn, mem, start)

            # Allow decade filtering in terms of either decade END (default) or START year
            # If user intends to filter by decade START (e.g., 2030), convert to the END label (2039)
            if ts == "decadal" and decade_label == "start":
                # invert: condition expects START, but label is END → transform the year
                # We'll test the *equivalent* end year by passing y+9 through condition.
                # Implement this by wrapping condition:
                keep = condition(label_year - 9)
            else:
                keep = condition(label_year)

            if keep:
                filtered.append((ts, scn, mem, start))

        self._records = filtered

    def filter_by_year(
        self,
        year: int,
        *,
        decade_label: Literal["end", "start"] = "end",
    ):
        self.filter_by_year_condition(lambda y: y == year, decade_label=decade_label)

    def filter_by_year_range(
        self,
        start_year: int,
        end_year: int,
        *,
        decade_label: Literal["end", "start"] = "end",
    ):
        self.filter_by_year_condition(
            lambda y: start_year <= y <= end_year, decade_label=decade_label
        )

    def filter_by_member(self, member: str):
        """Filter the dataset by a specific ensemble member."""
        filtered_records = []
        for record in self._records:
            ts, scn, mem, start = record
            if mem == member:
                filtered_records.append(record)
        self._records = filtered_records

    def buffer(self, buffer_size: int):
        """Buffer the dataset by a specific size."""
        if buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0.")
        first_record = self._records[0]
        self._records += [self._records[0]] * buffer_size

    def reset(self):
        """Reset the dataset to its original state."""
        self._records = self._original_records.copy()

    def _to_tensor(self, xr_block: xr.Dataset, stats: dict) -> torch.Tensor:
        """
        Convert xr.Dataset -> torch.Tensor with guaranteed shape.
        The canonical layout is (time, channel, lat, lon). Current configs use
        swap_lat_lon to reproduce checkpoints trained with swapped spatial axes.

        Steps
        1. normalise & fill NaN
        2. drop member axis if it exists
        3. transpose to (time, var, lat, lon)
        4. torch conversion
        """
        xr_block = xr_block.fillna(0.0)
        xr_block = normalize(xr_block, stats, self.log_vars, self.min_max_norm)

        # --- ensure there is no member axis ---------------------------------
        if "member" in xr_block.dims:
            xr_block = xr_block.squeeze("member", drop=True)
        elif "member" in xr_block.coords:
            xr_block = xr_block.drop_vars("member")

        # --- ensure there is no experiment axis --------------------------------
        if "experiment" in xr_block.dims:
            xr_block = xr_block.squeeze("experiment", drop=True)
        elif "experiment" in xr_block.coords:
            xr_block = xr_block.drop_vars("experiment")

        # --- ensure there is no scenario axis --------------------------------
        if "scenario" in xr_block.dims:
            if xr_block.sizes["scenario"] == 2:
                # Set it to the second scenario
                xr_block = xr_block.isel(scenario=1)
            else:
                xr_block = xr_block.squeeze("scenario", drop=True)
        elif "scenario" in xr_block.coords:
            xr_block = xr_block.drop_vars("scenario")

        # --- put axes in canonical order -----------------------------------
        xr_block = xr_block.transpose("time", ..., "lat", "lon")

        t = torch.as_tensor(xr_block.to_array().values, dtype=torch.float32)
        # to_array() puts 'variable' first, so move to second axis
        t = t.movedim(0, 1)                     # (var, time, lat, lon) → (time, var, lat, lon)

        # Keep swap_lat_lon for reproducing checkpoints trained with swapped spatial axes.
        if self.spatial_layout == "swap_lat_lon":
            t = t.movedim(-1, -2)               # (time, var, lat, lon) -> (time, var, lon, lat)

        return t

    def __getitem__(self, idx: int):
        ts, scn, mem, start = self._records[idx]
        inp_ds, out_ds = self._open_pair(ts, scn, mem)

        # Slice the time dimension and load the data
        in_len = self.in_len[ts]
        out_len = self.out_len[ts]
        in_end = start + in_len
        out_end = start + out_len
        if ts != "monthly":
            in_end = min(in_end, len(inp_ds.time))
            out_end = min(out_end, len(out_ds.time))

        t_in = slice(start, in_end)
        t_out = slice(start, out_end)

        sel_in  = inp_ds.isel(time=t_in )[self.ivars]
        sel_out = out_ds.isel(time=t_out)[self.ovars]

        if self.aod is not None:
            # If AOD is set, overwrite it in the input variables
            if "AOD" in self.ivars:
                if len(sel_in["time"].values) > 1:
                    raise ValueError("AOD variable cannot be set for multiple time steps.")
                year = pd.DatetimeIndex(sel_in["time"].values)[0].year
                aod_slice = self.aod.sel(year=year)        # dims: (lat, lon)
                aod_slice = aod_slice.expand_dims(time=[sel_in["time"].values[0]])
                aod_slice = aod_slice.expand_dims(experiment=sel_in['experiment'])
                aod_slice = aod_slice.transpose(*sel_in['AOD'].dims)
                sel_in["AOD"] = aod_slice
            else:
                raise ValueError("AOD variable not found in input variables.")

        X = self._to_tensor(sel_in, self._in_stats[ts])
        Y = self._to_tensor(sel_out, self._out_stats[ts])

        latitude = self.grid.lat_bnds.values[0].mean(axis=1)
        longitude = self.grid.lon_bnds.values[0].mean(axis=1)

        input_latitude = self.input_grid.lat.values
        input_longitude = self.input_grid.lon.values

        X_years = sel_in["time"].dt.year.values
        Y_years = sel_out["time"].dt.year.values

        meta = {
            "climate_model": self.climate_model,
            "scenario": scn,
            "member": mem,
            "latitude": latitude,
            "longitude": longitude,
            "input_latitude": input_latitude,
            "input_longitude": input_longitude,
            "spatial_layout": self.spatial_layout,
            "input_stats": self._in_stats[ts],
            "stats": self._out_stats[ts],
            # NOTE: The year for decadal is the _last_ year of the decade (e.g. 2039 for 2030-2039).
            "X_time": (X_years.min(), X_years.max()),
            "Y_time": (Y_years.min(), Y_years.max()),
        }

        return {"X": X, "Y": Y, "metadata": meta}
