import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable, Sequence

from huggingface_hub import snapshot_download


def _repo_cache_name(repo_id: str, revision: str) -> str:
    return f"{repo_id.replace('/', '--')}--{revision.replace('/', '--')}"


def _default_allow_patterns(
    climate_models: Sequence[str] | None,
    timescales: Sequence[str] | None,
) -> list[str]:
    patterns = [
        "README.md",
        "manifest.json",
        "grid.nc",
    ]
    timescales = list(timescales or ["decadal", "yearly", "monthly"])
    for timescale in timescales:
        patterns.append(f"inputs/{timescale}/*.zarr.tar.zst")
    if climate_models:
        for model in climate_models:
            for timescale in timescales:
                patterns.append(f"outputs/{model}/{timescale}/*.zarr.tar.zst")
    else:
        patterns.append("outputs/**/*.zarr.tar.zst")
    return patterns


def _safe_members(tar: tarfile.TarFile, destination: Path) -> Iterable[tarfile.TarInfo]:
    destination = destination.resolve()
    for member in tar:
        target = (destination / member.name).resolve()
        if not str(target).startswith(str(destination)):
            raise RuntimeError(f"Refusing to extract archive member outside destination: {member.name}")
        yield member


def _extract_tar_zst(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    try:
        import zstandard as zstd
    except ImportError:
        if shutil.which("zstd") is None:
            raise ImportError("Install zstandard or make the zstd CLI available to extract ClimateSuite .tar.zst shards.")
        proc = subprocess.Popen(["zstd", "-dcf", str(archive_path)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                tar.extractall(destination, members=_safe_members(tar, destination))
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"Failed to decompress {archive_path}")
    else:
        with archive_path.open("rb") as compressed:
            reader = zstd.ZstdDecompressor().stream_reader(compressed)
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(destination, members=_safe_members(tar, destination))


def _link_metadata_file(src: Path, dst: Path) -> None:
    if os.path.lexists(dst):
        if not dst.is_symlink():
            return
        if dst.exists() and dst.resolve() == src.resolve():
            return
        dst.unlink()

    try:
        dst.symlink_to(src)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(src, dst)


def _extract_archives(snapshot_path: Path, extracted_root: Path) -> Path:
    archive_paths = sorted(snapshot_path.rglob("*.zarr.tar.zst"))
    if not archive_paths:
        return snapshot_path

    extracted_root.mkdir(parents=True, exist_ok=True)
    grid_src = snapshot_path / "grid.nc"
    if grid_src.exists():
        _link_metadata_file(grid_src, extracted_root / "grid.nc")

    manifest_src = snapshot_path / "manifest.json"
    if manifest_src.exists():
        _link_metadata_file(manifest_src, extracted_root / "manifest.json")

    for archive_path in archive_paths:
        relative_archive = archive_path.relative_to(snapshot_path)
        marker = extracted_root / ".extract_markers" / relative_archive.with_suffix("").with_suffix(".done")
        if marker.exists():
            continue
        shard_destination = extracted_root / relative_archive.parent
        _extract_tar_zst(archive_path, shard_destination)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"archive": str(archive_path), "size": archive_path.stat().st_size}))

    return extracted_root


def download_climate_dataset(
    repo_id: str,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    subdir: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    local_files_only: bool = False,
    max_workers: int = 16,
    climate_models: Sequence[str] | None = None,
    timescales: Sequence[str] | None = None,
    extract_archives: bool = True,
) -> Path:
    """Download a Hugging Face dataset repo and return the local dataset root."""
    if not repo_id:
        raise ValueError("repo_id must be set when data_source='huggingface'.")
    if allow_patterns is None:
        allow_patterns = _default_allow_patterns(climate_models, timescales)

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            local_files_only=local_files_only,
            max_workers=max_workers,
        )
    )
    if subdir:
        snapshot_path = snapshot_path / subdir
    if extract_archives:
        cache_base = Path(cache_dir) if cache_dir is not None else snapshot_path.parent
        extracted_root = cache_base / "extracted" / _repo_cache_name(repo_id, revision)
        return _extract_archives(snapshot_path, extracted_root)
    return snapshot_path
