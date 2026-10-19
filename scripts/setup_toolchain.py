"""Portable, no-admin toolchain installer for the S02 case study (Windows x64).

Downloads OFFICIAL packages into ./tools (nothing is installed system-wide):
  * NVIDIA CUDA 12.9.1 redistributable archives (nvcc, cudart, cccl, profiler_api)
    and Nsight Compute 2025.2.1   -- https://developer.download.nvidia.com/compute/cuda/redist/
  * MSVC 14.44 x64 compiler + CRT and Windows SDK 10.0.26100 headers/libs, from the
    official Visual Studio 2022 channel manifest -- https://aka.ms/vs/17/release/channel
  * pandoc (portable zip) -- https://github.com/jgm/pandoc/releases
Every download is SHA-256 verified against the vendor manifest where one is published.
Running this script implies acceptance of the NVIDIA CUDA EULA and the Microsoft
Visual Studio Build Tools / Windows SDK licence terms.

Usage:  python scripts/setup_toolchain.py      (then:  . scripts\\env.ps1)
"""
import hashlib, json, re, shutil, subprocess, sys, urllib.parse, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
DL = TOOLS / "_downloads"

CUDA_REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDA_VER = "12.9.1"
CUDA_COMPONENTS = ["cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_profiler_api"]
MSVC_VER = "14.44.17.14"
SDK_PKG = "win11sdk_10.0.26100"
SDK_MSIS = [
    "Windows SDK for Windows Store Apps Tools-x86_en-us.msi",
    "Windows SDK for Windows Store Apps Headers-x86_en-us.msi",
    "Windows SDK for Windows Store Apps Headers OnecoreUap-x86_en-us.msi",
    "Windows SDK for Windows Store Apps Libs-x86_en-us.msi",
    "Universal CRT Headers Libraries and Sources-x86_en-us.msi",
    "Windows SDK Desktop Headers x64-x86_en-us.msi",
    "Windows SDK Desktop Headers x86-x86_en-us.msi",
    "Windows SDK Desktop Libs x64-x86_en-us.msi",
]
UA = {"User-Agent": "s02-setup/1.0"}


def fetch(url, dest, sha256=None):
    dest = Path(dest)
    if dest.exists() and (sha256 is None or sha(dest) == sha256.lower()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name}", flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if sha256 and sha(dest) != sha256.lower():
        dest.unlink()
        sys.exit(f"SHA-256 mismatch for {url}")
    return dest


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unzip_strip(zp, out, strip_prefix_re):
    """Extract only entries under a leading prefix (regex), with that prefix removed."""
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            name = urllib.parse.unquote(m.filename.replace("\\", "/"))
            mt = re.match(strip_prefix_re, name)
            if not mt or not name[mt.end():]:
                continue
            target = out / name[mt.end():]
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as s, open(target, "wb") as d:
                shutil.copyfileobj(s, d)


def install_cuda():
    print("[CUDA] reading redist manifest")
    man = json.loads(urllib.request.urlopen(f"{CUDA_REDIST}redistrib_{CUDA_VER}.json").read())
    for comp, out in [(c, TOOLS / "cuda") for c in CUDA_COMPONENTS] + [("nsight_compute", TOOLS / "nsight-compute")]:
        w = man[comp]["windows-x86_64"]
        zp = fetch(CUDA_REDIST + w["relative_path"], DL / Path(w["relative_path"]).name, w["sha256"])
        print(f"  extracting {comp} -> {out.relative_to(ROOT)}")
        unzip_strip(zp, out, r"^[^/]+-archive/")


def install_msvc():
    print("[MSVC] reading VS 2022 channel manifest")
    ch = json.loads(urllib.request.urlopen("https://aka.ms/vs/17/release/channel").read().decode("utf-8-sig"))
    vsman_url = next(i for i in ch["channelItems"] if i["id"] == "Microsoft.VisualStudio.Manifests.VisualStudio")["payloads"][0]["url"]
    vsman = json.loads(urllib.request.urlopen(vsman_url).read().decode("utf-8-sig"))
    pk = {}
    for p in vsman["packages"]:
        pk.setdefault(p["id"].lower(), []).append(p)
    out = TOOLS / "msvc"
    for suffix in ["Tools.HostX64.TargetX64.base", "Tools.HostX64.TargetX64.Res.base",
                   "CRT.Headers.base", "CRT.x64.Desktop.base", "CRT.x64.Store.base"]:
        pid = f"Microsoft.VC.{MSVC_VER}.{suffix}".lower()
        p = [q for q in pk[pid] if q.get("language") in (None, "en-US")][0]
        for pl in p["payloads"]:
            zp = fetch(pl["url"], DL / "msvc" / f"{pid}.vsix", pl.get("sha256"))
            print(f"  extracting {pid}")
            unzip_strip(zp, out, r"^Contents/")
    # Windows SDK: MSIs + the .cab files they reference, then msiexec administrative extract
    sdk = pk[SDK_PKG][0]
    payloads = {pl["fileName"].split("\\")[-1]: pl for pl in sdk["payloads"]}
    sdkdl = DL / "sdk"
    cabs = set()
    for msi in SDK_MSIS:
        pl = payloads[msi]
        mp = fetch(pl["url"], sdkdl / msi, pl.get("sha256"))
        cabs |= set(re.findall(rb"[0-9a-f]{32}\.cab", mp.read_bytes()))
    for c in sorted(cabs):
        c = c.decode()
        pl = payloads.get(c)
        if pl:
            fetch(pl["url"], sdkdl / c, pl.get("sha256"))
    for msi in SDK_MSIS:
        print(f"  msiexec /a {msi}")
        cmd = f'msiexec.exe /a "{sdkdl / msi}" /qn TARGETDIR="{out}"'
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"msiexec failed ({r.returncode}) on {msi}")
    for f in out.glob("*.msi"):
        f.unlink()
    # nvcc refuses a host compiler ("Host compiler targets unsupported OS") unless a VS-style
    # VC\Auxiliary\Build\vcvarsall.bat / vcvars64.bat exist; provide stand-ins that call env.bat.
    b = out / "VC" / "Auxiliary" / "Build"
    b.mkdir(parents=True, exist_ok=True)
    (b / "vcvars64.bat").write_text(
        "@rem Portable stand-in for VS vcvars64.bat (nvcc requires it). Sets PATH/INCLUDE/LIB for tools\\msvc.\n"
        '@call "%~dp0..\\..\\..\\..\\..\\scripts\\env.bat" >nul\n')
    (b / "vcvarsall.bat").write_text(
        "@rem Portable stand-in for VS vcvarsall.bat (nvcc checks for it).\n"
        '@call "%~dp0vcvars64.bat"\n')


def install_pandoc():
    print("[pandoc] resolving latest release")
    req = urllib.request.Request("https://github.com/jgm/pandoc/releases/latest", headers=UA)
    tag = urllib.request.urlopen(req).geturl().rstrip("/").split("/")[-1]
    url = f"https://github.com/jgm/pandoc/releases/download/{tag}/pandoc-{tag}-windows-x86_64.zip"
    zp = fetch(url, DL / f"pandoc-{tag}-windows-x86_64.zip")
    unzip_strip(zp, TOOLS / "pandoc", r"^pandoc-[^/]+/")


if __name__ == "__main__":
    which = sys.argv[1:] or ["cuda", "msvc", "pandoc"]
    if "cuda" in which: install_cuda()
    if "msvc" in which: install_msvc()
    if "pandoc" in which: install_pandoc()
    print("done. Activate with:  . scripts\\env.ps1   (PowerShell)  or  scripts\\env.bat  (cmd)")
