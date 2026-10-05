from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "claim_data"
PAPER = ROOT / "paper.tex"
BIB = ROOT / "references.bib"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(path: Path) -> dict[str, str]:
    found = rows(path)
    if len(found) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(found)}")
    return found[0]


def close(actual: float, expected: float, tolerance: float = 5e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"value mismatch: {actual:.12g} != {expected:.12g}")


def require(text: str, fragment: str) -> None:
    if fragment not in text:
        raise RuntimeError(f"paper is missing required claim: {fragment}")


def compact_decimal(value: float, digits: int, *, upper: bool = False) -> str:
    quantum = Decimal(1).scaleb(-digits)
    rounded = Decimal(str(value)).quantize(
        quantum,
        rounding=ROUND_CEILING if upper else ROUND_HALF_UP,
    )
    rendered = format(rounded, f".{digits}f")
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def verify_bundles() -> None:
    for k in (5, 6, 7):
        directory = DATA / f"k{k}"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest_input = manifest["input"]
        if (
            manifest_input["K"] != k
            or manifest_input["group_order"] != math.factorial(k)
            or manifest_input["n_seeds"] != 10
            or len(set(manifest_input["seed_ids"])) != 10
        ):
            raise RuntimeError(f"K={k} manifest input metadata mismatch")
        expected_target = [k - 3, 2, 1]
        if (
            manifest["config"]["high_order_min"] != 3
            or manifest["config"]["allocation_targets"] != [expected_target]
        ):
            raise RuntimeError(f"K={k} manifest audit configuration mismatch")
        required_outputs = {
            "high_order_summary.csv",
            "order_spectrum.csv",
            "partition_spectrum.csv",
            "within_order_allocation.csv",
        }
        if not required_outputs.issubset(manifest["outputs"]):
            missing = sorted(required_outputs - set(manifest["outputs"]))
            raise RuntimeError(f"K={k} manifest omits required outputs: {missing}")
        for name, metadata in manifest["outputs"].items():
            path = directory / name
            if not path.is_file():
                raise RuntimeError(f"missing K={k} output: {name}")
            if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
                raise RuntimeError(f"K={k} output hash mismatch: {name}")
        diagnostics = manifest["diagnostics"]
        if not diagnostics["parseval_passed"]:
            raise RuntimeError(f"K={k} Parseval check failed")
        if diagnostics["burnside_sum_d2"] != diagnostics["burnside_expected"]:
            raise RuntimeError(f"K={k} representation dimension check failed")


def holm(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[key] = running
    return adjusted


def verify_primary(text: str) -> None:
    expected = {
        5: (0.0047303332387019165, 0.0032791687366586783, 0.006181497740745155,
            0.16139097979336303, 0.12131627823356424, 0.20146568135316184, 0.001953125),
        6: (0.004052014145105373, 0.00287217109830406, 0.005231857191906685,
            0.19426926689196036, 0.11415734650480576, 0.274381187279115, 0.001953125),
    }
    for k, target in expected.items():
        row = one(DATA / f"k{k}" / "high_order_summary.csv")
        if (
            int(row["K"]) != k
            or int(row["group_order"]) != math.factorial(k)
            or int(row["n_seeds"]) != 10
            or int(row["n_sign_patterns"]) != 512
            or row["seed_signflip_exact"] != "true"
        ):
            raise RuntimeError(f"K={k} primary-summary metadata mismatch")
        actual = tuple(float(row[name]) for name in (
            "signal_energy", "signal_ci_lower", "signal_ci_upper",
            "signal_fraction_nontrivial", "fraction_ci_lower", "fraction_ci_upper",
            "seed_signflip_p_raw",
        ))
        for value, wanted in zip(actual, target):
            close(value, wanted)
        low_signal = 1.0 - actual[3]
        low_ci = (1.0 - actual[5], 1.0 - actual[4])
        low_plugin = 1.0 - float(row["plugin_fraction_nontrivial"])
        require(text, (
            f"{k} & {compact_decimal(low_plugin, 3)} & {compact_decimal(low_signal, 3)} "
            f"[{compact_decimal(low_ci[0], 3)},{compact_decimal(low_ci[1], 3, upper=True)}]"
        ))
        require(text, (
            f"{compact_decimal(actual[0], 6)} "
            f"[{compact_decimal(actual[1], 6)},{compact_decimal(actual[2], 6, upper=True)}]"
        ))
    require(text, "$16.1\\%$ (approximate $95\\%$ CI $[12.1\\%,20.1\\%]$)")
    require(text, "$19.4\\%$ ($[11.4\\%,27.4\\%]$)")
    require(text, "$p=1/512$")
    require(text, "uniquely largest among all $512$ whole-seed sign classes")
    require(text, "central symmetry")
    require(text, "all $120$ orders")
    require(text, "all $720$ orders")
    require(text, "$5{,}040$ orders")

    rms_percentages = []
    for k in (5, 6):
        fraction = float(one(DATA / f"k{k}" / "high_order_summary.csv")["signal_fraction_nontrivial"])
        rms_percentages.append(int(Decimal(str(100.0 * math.sqrt(fraction))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
    require(text, f"${rms_percentages[0]}\\%$ and ${rms_percentages[1]}\\%$")


def verify_allocation(text: str) -> None:
    allocation = {k: one(DATA / f"k{k}" / "within_order_allocation.csv") for k in (5, 6, 7)}
    adjusted = holm({k: float(row["allocation_p_raw_two_sided"]) for k, row in allocation.items()})
    partitions = {5: "(2,2,1)", 6: "(3,2,1)", 7: "(4,2,1)"}
    for k, row in allocation.items():
        baseline = float(row["coordinate_share_within_order"])
        share = float(row["signal_target_share_ustat"])
        lower = float(row["signal_share_contrast_inversion_ci_lower"])
        upper = float(row["signal_share_contrast_inversion_ci_upper"])
        target_coordinates = int(row["target_coordinates_d2"])
        order_coordinates = int(row["order_coordinates_d2"])
        close(baseline, target_coordinates / order_coordinates, 1e-15)
        close(
            share,
            float(row["signal_target_energy_ustat"]) / float(row["signal_order_total_energy_ustat"]),
            1e-15,
        )
        if k in (5, 7):
            p_text = "$1.42\\!\\times\\!10^{-4}$"
        else:
            p_text = ".0351"
        require(text, (
            f"{k} & ${partitions[k]}$ & {compact_decimal(baseline, 4)} & "
            f"{compact_decimal(share, 4)} "
            f"[{compact_decimal(lower, 4)},{compact_decimal(upper, 4)}] & {p_text}"
        ))
    close(adjusted[5], 0.00014192387729445716)
    close(adjusted[6], 0.035146101016684445)
    close(adjusted[7], 0.00014192387729445716)


def verify_svd(text: str) -> None:
    found = {(int(row["K"]), row["partition"]): row for row in rows(DATA / "hookshape_svd_summary.csv")}
    for key, expected in {
        (5, "(2, 2, 1)"): (0.781943, 1.5608, 1.3996, 1.8052),
        (6, "(3, 2, 1)"): (0.656363, 2.1711, 2.0099, 3.2986),
    }.items():
        row = found[key]
        actual = tuple(float(row[name]) for name in (
            "top1_frac", "eff_rank", "eff_rank_boot_2_5", "eff_rank_boot_97_5"
        ))
        for value, wanted in zip(actual, expected):
            close(value, wanted, 5e-7)
    require(text, "$1.56$ (seed-resampling interval $[1.40,1.81]$)")
    require(text, "$2.17$ ($[2.01,3.30]$)")
    if "1.83" in text:
        raise RuntimeError("stale K=5 effective-rank upper endpoint 1.83")


def verify_spectra(text: str) -> None:
    spectra: dict[int, list[dict[str, str]]] = {
        k: rows(DATA / f"k{k}" / "partition_spectrum.csv") for k in (5, 6, 7)
    }
    for k, partition_rows in spectra.items():
        if sum(int(row["dimension"]) ** 2 for row in partition_rows) != math.factorial(k):
            raise RuntimeError(f"K={k} partition dimensions fail the Burnside identity")
        nontrivial_plugin = sum(
            float(row["plugin_energy"])
            for row in partition_rows
            if int(row["interaction_order"]) > 0
        )
        high_plugin = sum(
            float(row["plugin_energy"])
            for row in partition_rows
            if int(row["interaction_order"]) > 2
        )
        high_signal = sum(
            float(row["signal_energy"])
            for row in partition_rows
            if int(row["interaction_order"]) > 2
        )
        summary = one(DATA / f"k{k}" / "high_order_summary.csv")
        close(nontrivial_plugin, float(summary["all_nontrivial_plugin_energy"]), 1e-14)
        close(high_plugin, float(summary["plugin_energy"]), 1e-14)
        close(high_signal, float(summary["signal_energy"]), 1e-14)

    k5 = {row["partition"]: row for row in spectra[5]}
    expected = {
        "(4, 1)": (4, 1, 0.004644518922786935),
        "(3, 2)": (5, 2, 0.016684692621711975),
        "(3, 1, 1)": (6, 2, 0.004094124323048114),
    }
    for partition, (dimension, order, energy) in expected.items():
        row = k5[partition]
        if int(row["dimension"]) != dimension or int(row["interaction_order"]) != order:
            raise RuntimeError(f"K=5 partition metadata mismatch: {partition}")
        close(float(row["plugin_energy"]), energy)

    nontrivial_k5 = sum(
        float(row["plugin_energy"])
        for row in k5.values()
        if int(row["interaction_order"]) > 0
    )
    displayed_k5 = {
        "(4, 1)": "(4,1)",
        "(3, 2)": "(3,2)",
        "(3, 1, 1)": "(3,1,1)",
        "(2, 2, 1)": "(2,2,1)",
        "(2, 1, 1, 1)": "(2,1,1,1)",
    }
    for partition, latex_partition in displayed_k5.items():
        row = k5[partition]
        energy = float(row["plugin_energy"])
        share = energy / nontrivial_k5
        require(text, (
            f"${latex_partition}$ & {int(row['dimension'])} & {int(row['interaction_order'])} & "
            f"{compact_decimal(energy, 4)} ({compact_decimal(share, 3)})"
        ))
    sign_row = k5["(1, 1, 1, 1, 1)"]
    if not float(sign_row["plugin_energy"]) < 0.0001:
        raise RuntimeError("K=5 sign-block energy is not below the displayed threshold")
    require(text, "$(1^5)$ & 1 & 4 & $<.0001$ (.0003)")

    k6_rows = sorted(rows(DATA / "k6" / "order_spectrum.csv"), key=lambda row: int(row["interaction_order"]))
    cumulative_dimension = 0
    expected_dimensions = {0: 1, 1: 26, 2: 207, 3: 588, 4: 719, 5: 720}
    cumulative_fraction = 0.0
    for row in k6_rows:
        order = int(row["interaction_order"])
        cumulative_dimension += int(row["coordinates_d2"])
        expected_dimension = expected_dimensions[order]
        if cumulative_dimension != expected_dimension:
            raise RuntimeError(f"K=6 cumulative dimension mismatch at degree {order}")
        increment = float(row["plugin_fraction_nontrivial"])
        cumulative_fraction += increment
        close(float(row["cumulative_plugin_fraction_nontrivial"]), cumulative_fraction, 1e-12)
        require(text, (
            f"{order} & {cumulative_dimension} & {compact_decimal(cumulative_fraction, 6)} & "
            f"{compact_decimal(increment, 6)}"
        ))
    require(text, ".781254")
    require(text, ".983105")


def verify_benchmark(text: str) -> None:
    benchmark = {int(row["K"]): row for row in rows(DATA / "audit_scaling_summary.csv")}
    expected = {4: (0.017, 36.0), 5: (0.029, 36.4), 6: (0.126, 40.9), 7: (1.406, 154.9)}
    for k, (seconds, mib) in expected.items():
        close(round(float(benchmark[k]["wall_seconds_median"]), 3), seconds)
        close(round(float(benchmark[k]["peak_rss_mib_median"]), 1), mib)
        if benchmark[k]["parseval_passed_all"] != "true" or benchmark[k]["burnside_passed_all"] != "true":
            raise RuntimeError(f"K={k} benchmark validation failed")
    require(text, "$0.017$, $0.029$, $0.126$, and $1.406$ seconds")
    require(text, "$36.0$, $36.4$, $40.9$, and $154.9$ MiB")


def verify_citations(text: str) -> None:
    cited: set[str] = set()
    for match in re.finditer(r"\\cite[pt]?(?:\[[^\]]*\])?\{([^}]+)\}", text):
        cited.update(key.strip() for key in match.group(1).split(","))
    available = set(re.findall(r"^@\w+\{([^,]+),", BIB.read_text(encoding="utf-8"), flags=re.MULTILINE))
    missing = cited - available
    unused = available - cited
    if missing or unused:
        raise RuntimeError(f"bibliography closure failed; missing={sorted(missing)}, unused={sorted(unused)}")


def verify_assets() -> None:
    expected = {
        ROOT / "figures" / "figure1_population_signal.png": "28c66250e0735f96e02dcab8a575073f529ee02d1fb7e803ab26f7e81e2ef396",
        ROOT / "figures" / "figure_2_spectrum_k6.pdf": "126684e61acf19b682f71edc1b276b1ebe669c49dc25e1cccad6186591532a72",
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise RuntimeError(f"figure asset hash mismatch: {path.name}")


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")
    verify_bundles()
    verify_primary(text)
    verify_allocation(text)
    verify_svd(text)
    verify_spectra(text)
    verify_benchmark(text)
    verify_citations(text)
    verify_assets()
    public_link = r"\href{https://github.com/szymonghost/Task-order-spectral-audit}{the public repository}"
    if text.count(public_link) != 1 or text.lower().count("github.com") != 1:
        raise RuntimeError("public artifact link is missing, duplicated, or inconsistent")
    forbidden_tokens = (
        "pre" + "register",
        "pre" + "specified",
        "anonymous.4open.science",
        "one" + "drive",
        "c:" + "\\Users\\",
        "/" + "home/",
        "/" + "Users/",
        "-" * 3,
    )
    for forbidden in forbidden_tokens:
        if forbidden.lower() in text.lower():
            raise RuntimeError(f"forbidden manuscript token: {forbidden}")
    print("Verified CL4FMAgents manuscript claims, citation closure, assets, and manifest-bound CSVs.")


if __name__ == "__main__":
    main()
