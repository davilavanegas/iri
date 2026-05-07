"""IRI calculation using the standard quarter-car model.

The core calculation has no third-party dependency. Plot output uses
matplotlib when the -plot_file option is provided.
"""

import argparse
import os
import re
from bisect import bisect_right
from pathlib import Path


STATION_ALIASES = {"sta", "station", "stationing", "chainage", "distance", "dist", "x"}
ELEVATION_ALIASES = {"elv", "elev", "elevation", "height", "z", "y"}


EXAMPLE_PROFILE = [
    (0.00, 0),
    (0.25, 2),
    (0.50, 5),
    (0.75, 3),
    (1.00, 6),
    (1.25, 4),
    (1.50, 1),
    (1.75, 3),
    (2.00, 0),
]

def interp_road(x, profile):
    """Linear interpolation of road elevation r(x)."""
    if x <= profile[0][0]:
        return profile[0][1]
    if x >= profile[-1][0]:
        return profile[-1][1]

    i = bisect_right(profile, (x, float("inf"))) - 1
    x0, z0 = profile[i]
    x1, z1 = profile[i + 1]
    t = (x - x0) / (x1 - x0)
    return z0 + t * (z1 - z0)


def quarter_car_derivatives(state, x, profile, speed, k1, k2, c, mu):
    """
    State vector:
    state[0] = zs   sprung mass displacement
    state[1] = zu   unsprung mass displacement
    state[2] = zsd  sprung mass velocity
    state[3] = zud  unsprung mass velocity
    """

    zs, zu, zsd, zud = state
    r = interp_road(x, profile)

    zsdd = -c * (zsd - zud) - k2 * (zs - zu)

    zudd = (
        c * (zsd - zud)
        + k2 * (zs - zu)
        - k1 * (zu - r)
    ) / mu

    return [zsd, zud, zsdd, zudd]


def rk4_step(state, x, dx, profile, speed, k1, k2, c, mu):
    """One Runge-Kutta 4 integration step."""

    dt = dx / speed

    def f(s, xpos):
        return quarter_car_derivatives(
            s, xpos, profile, speed, k1, k2, c, mu
        )

    k_1 = f(state, x)

    s2 = [state[i] + 0.5 * dt * k_1[i] for i in range(4)]
    k_2 = f(s2, x + 0.5 * dx)

    s3 = [state[i] + 0.5 * dt * k_2[i] for i in range(4)]
    k_3 = f(s3, x + 0.5 * dx)

    s4 = [state[i] + dt * k_3[i] for i in range(4)]
    k_4 = f(s4, x + dx)

    new_state = [
        state[i] + (dt / 6.0) * (
            k_1[i] + 2.0 * k_2[i] + 2.0 * k_3[i] + k_4[i]
        )
        for i in range(4)
    ]

    return new_state


def integrate_profile(profile_mm, dx=0.01):
    """
    Integrate the quarter-car model over the full profile.

    Returns cumulative suspension motion as a list of
    (station_m, cumulative_motion_m).
    """

    # Convert elevation from mm to m
    profile = [(x, z / 1000.0) for x, z in profile_mm]

    # Standard IRI quarter-car parameters
    speed = 80.0 / 3.6   # 80 km/h in m/s
    k1 = 653.0           # tire stiffness
    k2 = 63.3            # suspension stiffness
    c = 6.0              # damping
    mu = 0.15            # mass ratio

    start_x = profile[0][0]
    end_x = profile[-1][0]

    # Initial state
    initial_road = profile[0][1]
    state = [
        initial_road,  # zs
        initial_road,  # zu
        0.0,           # zsd
        0.0            # zud
    ]

    total_suspension_motion = 0.0
    cumulative_motion = [(start_x, total_suspension_motion)]

    x = start_x

    while x < end_x:
        step = min(dx, end_x - x)

        old_relative_velocity = abs(state[2] - state[3])

        new_state = rk4_step(
            state, x, step, profile, speed, k1, k2, c, mu
        )

        new_relative_velocity = abs(new_state[2] - new_state[3])

        dt = step / speed

        # Trapezoidal integration of suspension velocity
        total_suspension_motion += (
            0.5 * (old_relative_velocity + new_relative_velocity) * dt
        )

        state = new_state
        x += step
        cumulative_motion.append((x, total_suspension_motion))

    return cumulative_motion


def calculate_iri(profile_mm):
    """
    profile_mm should be a list of:
    (distance_m, elevation_mm)

    Returns IRI in m/km.
    """

    road_length = profile_mm[-1][0] - profile_mm[0][0]

    if road_length <= 0:
        raise ValueError("Profile road length must be greater than zero.")

    cumulative_motion = integrate_profile(profile_mm)
    iri_m_per_m = cumulative_motion[-1][1] / road_length
    iri_m_per_km = iri_m_per_m * 1000.0

    return iri_m_per_km


def calculate_iri_segments(profile_mm, segment_length=20.0, step=0.0):
    """
    Calculate IRI values for fixed station segments.

    Returns a list of (start_station, end_station, iri_m_per_km).
    """
    if segment_length <= 0:
        raise ValueError("Segment length must be greater than zero.")
    if step < 0:
        raise ValueError("Step must be zero or greater.")

    start_x = profile_mm[0][0]
    end_x = profile_mm[-1][0]
    road_length = end_x - start_x

    if road_length <= 0:
        raise ValueError("Profile road length must be greater than zero.")

    cumulative_motion = integrate_profile(profile_mm)

    if segment_length >= road_length:
        iri = cumulative_motion[-1][1] / road_length * 1000.0
        return [(start_x, end_x, iri)]

    segment_step = step if step > 0 else segment_length
    segments = []
    x = start_x

    while x + segment_length <= end_x + 1e-9:
        segment_end = x + segment_length
        segment_motion = interp_road(segment_end, cumulative_motion) - interp_road(x, cumulative_motion)
        iri = segment_motion / segment_length * 1000.0
        segments.append((x, segment_end, iri))
        x += segment_step

    return segments


def normalize_header(value):
    """Normalize a column name for matching STA/ELV style headers."""
    return re.sub(r"[^a-z0-9]", "", value.strip().strip('"').strip("'").lower())


def split_row(line, delimiter):
    if delimiter is None:
        return [value for value in re.split(r"[\s,;]+", line.strip()) if value]
    return [value.strip() for value in line.strip().split(delimiter)]


def parse_number(value):
    text = value.strip().strip('"').strip("'")

    try:
        return float(text)
    except ValueError:
        pass

    if "+" in text:
        major, minor = text.split("+", 1)
        return float(major) * 1000.0 + float(minor)

    raise ValueError(f"could not parse number: {value!r}")


def looks_numeric_pair(values):
    if len(values) < 2:
        return False

    try:
        parse_number(values[0])
        parse_number(values[1])
    except ValueError:
        return False

    return True


def find_alias_column(headers, aliases):
    normalized_headers = [normalize_header(header) for header in headers]
    for index, header in enumerate(normalized_headers):
        if header in aliases:
            return index
    return None


def resolve_column(column, headers, aliases, default_index, label):
    if column is None:
        if headers:
            alias_index = find_alias_column(headers, aliases)
            if alias_index is not None:
                return alias_index
        return default_index

    try:
        index = int(column) - 1
    except ValueError:
        index = None

    if index is not None:
        if index < 0:
            raise ValueError(f"{label} column numbers are 1-based.")
        return index

    if not headers:
        raise ValueError(f"{label} column name {column!r} was provided, but no header row was found.")

    wanted = normalize_header(column)
    normalized_headers = [normalize_header(header) for header in headers]

    if wanted not in normalized_headers:
        raise ValueError(f"{label} column {column!r} was not found in the header.")

    return normalized_headers.index(wanted)


def read_profile_txt(
    input_file,
    delimiter=None,
    skip_header=0,
    station_column=None,
    elevation_column=None,
    elevation_unit="mm",
):
    """
    Read a text file containing station/elevation columns.

    By default the reader accepts whitespace, comma, or semicolon separated
    rows. If a header contains STA/STATION and ELV/ELEVATION columns, those
    columns are used; otherwise the first two columns are used.
    """
    rows = []

    with open(input_file, "r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].strip()

            if not line:
                continue

            if skip_header > 0:
                skip_header -= 1
                continue

            values = split_row(line, delimiter)
            if values:
                rows.append((line_number, values))

    if not rows:
        raise ValueError(f"No profile rows found in {input_file!r}.")

    first_line_number, first_values = rows[0]
    has_header = not looks_numeric_pair(first_values)
    headers = first_values if has_header else None
    data_rows = rows[1:] if has_header else rows

    station_index = resolve_column(station_column, headers, STATION_ALIASES, 0, "Station")
    elevation_index = resolve_column(elevation_column, headers, ELEVATION_ALIASES, 1, "Elevation")

    required_columns = max(station_index, elevation_index) + 1
    profile_mm = []

    for line_number, values in data_rows:
        if len(values) < required_columns:
            raise ValueError(
                f"Line {line_number} has {len(values)} columns, but column {required_columns} is required."
            )

        try:
            station_m = parse_number(values[station_index])
            elevation = parse_number(values[elevation_index])
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: {exc}") from exc

        if elevation_unit == "m":
            elevation *= 1000.0

        profile_mm.append((station_m, elevation))

    if has_header and not data_rows:
        raise ValueError(f"Header was found on line {first_line_number}, but no profile data followed it.")

    profile_mm.sort(key=lambda row: row[0])
    validate_profile(profile_mm)

    return profile_mm


def validate_profile(profile_mm):
    if len(profile_mm) < 2:
        raise ValueError("At least two station/elevation rows are required.")

    for index in range(len(profile_mm) - 1):
        if profile_mm[index + 1][0] <= profile_mm[index][0]:
            raise ValueError("Station values must be unique after sorting.")


def ensure_output_folder(file_path):
    output_folder = os.path.dirname(file_path)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)


def normalize_plot_file(plot_file):
    output_path = Path(plot_file)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".jpg")
    return str(output_path)


def get_pyplot():
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_config_dir

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with: pip install matplotlib") from exc

    return plt


def plot_profile(profile_mm, iri, plot_file, elevation_unit="mm", iri_segments=None):
    plot_file = normalize_plot_file(plot_file)
    ensure_output_folder(plot_file)

    stations = [point[0] for point in profile_mm]
    if elevation_unit == "m":
        elevations = [point[1] / 1000.0 for point in profile_mm]
        elevation_label = "Elevation [m]"
    else:
        elevations = [point[1] for point in profile_mm]
        elevation_label = "Elevation [mm]"

    plt = get_pyplot()
    if iri_segments:
        fig, (profile_ax, iri_ax) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig, profile_ax = plt.subplots(figsize=(12, 6))
        iri_ax = None

    profile_ax.plot(stations, elevations, color="tab:blue", linewidth=1.6)
    profile_ax.set_ylabel(elevation_label)
    profile_ax.set_title(f"Road Profile - IRI {iri:.3f} m/km")
    profile_ax.grid(True, alpha=0.3)

    if iri_ax is not None:
        segment_centers = [(start + end) / 2.0 for start, end, _ in iri_segments]
        iri_values = [segment_iri for _, _, segment_iri in iri_segments]
        iri_ax.plot(segment_centers, iri_values, color="tab:orange", marker="o", linewidth=1.6)
        iri_ax.axhline(iri, color="black", linestyle="--", linewidth=1.2, label=f"Overall IRI {iri:.3f} m/km")
        iri_ax.set_ylabel("IRI [m/km]")
        iri_ax.set_xlabel("Station [m]")
        iri_ax.grid(True, alpha=0.3)
        iri_ax.legend()
    else:
        profile_ax.set_xlabel("Station [m]")

    fig.tight_layout()
    fig.savefig(plot_file, dpi=160)
    plt.close(fig)

    return plot_file


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate IRI from a built-in example or a station/elevation text file."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Text file with station and elevation columns. If omitted, the built-in example is used.",
    )
    parser.add_argument(
        "-delimiter",
        "--delimiter",
        default=None,
        help="Input delimiter. Omit to accept whitespace, comma, or semicolon separated rows.",
    )
    parser.add_argument(
        "-skip_header",
        "--skip-header",
        type=int,
        default=0,
        help="Number of non-empty header rows to skip before reading the profile.",
    )
    parser.add_argument(
        "-station_column",
        "--station-column",
        default=None,
        help="Station column name or 1-based column number. Defaults to STA/STATION or column 1.",
    )
    parser.add_argument(
        "-elevation_column",
        "--elevation-column",
        default=None,
        help="Elevation column name or 1-based column number. Defaults to ELV/ELEVATION or column 2.",
    )
    parser.add_argument(
        "-elevation_unit",
        "--elevation-unit",
        choices=("mm", "m"),
        default="mm",
        help="Elevation unit in the input file. Defaults to mm.",
    )
    parser.add_argument(
        "-segment_length",
        "--segment-length",
        type=float,
        default=20.0,
        help="Segment length used for the IRI value plot. Defaults to 20 m.",
    )
    parser.add_argument(
        "-step",
        "--step",
        type=float,
        default=0.0,
        help="Station shift between IRI plot segments. Defaults to segment length.",
    )
    parser.add_argument(
        "-plot_file",
        "--plot-file",
        help="Output graph file, for example output/profile.jpg. If no extension is given, .jpg is used.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.input_file:
        profile = read_profile_txt(
            args.input_file,
            delimiter=args.delimiter,
            skip_header=args.skip_header,
            station_column=args.station_column,
            elevation_column=args.elevation_column,
            elevation_unit=args.elevation_unit,
        )
        source = args.input_file
        plot_elevation_unit = args.elevation_unit
    else:
        profile = EXAMPLE_PROFILE
        source = "built-in example"
        plot_elevation_unit = "mm"

    iri = calculate_iri(profile)

    print(f"Profile source: {source}")
    print(f"Profile points: {len(profile)}")
    print("IRI =", round(iri, 3), "m/km")

    if args.plot_file:
        iri_segments = calculate_iri_segments(profile, args.segment_length, args.step)
        print(f"IRI plot segments: {len(iri_segments)}")
        saved_plot = plot_profile(profile, iri, args.plot_file, plot_elevation_unit, iri_segments)
        print(f"Plot saved to {saved_plot}")


if __name__ == "__main__":
    main()
