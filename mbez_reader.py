#!/usr/bin/env python3
"""
Fan transformation for multibeam sonar data from .mbez files.

Converts raw polar sonar data (range/bearing) to Cartesian fan-shaped image
suitable for visualization and image processing.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import blueye.protocol as bp
from blueye.sdk.logs import LogStream
from pathlib import Path


def parse_logfile(log: Path) -> LogStream:
    """
    Read a .mbez logfile and return a LogStream generator.

    Note: This loads the entire file into memory - not suitable for very large files.
    """
    with open(log, "rb") as f:
        log_bytes = f.read()
    return LogStream(log_bytes)


def extract_pings(logfile: Path, start_index: int = 0, count: int = 1):
    """
    Extract MultibeamPingTel messages from a logfile as a generator.

    Args:
        logfile: Path to the .mbez file
        start_index: Zero-based index of the first ping to extract
        count: Number of pings to extract (None for all remaining pings)

    Yields:
        MultibeamPingTel messages

    Raises:
        ValueError: If no pings are found at the starting index
    """
    generator = parse_logfile(logfile)
    current_index = 0
    yielded_count = 0
    found_any = False

    for unix_ts, delta, msg_type, msg in generator:
        if msg_type.__name__ == "MultibeamPingTel":
            if current_index >= start_index:
                found_any = True
                yield msg
                yielded_count += 1
                if count is not None and yielded_count >= count:
                    return
            current_index += 1

    if not found_any:
        raise ValueError(
            f"No MultibeamPingTel messages found starting at index {start_index}"
        )
    elif count is not None and yielded_count < count:
        print(
            f"Warning: Only found {yielded_count} ping(s) starting at index {start_index}, "
            f"requested {count}"
        )


def create_fan_image(raw_data: np.ndarray, bearings: np.ndarray) -> np.ndarray:
    """
    Convert raw sonar data from polar to Cartesian coordinates.

    The transformation maps (range, bearing) polar coordinates to (x, y) Cartesian
    coordinates, creating a fan-shaped image where the sonar origin is at the bottom
    center of the image.

    Args:
        raw_data: 2D array of shape (num_ranges, num_beams) containing intensity values
        bearings: 1D array of bearing angles in degrees (typically -FOV/2 to +FOV/2)

    Returns:
        fan: 2D Cartesian image array of shape (HEIGHT, WIDTH)

    Coordinate system:
        - Origin at bottom center of output image
        - Y-axis points upward (increasing range)
        - X-axis points right
        - Bearings: negative=left, positive=right
    """
    bearings = np.asarray(bearings)

    # Precompute geometry constants
    min_bearing = bearings.min()
    max_bearing = bearings.max()
    fov = max_bearing - min_bearing
    height = raw_data.shape[0]
    width = int(2 * height * np.sin(np.deg2rad(fov / 2)))

    print(f"Creating fan image:")
    print(f"  FOV: {fov:.2f}° (bearings: {min_bearing:.1f}° to {max_bearing:.1f}°)")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Output dimensions: {width}x{height}")

    # Precompute pixel coordinates relative to origin
    half_width = width / 2
    x_coords = np.arange(width) - half_width
    y_coords = height - np.arange(height)

    # Create meshgrid for vectorized calculations
    X, Y = np.meshgrid(x_coords, y_coords)

    # Convert Cartesian to polar coordinates (vectorized)
    theta = np.rad2deg(np.arctan2(X, Y))  # Bearing angle for each pixel
    r = np.sqrt(X**2 + Y**2)  # Range for each pixel

    # Initialize output
    fan = np.zeros((height, width), dtype=raw_data.dtype)

    # Create mask for pixels within field of view
    fov_mask = (theta > min_bearing) & (theta < max_bearing)

    # For pixels in FOV, find closest bearing and sample from raw data
    valid_theta = theta[fov_mask]
    valid_r = r[fov_mask]

    # Find closest bearing index for each valid pixel
    # Broadcasting: (n_valid, 1) - (1, n_bearings) -> (n_valid, n_bearings)
    bearing_diffs = np.abs(valid_theta[:, np.newaxis] - bearings[np.newaxis, :])
    beam_indices = np.argmin(bearing_diffs, axis=1)
    range_indices = valid_r.astype(int)

    # Bounds checking and sampling
    valid_samples = (range_indices >= 0) & (range_indices < raw_data.shape[0])
    valid_samples &= (beam_indices >= 0) & (beam_indices < raw_data.shape[1])

    # Extract values where both indices are valid
    y_indices, x_indices = np.where(fov_mask)
    y_valid = y_indices[valid_samples]
    x_valid = x_indices[valid_samples]
    r_valid = range_indices[valid_samples]
    b_valid = beam_indices[valid_samples]

    fan[y_valid, x_valid] = raw_data[r_valid, b_valid]

    return fan


def visualize_transformation(
    raw_data: np.ndarray,
    fan_image: np.ndarray,
    bearings: np.ndarray,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Create side-by-side visualization of raw and transformed fan images.

    Args:
        raw_data: Original polar sonar data (range x beam)
        fan_image: Transformed Cartesian fan image
        bearings: Bearing angles array
        save_path: Optional path to save the figure
        show: Whether to display the interactive plot window
    """
    bearings = np.asarray(bearings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot raw data (polar coordinates: each column is a beam, each row is a range)
    im1 = axes[0].imshow(raw_data, cmap="viridis", aspect="auto")
    axes[0].set_title("Raw Sonar Data (Polar)")
    axes[0].set_xlabel("Beam Index")
    axes[0].set_ylabel("Range Index")
    plt.colorbar(im1, ax=axes[0], label="Intensity")

    # Plot fan image (Cartesian coordinates)
    im2 = axes[1].imshow(fan_image, cmap="viridis", aspect="equal", origin="upper")
    fov = bearings.max() - bearings.min()
    axes[1].set_title(f"Fan Image (Cartesian)\nFOV: {fov:.1f}°")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=axes[1], label="Intensity")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Convert multibeam sonar data from .mbez files to Cartesian fan images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .mbez file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output visualization images (created if it doesn't exist)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first ping to process",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=1,
        help="Number of pings to process (ignored if --all is specified)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all pings from start index to end of file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the interactive plot window (only save to file)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1

    # Validate arguments
    if args.start < 0:
        print(f"Error: Start index must be >= 0, got {args.start}")
        return 1
    if not args.all and args.count < 1:
        print(f"Error: Count must be >= 1, got {args.count}")
        return 1

    # Determine count (None means all)
    count = None if args.all else args.count

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Extract and process pings
    print(f"\nReading {input_path}...")
    if args.all:
        print(f"Extracting all pings starting at index {args.start}")
    else:
        print(f"Extracting {args.count} ping(s) starting at index {args.start}")
    
    ping_generator = extract_pings(input_path, start_index=args.start, count=count)

    # Process each ping
    for i, ping in enumerate(ping_generator):
        ping_index = args.start + i
        print(f"\n{'-' * 70}")
        print(f"Processing ping #{ping_index} ({i+1})")
        print(f"{'-' * 70}")

        # Display ping metadata
        print(f"Ping data:")
        print(f"  Range count: {ping.ping.number_of_ranges}")
        print(f"  Beam count: {ping.ping.number_of_beams}")
        print(f"  Bearings: {len(ping.ping.bearings)} angles")
        print(
            f"  Bearing range: [{min(ping.ping.bearings):.2f}°, "
            f"{max(ping.ping.bearings):.2f}°]"
        )

        # Convert raw bytes to 2D numpy array
        raw_data = np.frombuffer(ping.ping.ping_data, dtype=np.uint8).reshape(
            ping.ping.number_of_ranges, ping.ping.number_of_beams
        )

        # Perform fan transformation
        fan_image = create_fan_image(raw_data, ping.ping.bearings)

        print(f"\nFan image created successfully!")
        print(f"  Shape: {fan_image.shape}")
        print(f"  Value range: [{fan_image.min()}, {fan_image.max()}]")
        print(f"  Non-zero pixels: {np.count_nonzero(fan_image):,}")

        # Generate output filename
        output_filename = f"ping_{ping_index:04d}_fan.png"
        output_path = output_dir / output_filename

        # Visualize and save
        print(f"\nSaving to {output_path}...")
        # Only show visualization for the last ping (when count is 1 or processing single ping)
        # For --all mode, never show (too many pings)
        show_viz = not args.no_show and not args.all and count == 1

        visualize_transformation(
            raw_data,
            fan_image,
            ping.ping.bearings,
            save_path=str(output_path),
            show=show_viz,
        )

    print(f"\n{'-' * 70}")
    print(f"Completed processing!")
    print(f"Output saved to: {output_dir.absolute()}")
    print(f"{'-' * 70}")

    return 0


if __name__ == "__main__":
    exit(main())
