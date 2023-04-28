from __future__ import annotations

import nickel_111


def main() -> None:
    nickel_111.s4_wavepacket.generate_nickel_wavepacket_sho()
    nickel_111.s5_overlap.calculate_overlap_nickel()
    nickel_111.s5_overlap_analysis.print_max_overlap_transforms()


if __name__ == "__main__":
    main()
