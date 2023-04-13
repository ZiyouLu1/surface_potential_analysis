import nickel_111


def main() -> None:
    nickel_111.s3_eigenstates.generate_eigenstates_data()
    nickel_111.s3_eigenstates_plot.analyze_band_convergence()


if __name__ == "__main__":
    main()
