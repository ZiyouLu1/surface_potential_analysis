use num_complex::{Complex, Complex64};

trait Basis {
    const N: usize;
}

trait IntoBasis<T> {
    fn into_basis(&self, basis: &T, vector: &[Complex64]) -> Vec<Complex64>;
}

struct FundamentalBasis(usize);

impl Basis for FundamentalBasis {
    const N: usize = self[0];
    // fn n(&self) -> usize {
    //     N
    // }
}

trait ProductBasis {
    fn shape(&self) -> Vec<usize>;
}
