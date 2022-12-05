use std::{collections::HashMap, f64::consts::PI};

use pyo3::prelude::*;

fn factorial(n: u32) -> u32 {
    (1..=n).product()
}
#[must_use]
const fn hermite_coefficient(n: i64, m: i64) -> i64 {
    match (n, m) {
        (0, 0) => 1,
        (1, 0) => 0,
        (1, 1) => 2,
        (_, 0) => -hermite_coefficient(n - 1, m + 1),
        (n, m) if n >= m => {
            2 * hermite_coefficient(n - 1, m - 1) - (m + 1) * hermite_coefficient(n - 1, m + 1)
        }
        _ => 0,
    }
}

fn hermite_val(x: f64, n: u32) -> f64 {
    return (0..=n)
        .map(|m| (hermite_coefficient(n.into(), m.into()) as f64) * x.powi(m.try_into().unwrap()))
        .sum();
}

fn calculate_sho_wavefunction(z_points: Vec<f64>, sho_omega: f64, mass: f64, n: u32) -> Vec<f64> {
    let norm = ((sho_omega * mass) / REDUCED_PLANCK_CONSTANT).sqrt();
    let prefactor = (norm / (((2_u32.pow(n) * factorial(n)) as f64) * PI.sqrt())).sqrt();
    return z_points
        .into_iter()
        .map(|p| -> f64 {
            let normalized_p = p * norm;
            let hermite_val = hermite_val(normalized_p, n);
            prefactor * hermite_val * f64::exp(-normalized_p.powi(2) / 2.0)
        })
        .collect();
}

const PLANCK_CONSTANT: f64 = 6.62607015E-34;
const REDUCED_PLANCK_CONSTANT: f64 = PLANCK_CONSTANT / (2.0 * PI);

struct SHOConfig {
    mass: f64,
    sho_omega: f64,
    z_offset: f64,
}
struct SurfaceHamiltonian {
    resolution: [usize; 3],
    sho_config: SHOConfig,
    ft_potential: Vec<Vec<Vec<f64>>>,
    dz: f64,
}

impl SurfaceHamiltonian {
    fn coordinates(&self) -> Vec<(i64, i64, usize)> {
        (0..self.resolution[0])
            .flat_map(|x| {
                (0..self.resolution[1]).flat_map(move |y| {
                    (0..self.resolution[2]).map(move |z| (x as i64, y as i64, z))
                })
            })
            .collect()
    }

    fn get_nx(&self) -> usize {
        return self.ft_potential.len();
    }

    fn get_ny(&self) -> usize {
        return self.ft_potential[0].len();
    }

    fn get_nz(&self) -> usize {
        return self.ft_potential[0][0].len();
    }

    fn get_z_points(&self) -> Vec<f64> {
        (0..self.get_nz())
            .map(|i| self.dz * (i as f64) + self.sho_config.z_offset)
            .collect()
    }

    fn calculate_sho_wavefunction(&self, n: u32) -> Vec<f64> {
        calculate_sho_wavefunction(
            self.get_z_points(),
            self.sho_config.sho_omega,
            self.sho_config.mass,
            n,
        )
    }

    fn calculate_off_diagonal_energies(&self) -> Vec<Vec<f64>> {
        let coordinates = self.coordinates();
        let cache: Vec<Vec<f64>> = (0..self.resolution[2])
            .map(|nz| -> Vec<f64> { self.calculate_sho_wavefunction(nz.try_into().unwrap()) })
            .collect();

        let mut g_points: HashMap<(usize, usize, usize, usize), f64> = HashMap::new();

        coordinates
            .iter()
            .map(|(nkx1, nky1, nz1)| -> Vec<f64> {
                coordinates
                    .iter()
                    .map(|(nkx2, nky2, nz2)| -> f64 {
                        let ndkx = (nkx2 - nkx1).rem_euclid(self.get_nx() as i64) as usize;
                        let ndky = (nky2 - nky1).rem_euclid(self.get_ny() as i64) as usize;
                        if let Some(a) = g_points.get(&(ndkx, ndky, *nz1, *nz2)) {
                            return *a;
                        }

                        let ft_pot_points = &self.ft_potential[ndkx][ndky];

                        let sho1: &Vec<f64> = &cache[*nz1];
                        let sho2: &Vec<f64> = &cache[*nz2];

                        let out = ft_pot_points
                            .iter()
                            .zip(sho1)
                            .zip(sho2)
                            .map(|((i, j), k)| i * j * k)
                            .sum::<f64>()
                            * self.dz;
                        g_points.insert((ndkx, ndky, *nz1, *nz2), out);
                        return out;
                    })
                    .collect()
            })
            .collect()
    }
}

#[pyfunction]
fn get_hermite_val(x: f64, n: u32) -> PyResult<f64> {
    Ok(hermite_val(x, n))
}

#[pyfunction]
fn get_sho_wavefunction(
    z_points: Vec<f64>,
    sho_omega: f64,
    mass: f64,
    n: u32,
) -> PyResult<Vec<f64>> {
    Ok(calculate_sho_wavefunction(z_points, sho_omega, mass, n))
}

#[pyfunction]
fn get_hamiltonian(
    ft_potential: Vec<Vec<Vec<f64>>>,
    resolution: [usize; 3],
    dz: f64,
    mass: f64,
    sho_omega: f64,
    z_offset: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let sho_config = SHOConfig {
        mass,
        sho_omega,
        z_offset,
    };
    let hamiltonian = SurfaceHamiltonian {
        dz,
        ft_potential,
        resolution,
        sho_config,
    };
    Ok(hamiltonian.calculate_off_diagonal_energies())
}

/// A Python module implemented in Rust.
#[pymodule]
fn hamiltonian_diag(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_hamiltonian, m)?)?;
    m.add_function(wrap_pyfunction!(get_sho_wavefunction, m)?)?;
    m.add_function(wrap_pyfunction!(get_hermite_val, m)?)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{SHOConfig, SurfaceHamiltonian};

    #[test]
    fn test_calculate_off_diagonal_energies() {
        let sho_config = SHOConfig {
            mass: 1.0,
            sho_omega: 1.0,
            z_offset: 0.0,
        };
        let hamiltonian = SurfaceHamiltonian {
            dz: 1.0,
            ft_potential: vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            ],
            resolution: [2, 2, 2],
            sho_config,
        };

        hamiltonian.calculate_off_diagonal_energies();
    }
}
