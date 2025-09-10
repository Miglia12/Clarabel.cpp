#include <clarabel.hpp>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <spd_matrix_P.mtx>\n";
        return 1;
    }
    const std::string pathP = argv[1];

    // Load SPD matrix P (Matrix Market, symmetric)
    Eigen::SparseMatrix<double, Eigen::ColMajor> P;
    if (!Eigen::loadMarket(P, pathP)) {
        std::cerr << "Could not load " << pathP << "\n";
        return 1;
    }
    P = 0.5 * (P + Eigen::SparseMatrix<double, Eigen::ColMajor>(P.transpose()));
    P.makeCompressed();

    std::cout << "Loaded P: " << P.rows() << " x " << P.cols()
              << "  nnz=" << P.nonZeros() << "\n";

    // q = -P * 1  -> solution should be x* = 1
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(P.cols());
    Eigen::VectorXd q    = -P * ones;

    // No constraints / cones
    Eigen::SparseMatrix<double, Eigen::ColMajor> Acon(0, P.cols()); Acon.makeCompressed();
    Eigen::VectorXd bcon(0);
    std::vector<clarabel::SupportedConeT<double>> cones;

    // Settings
    auto settings = clarabel::DefaultSettings<double>::default_settings();
    settings.verbose = true;
    settings.direct_solve_method = clarabel::ClarabelDirectSolveMethods::PARDISO_MKL;
    settings.pardiso_verbose = false;
    settings.pardiso_iparm[1] = 0;   // try minimum degree ordering
    settings.max_threads = 8;        // or set via OMP_NUM_THREADS/MKL_NUM_THREADS

    clarabel::DefaultSolver<double> solver(P, q, Acon, bcon, cones, settings);

    auto t0 = std::chrono::high_resolution_clock::now();
    solver.solve();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    auto sol  = solver.solution();
    auto info = solver.info();

    // Checks
    Eigen::VectorXd stat = (P * sol.x + q);
    double stat_norm   = stat.norm();
    double err_to_ones = (sol.x - ones).norm();

    std::cout << "\nstatus=" << (int)sol.status
              << "  iters=" << sol.iterations
              << "  obj=" << sol.obj_val
              << "  time=" << ms << " ms\n";
    std::cout << "||P x + q|| = " << stat_norm
              << "   ||x - 1|| = " << err_to_ones << "\n";
    std::cout << "linsolver id=" << (int)info.linsolver.name
              << "  threads=" << info.linsolver.threads
              << "  KKT nnz=" << info.linsolver.nnzA
              << "  factor nnz=" << info.linsolver.nnzL << "\n";

    return 0;
}
