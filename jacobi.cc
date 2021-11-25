#include <torch/extension.h>

using Vector = torch::PackedTensorAccessor<double, 1>;
using Matrix = torch::PackedTensorAccessor<double, 2>;

inline const auto EPS = std::numeric_limits<double>::epsilon();
inline const auto MAX_ITER = 50;
inline const auto TOLERANCE = 1e-7;

inline torch::Tensor clone_check_matrix(torch::Tensor A)
{

    TORCH_CHECK(A.dim() == 2, "matrix is required");
    TORCH_CHECK(A.dtype() == torch::kFloat64, "double matrix is required");
    return A.clone();
}

inline void rot(Matrix &a, const double s, const double tau, const int64_t i,
                const int64_t j, const int64_t k, const int64_t l)
{
    double g = a[i][j];
    double h = a[k][l];
    a[i][j] = g - s * (h + g * tau);
    a[k][l] = h + s * (g - h * tau);
}

inline double max_abs_offdiagonal(Matrix a, int64_t n)
{
    double sm = 0.0;
    for (int64_t ip = 0; ip < n - 1; ip++)
        for (int64_t iq = ip + 1; iq < n; iq++)
            sm = std::max(abs(a[ip][iq]), sm);
    return sm;
}

inline void jacobi_iteration(
    Matrix a,
    Matrix v,
    Vector d,
    Vector z,
    int64_t n)
{
    for (int64_t ip = 0; ip < n - 1; ip++)
    {
        for (int64_t iq = ip + 1; iq < n; iq++)
        {
            const auto g = 100.0 * abs(a[ip][iq]);

            if (g <= EPS * abs(d[ip]) && g <= EPS * abs(d[iq]))
                a[ip][iq] = 0.0;
            else
            {
                auto h = d[iq] - d[ip];
                auto t = 0.0;
                if (g <= EPS * abs(h))
                    t = (a[ip][iq]) / h;
                else
                {
                    const auto theta = 0.5 * h / (a[ip][iq]);
                    t = 1.0 / (abs(theta) + sqrt(1.0 + theta * theta));
                    if (theta < 0.0)
                        t = -t;
                }
                const auto c = 1.0 / sqrt(1 + t * t);
                const auto s = t * c;
                const auto tau = s / (1.0 + c);
                h = t * a[ip][iq];
                z[ip] -= h;
                z[iq] += h;
                d[ip] -= h;
                d[iq] += h;
                a[ip][iq] = 0.0;
                for (int64_t j = 0; j < ip; j++)
                    rot(a, s, tau, j, ip, j, iq);
                for (int64_t j = ip + 1; j < iq; j++)
                    rot(a, s, tau, ip, j, j, iq);
                for (int64_t j = iq + 1; j < n; j++)
                    rot(a, s, tau, ip, j, iq, j);
                for (int64_t j = 0; j < n; j++)
                    rot(v, s, tau, j, ip, j, iq);
            }
        }
    }
}

inline void update_diagonal(
    Vector d,
    Vector z,
    Vector b,
    int64_t n)
{
    for (int64_t ip = 0; ip < n; ip++)
    {
        b[ip] += z[ip];
        d[ip] = b[ip];
        z[ip] = 0.0;
    }
}

std::tuple<torch::Tensor, torch::Tensor> symeig(
    torch::Tensor A)
{

    auto A_ = clone_check_matrix(A);

    const auto D = torch::diagonal(A_).contiguous();
    const auto n = D.numel();
    const auto V = torch::eye(n, D.options());

    const auto B = D.clone();
    const auto Z = torch::zeros_like(D);

    auto a = A_.packed_accessor<double, 2>();
    auto v = V.packed_accessor<double, 2>();
    auto d = D.packed_accessor<double, 1>();
    auto b = B.packed_accessor<double, 1>();
    auto z = Z.packed_accessor<double, 1>();

    auto sm = max_abs_offdiagonal(a, n);
    auto iter = 0;

    while (iter < MAX_ITER)
    {

        if (sm < TOLERANCE)
            break;
        else
        {
            jacobi_iteration(a,v,d,z,n);
            update_diagonal(d,z,b,n);
            sm = max_abs_offdiagonal(a, n);
            iter++;
        }
    }

    TORCH_CHECK(sm < TOLERANCE, "Jacobi algorithm did not converge");

    return std::make_tuple(D, V);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("symeig", &symeig,
          "Computes the eigenvalues and eigenvectors of a symmetric matrix via Jacobi rotations");
}