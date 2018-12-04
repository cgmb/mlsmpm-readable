// 2D Moving Least Squares Material Point Method (MLS-MPM) [with comments]
#define TC_IMAGE_IO
#include "taichi.h"
using namespace taichi;
using Vec2 = Vector2;
using Mat2 = Matrix2;

/* grid_resolution (cells) */
const int n = 80,
      window_size = 800;
const real dt = 1e-4_f,
      frame_dt = 1e-3_f,
      dx = 1.0_f / n,
      inv_dx = 1.0_f / dx;
auto particle_mass = 1.0_f,
     vol = 1.0_f;
auto hardening = 10.0_f,
     E = 1e4_f,
     nu = 0.2_f;
real mu_0 = E / (2 * (1 + nu)),
     lambda_0 = E * nu / ((1+nu) * (1 - 2 * nu));

bool plastic = true;

struct Particle {
    Vec2 x,  // position
         v;  // velocity
    Mat2 F,  // deformation gradient
         C;  // velocity field
    real Jp; // Jacobian determinant
    int c;   // color

    Particle(Vec2 x, int c, Vec2 v=Vec2(0))
        : x(x), v(v), F(1), C(0), Jp(1), c(c)
    {}
};

std::vector<Particle> particles;

struct GridCell {
    Vec2 p;  // momentum
    float m; // mass
};

// node_res = cell_res + 1
GridCell grid[n + 1][n + 1];
Vec2 grid_velocity[n + 1][n + 1];

void reset_grid() {
    std::memset(grid, 0, sizeof(grid));
    std::memset(grid_velocity, 0, sizeof(grid_velocity));
}

void advance(real dt) {
    reset_grid();

    // Particles to grid transfer
    #if defined(_OPENMP)
        #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
    //for (auto &p : particles) {
        auto &p = particles[i];
        // element-wise floor
        Vector2i base_coord = (p.x * inv_dx - Vec2(0.5)).cast<int>();
        Vec2 fx = p.x * inv_dx - base_coord.cast<real>();
        // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        Vec2 w[3]{
            Vec2(0.5) * sqr(Vec2(1.5) - fx),
            Vec2(0.75) - sqr(fx - Vec2(1.0)),
            Vec2(0.5) * sqr(fx - Vec2(0.5))
        };
        // MPM Course Notes (Snow Plasticity), Eq'n 87 (p.26)
        auto e = std::exp(hardening * (1.f - p.Jp)),
             mu = mu_0 * e,
             lambda = lambda_0 * e;

        // Current volume
        real J = determinant(p.F);

        //Polar decomp. for fixed corotated model
        Mat2 r, s;
        polar_decomp(p.F, r, s);

        // Piola-Kirchhoff stress times Fᵀ
        // MPM Course Notes, Eq'n 52 (p.20)
        // MLS-MPM Paper notes below Eq'n 18
        auto PFt = 2*mu*(p.F-r)*transposed(p.F) + lambda*(J-1)*J;
        // Inverse mass for quadratic B-splines, MLS-MPM Paper below Eq'n 15
        auto inv_Mp = 4*inv_dx*inv_dx;
        // MLS-MPM Paper, Eq'n 18 (f_i = -∂E/∂x_i)
        auto f_p = -inv_Mp*vol*PFt;
        // Q_p from p.11 of the MLS-MPM Paper
        auto affine_momentum = f_p*dt + particle_mass*p.C;

        // Scatter mass and momentum to grid
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                auto dpos = (Vec2(i, j) - fx) * dx;
                auto w_ij = w[i].x*w[j].y;
                auto &g = grid[base_coord.x + i][base_coord.y + j];
                // translational momentum
                Vec2 mv = p.v*particle_mass;
                // add velocity field momentum & change due to stress forces
                g.p += w_ij * (mv + affine_momentum*dpos);
                g.m += w_ij * particle_mass;
            }
        }
    }

    // Calculate velocity for grid cells
    #if defined(_OPENMP)
        #pragma omp parallel for
    #endif
    for(int i = 0; i <= n; i++) {
        for(int j = 0; j <= n; j++) {
            auto &g = grid[i][j];
            auto &gv = grid_velocity[i][j];
            // No need for fuzzy comparison here
            if (g.m > 0) {
                // Get velocity from momentum
                gv = g.p / g.m;

                // Gravity
                gv.y -= 200 * dt;

                // boundary thick.,node coord
                real boundary=0.05,
                     x=(real)i/n,
                     y=real(j)/n;

                // Ceiling and Walls use "Sticky" collision handling
                if (x < boundary||x > 1-boundary||y > 1-boundary) {
                    gv = Vec2(0);
                }

                // Floor uses "Separate" collision handling
                if (y < boundary) {
                    gv.y = std::max(0.f, gv.y);
                }
            }
        }
    }

    // Grid to particle
    #if defined(_OPENMP)
        #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        auto &p = particles[i];
        Vector2i base_coord = (p.x*inv_dx - Vec2(0.5)).cast<int>(); // element-wise floor
        Vec2 fx = p.x * inv_dx - base_coord.cast<real>();
        Vec2 w[3]{
            Vec2(0.5) * sqr(Vec2(1.5) - fx),
            Vec2(0.75) - sqr(fx - Vec2(1.0)),
            Vec2(0.5) * sqr(fx - Vec2(0.5))
        };
        p.C = Mat2(0);
        p.v = Vec2(0);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                auto dpos = (Vec2(i, j) - fx),
                    grid_v = grid_velocity[base_coord.x + i][base_coord.y + j];
                auto weight = w[i].x * w[j].y;
                // Velocity
                p.v += weight * grid_v;
                // APIC C
                p.C += 4 * inv_dx * Mat2::outer_product(weight * grid_v, dpos);
            }
        }
        // Advection
        p.x += dt * p.v;
        // MLS-MPM F-update, MPM Course Notes Eq'n 180,181
        // or MLS-MPM Paper, Eq'n 17
        auto F = (Mat2(1) + dt * p.C) * p.F;
        // Calculate the SVD of F
        Mat2 svd_u, sig, svd_v;
        svd(F, svd_u, sig, svd_v);
        // Snow Plasticity, MPM Course Notes Eq'n 82 (p.26)
        if (plastic) {
          for (int i = 0; i < 2; i++) {
              sig[i][i] = clamp(sig[i][i], 1.f - 2.5e-2f, 1.f + 7.5e-3f);
          }
        }
        real oldJ = determinant(F);
        F = svd_u * sig * transposed(svd_v);
        p.Jp = clamp(p.Jp * oldJ / determinant(F), 0.6f, 20.f);
        p.F = F;
    }
}
// Seed particles with position and color
void add_object(Vec2 center, int c) {
  // Randomly sample 1000 particles in the square
    for (int i = 0; i < 500; i++) {
        particles.push_back(Particle((Vec2::rand()*2.f-Vec2(1))*0.08f + center, c));
    }
}
int main() {
    GUI gui("Real-time 2D MLS-MPM", window_size, window_size);
    add_object(Vec2(0.55,0.45), 0xED553B);
    add_object(Vec2(0.45,0.65), 0xF2B134);
    add_object(Vec2(0.55,0.85), 0x068587);
    auto &canvas = gui.get_canvas();

    // Main Loop
    for (int i = 0;; i++) {
        // Advance simulation
        advance(dt);
        // Visualize frame
        if (i % int(frame_dt / dt) == 0) {
            // Clear background
            canvas.clear(0x112F41);
            // Box
            canvas.rect(Vec2(0.04), Vec2(0.96)).radius(2).color(0x4FB99F).close();
            for(auto p: particles) {
                //Particles
                canvas.circle(p.x).radius(2).color(p.c);
            }
            // Update image
            gui.update();
            // canvas.img.write_as_image(fmt::format("tmp/{:05d}.png", i));
        }
    }
}
/* -----------------------------------------------------------------------------
** Reference: A Moving Least Squares Material Point Method with Displacement
              Discontinuity and Two-Way Rigid Body Coupling (SIGGRAPH 2018)

  By Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana,
  Chenfanfu Jiang

** FAQ:
Q: What is "real"?
A: real = float in this file.

Q: What are the hex numbers like 0xED553B?
A: They are RGB color values.
   The color scheme is borrowed from
   https://color.adobe.com/Copy-of-Copy-of-Core-color-theme-11449181/

Q: How can I get higher-quality?
A: Change n to 320; Change dt to 1e-5; Change E to 2e4;
   Change particle per cube from 500 to 8000 (Ln 72).
   After the change the whole animation takes ~3 minutes on my computer.

Q: How to record the animation?
A: Uncomment Ln 2 and 85 and create a folder named "tmp".
   The frames will be saved to "tmp/XXXXX.png".

   To get a video, you can use ffmpeg. If you already have taichi installed,
   you can simply go to the "tmp" folder and execute

     ti video 60

   where 60 stands for 60 FPS. A file named "video.mp4" is what you want.

Based on taich_mpm v1.3 (Oct 30, 2018) by Yuanming Hu
https://github.com/yuanming-hu/taichi_mpm

----------------------------------------------------------------------------- */
