#pragma once

#include "particles/base_particle.hpp"

namespace md { namespace integrators {

// Base class for all integrators
template <class Derived, class ParticleT>
class BaseIntegrator {
public:
    using P = ParticleT;

    explicit BaseIntegrator(P& p) : p_(p) {}
    BaseIntegrator(const BaseIntegrator&) = delete;
    BaseIntegrator& operator=(const BaseIntegrator&) = delete;

    // Update the state of the particles using the integrator logic
    inline void step() { derived().step_impl(); }

    // Initialize whatever dynamics needs to be done before the first step
    inline void init() { derived().init_impl(); }

protected:
    inline Derived&       derived()       { return static_cast<Derived&>(*this); }
    inline const Derived& derived() const { return static_cast<const Derived&>(*this); }

    inline P&       particle()       { return p_; }
    inline const P& particle() const { return p_; }

private:
    P& p_;
};


} // namespace integrators
} // namespace md