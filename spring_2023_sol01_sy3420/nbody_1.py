"""
    N-body simulation.
"""

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES = {
    'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS),

    'jupiter': ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                9.54791938424326609e-04 * SOLAR_MASS),

    'saturn': ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               2.85885980666130812e-04 * SOLAR_MASS),

    'uranus': ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               4.36624404335156298e-05 * SOLAR_MASS),

    'neptune': ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                5.15138902046611451e-05 * SOLAR_MASS)}


def advance(dt):
    '''
        advance the system one timestep
    '''

    #we take all the functions above and modify it into this function here so it does not need
    #to call it every time.
    seenit = []
    for body1 in BODIES.keys():
        for body2 in BODIES.keys():
            if (body1 != body2) and not (body2 in seenit):
                ([x1, y1, z1], v1, m1) = BODIES[body1]
                ([x2, y2, z2], v2, m2) = BODIES[body2]
                # (dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2) don't need it anymore
                # update_vs(v1, v2, dt, dx, dy, dz, m1, m2) don't need it anymore
                # seenit.append(body1) don't need it anymore
                (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2)
                mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
                val1 = m1 * mag
                val2 = m2 * mag
                v1[0] -= dx * val2
                v1[1] -= dy * val2
                v1[2] -= dz * val2
                v2[0] += dx * val1
                v2[1] += dy * val1
                v2[2] += dz * val1



        
    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        # update_rs(r, dt, vx, vy, vz) don't need it anymore
        r[0] += dt * vx
        r[1] += dt * vy
        r[2] += dt * vz
def compute_energy(m1, m2, dx, dy, dz):
    return (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
    
def report_energy(e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    seenit = []
    for body1 in BODIES.keys():
        for body2 in BODIES.keys():
            if (body1 != body2) and not (body2 in seenit):
                ((x1, y1, z1), v1, m1) = BODIES[body1]
                ((x2, y2, z2), v2, m2) = BODIES[body2]
                # (dx, dy, dz) = compute_deltas(x1, x2, y1, y2, z1, z2)
                (dx, dy, dz) = (x1 - x2, y1 - y2, z1 - z2)
                # e -= compute_energy(m1, m2, dx, dy, dz)
                e -= (m1 * m2) / ((dx * dx + dy * dy + dz * dz) ** 0.5)
                seenit.append(body1)
        
    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        e += m * (vx * vx + vy * vy + vz * vz) / 2.
        
    return e

def offset_momentum(ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''
    for body in BODIES.keys():
        (r, [vx, vy, vz], m) = BODIES[body]
        px -= vx * m
        py -= vy * m
        pz -= vz * m
        
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m


def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    offset_momentum(BODIES[reference])

    for _ in range(loops):
        report_energy()
        for _ in range(iterations):
            advance(0.01)
        print(report_energy())

if __name__ == '__main__':
    nbody(100, 'sun', 20000)

