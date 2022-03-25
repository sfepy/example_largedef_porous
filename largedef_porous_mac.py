# This example implements homogenization of porous structures undergoing finite strains.
#
# largedef_porous_mac.py - problem at (global) macroscopic level
# largedef_porous_mic.py - local subproblems, homogenized coefficients
#
# The mathematical model and numerical results are described in: 
#
# LUKEŠ V., ROHAN E.
# Homogenization of large deforming fluid-saturated porous structures
# https://arxiv.org/abs/2012.03730
#
# Run simulation:
#
#   ./simple.py example_largedef_porous-1/largedef_porous_mac.py
#
# The results are stored in `example_largedef_porous-1/results` directory.
#

import numpy as nm
import six
import os.path as osp
from sfepy import data_dir
from sfepy.base.base import Struct, output, debug
from sfepy.terms.terms_hyperelastic_ul import HyperElasticULFamilyData
from sfepy.homogenization.micmac import get_homog_coefs_nonlinear
import sfepy.linalg as la
from sfepy.solvers.ts import TimeStepper

wdir = osp.dirname(__file__)

hyperelastic_data = {
    'update_materials': True,
    'state': {'u': None, 'du': None,
              'p': None, 'dp': None},
    'mapping0': None,
    'coors0': None,
    'macro_data': None,
}


def post_process(out, pb, state, extend=False):
    ts = hyperelastic_data['ts']

    if isinstance(state, dict):
        pass
    else:
        stress = pb.evaluate('ev_integrate_mat.i.Omega(solid.S, u)',
                             mode='el_avg')

        out['cauchy_stress'] = Struct(name='output_data',
                                      mode='cell',
                                      data=stress)

        ret_stress = pb.evaluate('ev_integrate_mat.i.Omega(solid.Q, u)',
                                 mode='el_avg')

        out['retardation_stress'] = Struct(name='output_data',
                                           mode='cell',
                                           data=ret_stress)

        strain = pb.evaluate('ev_integrate_mat.i.Omega(solid.E, u)',
                             mode='el_avg')

        out['green_strain'] = Struct(name='output_data',
                                     mode='cell',
                                     data=strain)

        he_state = hyperelastic_data['state']
        for ch in pb.conf.chs:
            plab = 'p%d' % ch
            out['p0_%d' % ch] = Struct(name='output_data',
                                       mode='vertex',
                                       data=he_state[plab][:, nm.newaxis])

            dvel = pb.evaluate('ev_diffusion_velocity.i.Omega(solid.C%d, %s)' % (ch, plab),
                               mode='el_avg')
            out['w%d' % ch] = Struct(name='output_data',
                                     mode='cell',
                                     data=dvel)

        out['u0'] = Struct(name='output_data',
                           mode='vertex',
                           data=he_state['u'])

    return out


def homog_macro_map(ccoors, macro, nel):
    nqpe = ccoors.shape[0] // nel
    macro_ = {k: nm.sum(v.reshape((nel, nqpe) + v.shape[1:]), axis=1) / nqpe
              for k, v in macro.items()}
    macro_['recovery_idxs'] = []
    ccoors_ = nm.sum(ccoors.reshape((nel, nqpe) + ccoors.shape[1:]), axis=1) / nqpe

    return ccoors_, macro_


def homog_macro_remap(homcf, ncoor):
    nqpe = ncoor // homcf['Volume_total'].shape[0]
    homcf_ = {k: nm.repeat(v, nqpe, axis=0) for k, v in homcf.items()
              if not k == 'Volume_total'}

    return homcf_


def get_homog_mat(ts, coors, mode, term=None, problem=None, **kwargs):
    hyperela = hyperelastic_data
    ts = hyperela['ts']

    output('get_homog_mat: mode=%s, update=%s'\
           % (mode, hyperela['update_materials']))

    if not mode == 'qp':
        return

    if not hyperela['update_materials']:
        out = hyperela['homog_mat']
        return {k: nm.array(v) for k, v in six.iteritems(out)}

    dim = problem.domain.mesh.dim
    nqp = coors.shape[0]

    svars = problem.equations.variables
    state_u = svars['u']
    if len(state_u.field.mappings0) == 0:
        state_u.field.get_mapping(term.region, term.integral,
                                  term.integration)
        state_u.field.save_mappings()

    state_u.field.clear_mappings()
    svars.set_state_parts({'u': hyperela['state']['u'].ravel()})

    mtx_f = problem.evaluate('ev_def_grad.i.Omega(u)',
                             mode='qp').reshape(-1, dim, dim)

    # relative deformation gradient
    if hasattr(problem, 'mtx_f_prev'):
        rel_mtx_f = la.dot_sequences(mtx_f, nm.linalg.inv(problem.mtx_f_prev),
                                     'AB')
    else:
        rel_mtx_f = mtx_f

    problem.mtx_f_prev = mtx_f.copy()

    macro_data = {
        'mtx_e_rel': rel_mtx_f - nm.eye(dim),  # relative macro strain
    }

    for ch in problem.conf.chs:
        plab = 'p%d' % ch
        svars.set_state_parts({plab: hyperela['state'][plab]})
        macro_data['p%d_0' % ch] = \
            problem.evaluate('ev_integrate.i.Omega(p%d)' % ch,
                             mode='qp').reshape(-1, 1, 1)
        macro_data['gp%d_0' % ch] = \
            problem.evaluate('ev_grad.i.Omega(p%d)' % ch,
                             mode='qp').reshape(-1, dim, 1)

        svars.set_state_parts({plab: hyperela['state']['d' + plab]})
        macro_data['dp%d_0' % ch] = \
            problem.evaluate('ev_integrate.i.Omega(p%d)' % ch,
                             mode='qp').reshape(-1, 1, 1)
        macro_data['gdp%d_0' % ch] = \
            problem.evaluate('ev_grad.i.Omega(p%d)' % ch,
                             mode='qp').reshape(-1, dim, 1)

    nel = term.region.entities[-1].shape[0]
    ccoors0, macro_data0 = homog_macro_map(coors, macro_data, nel)
    macro_data0['macro_ccoor'] = ccoors0
    out0 = get_homog_coefs_nonlinear(ts, ccoors0, mode, macro_data0,
                                     term=term, problem=problem,
                                     iteration=ts.step, **kwargs)
    out0['C1'] += nm.eye(2) * 1e-12  # ! auxiliary permeability
    out0['C2'] += nm.eye(2) * 1e-12  # ! auxiliary permeability
    out = homog_macro_remap(out0, nqp)

    # Green strain
    out['E'] = 0.5 * (la.dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim))

    for ch in problem.conf.chs:
        out['B%d' % ch] = out['B%d' % ch].reshape((nqp, dim, dim)) 
    out['Q'] = out['Q'].reshape((nqp, dim, dim))

    hyperela['time'] = ts.step
    hyperela['homog_mat'] = \
        {k: nm.array(v) for k, v in six.iteritems(out)}
    hyperela['update_materials'] = False
    hyperela['macro_data'] = macro_data

    return out


def incremental_algorithm(pb):
    hyperela = hyperelastic_data
    chs = pb.conf.chs
    ts = pb.conf.ts

    hyperela['ts'] = ts
    hyperela['ofn_trunk'] = pb.ofn_trunk + '_%03d'
    pb.domain.mesh.coors_act = pb.domain.mesh.coors.copy()

    pbvars = pb.get_variables()
    he_state = hyperela['state']

    out = []
    out_data = {}

    coors0 = pbvars['u'].field.get_coor()

    he_state['coors0'] = coors0.copy()
    he_state['u'] = nm.zeros_like(coors0)
    he_state['du'] = nm.zeros_like(coors0)

    for ch in chs:
        plab = 'p%d' % ch
        press0 = pbvars[plab].field.get_coor()[:, 0].squeeze()
        he_state[plab] = nm.zeros_like(press0)
        he_state['d' + plab] = nm.zeros_like(press0)

    for step, time in ts:
        print('>>> step %d (%e):' % (step, time))
        hyperela['update_materials'] = True

        pb.ofn_trunk = hyperela['ofn_trunk'] % step

        yield pb, out

        state = out[-1][1]
        result = state.get_state_parts()
        du = result['u']

        he_state['u'] += du.reshape(he_state['du'].shape)
        he_state['du'][:] = du.reshape(he_state['du'].shape)
        pb.set_mesh_coors(he_state['u'] + he_state['coors0'],
                          update_fields=True, actual=True, clear_all=False)

        for ch in chs:
            plab = 'p%d' % ch
            dp = result[plab]
            he_state[plab] += dp
            he_state['d' + plab][:] = dp

        out_data = post_process(out_data, pb, state, extend=False)
        filename = pb.get_output_name()
        pb.save_state(filename, out=out_data)

        yield None

        print('<<< step %d finished' % step)


def move(ts, coor, problem=None, ramp=0.4, **kwargs):
    ts = problem.conf.ts
    nrs = round(ts.n_step * ramp)
    switch = 1 if (ts.step <= nrs) and (ts.step > 0) else 0
    displ = nm.ones((coor.shape[0],)) * problem.conf.move_val / nrs * switch

    return displ


def define():
    chs = [1, 2]

    ts = TimeStepper(0, 0.15, n_step=30)

    options = {
        'output_dir': osp.join(wdir, 'results'),
        'micro_filename': osp.join(wdir, 'largedef_porous_mic.py'),
        'parametric_hook': 'incremental_algorithm',
    }

    materials = {
        'solid': 'get_homog',
    }

    fields = {
        'displacement': ('real', 'vector', 'Omega', 1),
        'pressure': ('real', 'scalar', 'Omega', 1),
    }

    variables = {
        'u': ('unknown field', 'displacement', 0),
        'v': ('test field', 'displacement', 'u'),
        'p1': ('unknown field', 'pressure', 1),
        'q1': ('test field', 'pressure', 'p1'),
        'p2': ('unknown field', 'pressure', 2),
        'q2': ('test field', 'pressure', 'p2'),
        'U': ('parameter field', 'displacement', 'u'),
    }

    filename_mesh = osp.join(wdir, 'macro_mesh_3x2.vtk')
    mesh_d, move_val = 0.24, -0.04

    regions = {
        'Omega': 'all',
        'Left': ('vertices in (x < 0.0001)', 'facet'),
        'Right': ('vertices in (x > %e)' % (mesh_d * 0.999), 'facet'),
        'Recovery': ('cell 1', 'cell'),
    }

    ebcs = {
        'left_fix_all': ('Left', {'u.all': 0.0}),
        'right_fix_x': ('Right', {'u.0': 0.0}),
        'right_move_x': ('Right', {'u.1': 'move'}),
    }

    micro_args = {
        'eps0': mesh_d / 3,
        'dt': ts.dt,
    }

    functions = {
        'move': (move,),
        'get_homog': (lambda ts, coors, mode, **kwargs:
                      get_homog_mat(ts, coors, mode,
                                    define_args=micro_args, **kwargs),),
    }

    integrals = {
        'i': 3,
    }

    equations = {
        #  eq. (60)
        'balance_of_forces': """
            dw_nonsym_elastic.i.Omega(solid.A, v, u)
          - dw_biot.i.Omega(solid.B1, v, p1)
          - dw_biot.i.Omega(solid.B2, v, p2)
          =
          - dw_lin_prestress.i.Omega(solid.S, v)
          - dw_lin_prestress.i.Omega(solid.Q, v)""",
        #  eq. (61), alpha = 1
        'mass_conservation_1': """
     - %e * dw_biot.i.Omega(solid.B1, u, q1)
          - dw_dot.i.Omega(solid.G11, q1, p1)
          - dw_dot.i.Omega(solid.G12, q1, p2)
          - dw_diffusion.i.Omega(solid.C1, q1, p1)
          =
            dw_volume_lvf.i.Omega(solid.Z1, q1)
        """ % (1 / ts.dt),
        #  eq. (61), alpha = 2
        'mass_conservation_2': """
     - %e * dw_biot.i.Omega(solid.B2, u, q2)
          - dw_dot.i.Omega(solid.G21, q2, p1)
          - dw_dot.i.Omega(solid.G22, q2, p2)
          - dw_diffusion.i.Omega(solid.C2, q2, p2)
          =
            dw_volume_lvf.i.Omega(solid.Z2, q2)
        """ % (1. / ts.dt),
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {
            'eps_a': 1e-3,
            'eps_r': 1e-3,
            'i_max': 1,
        }),
    }

    return locals()