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
import os.path as osp
from sfepy.homogenization.utils import define_box_regions
import sfepy.homogenization.coefs_base as cb
import sfepy.discrete.fem.periodic as per
from sfepy.base.base import Struct, output, debug, get_default
from sfepy.terms.terms_hyperelastic_ul import\
    HyperElasticULFamilyData, NeoHookeanULTerm, BulkPenaltyULTerm
from sfepy.terms.extmods.terms import sym2nonsym
from sfepy.discrete.functions import ConstantFunctionByRegion
import sfepy.linalg as la

wdir = osp.dirname(__file__)

sym_eye = {
    2: nm.array([[1, 1, 0]]).T,
    3: nm.array([[1, 1, 1, 0, 0, 0]]).T,
}

nonsym_delta = {
    2: nm.array([[0, 0, 0, -1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [-1, 0, 0, 0]]),
    3: nm.array([[0, 0, 0, 0, -1, 0, 0, 0, -1],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [-1, 0, 0, 0, 0, 0, 0, 0, -1],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [-1, 0, 0, 0, -1, 0, 0, 0, 0]]),
}

material_cache = {}


class MyCoefNonSymNonSym(cb.CoefSymSym):
    from sfepy.homogenization.utils import iter_nonsym

    iter_sym = staticmethod(iter_nonsym)
    is_sym = False

    def __call__(self, volume, problem=None, data=None):
        problem = get_default(problem, self.problem)
        isym = [ii for ii in self.iter_sym(problem.get_dim())]

        return self.get_coef(isym, isym, volume, problem, data)


class CorrStatePressureCh(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        micro_state, im = problem.micro_state
        macro_data = problem.homogenization_macro_data

        pvar = self.variable

        if micro_state[pvar] is not None:
            coors = problem.fields['pressure' + pvar[-1]].coors
            press = nm.dot(coors, macro_data['g' + pvar + '_0'][im])[:, 0] \
                + micro_state[pvar][im] - macro_data[pvar + '_0'][im][:, 0]
        else:
            ndof = problem.fields['pressure' + pvar[-1]].n_vertex_dof
            press = nm.zeros((ndof,), dtype=nm.float64)
        
        corr_sol = cb.CorrSolution(name=self.name,
                                   state={pvar: press})

        return corr_sol


class CorrStatePressureM(cb.CorrMiniApp):
    def __call__(self, problem=None, data=None):
        problem = get_default(problem, self.problem)

        micro_state, im = problem.micro_state

        if micro_state['p'] is not None:
            press = micro_state['p'][im]
        else:
            ndof = problem.fields['pressure'].n_vertex_dof
            press = nm.zeros((ndof,), dtype=nm.float64)
        
        corr_sol = cb.CorrSolution(name=self.name,
                                   state={'p': press})

        return corr_sol


def post_process_hook(pb, nd_data, qp_data, ccoor, vol, im, tstep, eps0,
                      recovery_file_tag=''):
    from sfepy.discrete.fem import Mesh

    elavg_data = {}
    elvol = nm.sum(vol, axis=1)
    sh0 = vol.shape[:2]
    for k in qp_data.keys():
        print(k, qp_data[k].shape, vol.shape, elvol.shape)
        sh1 = qp_data[k].shape[1:]
        val = qp_data[k].reshape(sh0 + sh1)
        elavg_data[k] = (nm.sum(val * vol, axis=1) / elvol)[:, None, ...]

    output_dir = pb.conf.options.get('output_dir', '.')
    suffix = '%03d.%03d' % (im, tstep)
    coors = pb.get_mesh_coors(actual=True)
    coors = (coors - 0.5*(nm.max(coors, axis=0)\
        - nm.min(coors, axis=0))) * eps0 + ccoor

    # Y
    out = {}
    out['displacement'] = Struct(name='output_data', mode='vertex',
                                 data=nd_data['u'] * eps0, variable='u')
    out['green_strain'] = Struct(name='output_data', mode='cell',
                                 data=elavg_data['E'])

    mesh = Mesh.from_region(pb.domain.regions['Y'], pb.domain.mesh)
    mesh.coors[:] = coors

    micro_name = pb.get_output_name(extra='recovered_Y_'
                                    + recovery_file_tag + suffix)
    filename = osp.join(output_dir, osp.basename(micro_name))

    output('  %s' % filename)
    mesh.write(filename, io='auto', out=out)

    p_tab = {'Ym': 'p', 'Yc1': 'p1', 'Yc2': 'p2'}
    mesh0 = pb.domain.mesh
    for rname in ['Ym'] + ['Yc%d' %ch for ch in pb.conf.chs]:
        reg = pb.domain.regions[rname]
        cells = reg.get_cells()

        out = {}
        out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                      data=elavg_data['S'][cells])
        out['velocity'] = Struct(name='output_data', mode='cell',
                                 data=elavg_data['w'][cells])
        out['pressure'] = Struct(name='output_data', mode='vertex',
                                 data=nd_data[p_tab[rname]][:, None])

        ac = nm.ascontiguousarray
        conn = mesh0.cmesh.get_cell_conn()
        cells = reg.entities[-1]
        verts = reg.entities[0]
        aux = nm.diff(conn.offsets)
        assert nm.sum(nm.diff(aux)) == 0
        conn = ac(conn.indices.reshape((mesh0.n_el, aux[0]))[cells])
        remap = -nm.ones(mesh0.n_nod)
        remap[verts] = nm.arange(verts.shape[0])
        conn = remap[conn]

        mesh = Mesh.from_data('region_%s' % rname,
                              ac(coors[verts]),
                              ac(mesh0.cmesh.vertex_groups[verts]),
                              [conn],
                              [ac(mesh0.cmesh.cell_groups[cells])],
                              [mesh0.descs[0]])

        micro_name = pb.get_output_name(extra='recovered_%s_' % rname 
                                        + recovery_file_tag + suffix)
        filename = osp.join(output_dir, osp.basename(micro_name))

        output('  %s' % filename)
        mesh.write(filename, io='auto', out=out)


def get_hyperelastic_Y(pb, term, micro_state, im, region_name='Y'):
    from sfepy.terms import Term

    region = pb.domain.regions[region_name]
    el = region.get_cells().shape[0]
    nqp = tuple(term.integral.qps.values())[0].n_point
    npts = el * nqp

    mvars = pb.create_variables(
        ['U', 'P'] + ['P%d' % ch for ch in pb.conf.chs])
    state_u, state_p = mvars['U'], mvars['P']

    termY = Term.new('ev_grad(U)', term.integral,
                     region, U=mvars['U'])

    if state_u.data[0] is None:
        state_u.init_data()

    u_mic = micro_state['coors'][im] - pb.domain.get_mesh_coors(actual=False)
    state_u.set_data(u_mic)
    state_u.field.clear_mappings()
    family_data = pb.family_data(state_u, region,
                                 term.integral, term.integration)

    if len(state_u.field.mappings0) == 0:
        state_u.field.save_mappings()

    n_el, n_qp, dim, _, _ = state_u.get_data_shape(term.integral,
                                                   term.integration,
                                                   region_name)

    # relative displacement
    state_u.set_data(micro_state['coors'][im] - micro_state['coors_prev'][im]) # \bar u (du_prev)
    grad_du_qp = state_u.evaluate(mode='grad',
        integral=term.integral).reshape((npts, dim, dim))
    div_du_qp = nm.trace(grad_du_qp, axis1=1, axis2=2).reshape((npts, 1, 1))
   
    press_qp = nm.zeros((n_el, n_qp, 1, 1), dtype=nm.float64)
    grad_press_qp = nm.zeros((n_el, n_qp, dim, 1), dtype=nm.float64)

    if micro_state['p'] is not None:
        p_mic = micro_state['p'][im]
        state_p.set_data(p_mic)
        cells = state_p.field.region.get_cells()
        press_qp[cells, ...] = state_p.evaluate(integral=term.integral)
        grad_press_qp[cells, ...] = state_p.evaluate(mode='grad',
                                                     integral=term.integral)

        pch_mic = {}
        for ch in pb.conf.chs:
            state_pi = mvars['P%d' % ch]
            pch_mic[ch] = micro_state['p%d' % ch][im]
            state_pi.set_data(micro_state['p%d' % ch][im])
            cells = mvars['P%d' % ch].field.region.get_cells()
            press_qp[cells, ...] = state_pi.evaluate(integral=term.integral)
            grad_press_qp[cells, ...] = state_pi.evaluate(mode='grad',
                                                          integral=term.integral)

        press_qp = press_qp.reshape((npts, 1, 1))
        grad_press_qp = grad_press_qp.reshape((npts, dim, 1))
    else:
        p_mic = nm.zeros((state_p.n_dof,), dtype=nm.float64)
        pch_mic = {ch: nm.zeros((mvars['P%d' % ch].n_dof,), dtype=nm.float64)
                   for ch in pb.conf.chs}
        press_qp = nm.zeros((npts, 1, 1), dtype=nm.float64)
        grad_press_qp = nm.zeros((npts, dim, 1), dtype=nm.float64)

    conf_mat = pb.conf.materials
    solid_key = [key for key in conf_mat.keys() if 'solid' in key][0]
    solid_mat = conf_mat[solid_key].values
    mat = {}
    for mat_key in ['mu', 'K']:
        if isinstance(solid_mat[mat_key], dict):
            mat_fun = ConstantFunctionByRegion({mat_key: solid_mat[mat_key]})
            mat0 = mat_fun.function(ts=None, coors=nm.empty(npts), mode='qp',
                                    term=termY, problem=pb)[mat_key]
            mat[mat_key] = mat0.reshape((n_el, n_qp) +  mat0.shape[-2:])
        else:
            mat[mat_key] = nm.ones((n_el, n_qp, 1, 1)) * solid_mat[mat_key]

    shape = family_data.green_strain.shape[:2]
    assert(npts == nm.prod(shape))
    sym = family_data.green_strain.shape[-2]
    dim2 = dim**2


    fargs = [family_data.get(name)
             for name in NeoHookeanULTerm.family_data_names]
    stress_eff = nm.empty(shape + (sym, 1), dtype=nm.float64)
    tanmod_eff = nm.empty(shape + (sym, sym), dtype=nm.float64)
    NeoHookeanULTerm.stress_function(stress_eff, mat['mu'], *fargs)
    NeoHookeanULTerm.tan_mod_function(tanmod_eff, mat['mu'], *fargs)

    stress_eff_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    tanmod_eff_ns = nm.zeros(shape + (dim2, dim2), dtype=nm.float64)
    sym2nonsym(stress_eff_ns, stress_eff)
    sym2nonsym(tanmod_eff_ns, tanmod_eff)

    J = family_data.det_f.reshape((npts, 1, 1))
    mtx_f = family_data.mtx_f.reshape((npts, dim, dim))

    stress_p = - press_qp * J * sym_eye[dim]

    mat_A = (tanmod_eff_ns + stress_eff_ns).reshape((npts, dim2, dim2))\
        + J * press_qp * nonsym_delta[dim]
   
    mtxI = nm.eye(dim)
    mat_BI = (mtxI * div_du_qp - grad_du_qp).transpose(0, 2, 1) + mtxI
    
    mat['K'] = mat['K'].reshape((npts, dim, dim))
    mat_H = div_du_qp * mat['K']\
        - la.dot_sequences(mat['K'], grad_du_qp, 'ABT')\
        - la.dot_sequences(grad_du_qp, mat['K'], 'ABT')

    out = {
        'E': 0.5 * (la.dot_sequences(mtx_f, mtx_f, 'ATB') - nm.eye(dim)),  # Green strain
        'S': (stress_eff.reshape((npts, sym, 1)) + stress_p) / J,  # Cauchy stress
        'A': mat_A / J,  # tangent elastic tensor, eq. (20)
        'BI': mat_BI,
        'KH': mat['K'] + mat_H,
        'H': mat_H,
        'dK': mat['K'] * 0,  # constant permeability => dK = 0
        'w': -grad_press_qp * mat['K'],  # perfusion velocity
    }

    return out


def def_mat(ts, coors, mode=None, term=None, problem=None, **kwargs):
    if not (mode == 'qp'):
        return

    pb = problem

    if not hasattr(pb, 'family_data'):
        pb.family_data = HyperElasticULFamilyData()

    macro_data = pb.homogenization_macro_data
    micro_state, im = pb.micro_state
    mac_id = micro_state['id'][im]
    cache_key = ('Y', term.integral.name, term.integration,
                 im, macro_data['macro_time_step'])

    if cache_key not in material_cache:
        out = get_hyperelastic_Y(pb, term, micro_state, im)
        material_cache[cache_key] = out

        # clear cache
        to_remove = []
        for k in material_cache.keys():
            if not(k[-1] == macro_data['macro_time_step']):
                to_remove.append(k)
        for k in to_remove:
            del(material_cache[k])

        if 'recovery_idxs' in macro_data and\
            mac_id in macro_data['recovery_idxs'] and\
            macro_data['macro_time_step'] > 0:

            output('>>> recovery: %d / %d / %d'\
                % (im, macro_data['macro_time_step'], mac_id))

            qp_data = {}
            for k in ['S', 'E', 'w']:
                qp_data[k] = out[k]

            nodal_data = {}
            nodal_data['u'] =\
                micro_state['coors'][im] - pb.domain.get_mesh_coors(actual=False)
            for st in ['p', 'p1', 'p2']:
                nodal_data[st] = micro_state[st][im]

            state_u = pb.create_variables(['U'])['U']
            state_u_det = state_u.field.get_mapping(term.region, term.integral,
                                                    term.integration,
                                                    get_saved=True)[0].det
            post_process_hook(pb, nodal_data, qp_data,
                            macro_data['macro_ccoor'][im], state_u_det,
                            im, macro_data['macro_time_step'], pb.conf.eps0)
    else:
        out = material_cache[cache_key]

    npts = coors.shape[0]
    region = term.region

    if region.name == 'Y':
        return out
    else:
        el = region.get_cells()[:, nm.newaxis]
        nel = el.shape[0]
        nqp = tuple(term.integral.qps.values())[0].n_point

        assert(npts == nel * nqp)

        idxs = (el * nm.array([nqp] * nqp) + nm.arange(nqp)).flatten()
        lout = {k: v[idxs, ...] for k, v in out.items()}

        return lout


def define(eps0=0.01, dt=0.1, nch=2, dim=2,
           filename_mesh='micro_mesh.vtk', approx_u=1, approx_p=1,
           multiprocessing=False):

    filename_mesh = osp.join(wdir, filename_mesh)

    chs = list(nm.arange(nch) + 1)
    update_u_by_p = [('corrs_%d' % ch, 'u', 'dp%d_0' % ch) for ch in chs]
    update_p_by_p = [('corrs_%d' % ch, 'p', 'dp%d_0' % ch) for ch in chs]

    options = {
        'coefs': 'coefs',
        'requirements': 'requirements',
        'volume': {'expression': 'd_volume.5.Y(u)'},
        'output_dir': osp.join(wdir, 'results'),
        'coefs_filename': 'coefs_hp',
        # 'chunks_per_worker': 2,
        'multiprocessing': multiprocessing,
        'file_per_var': True,
        'micro_update': {
            'coors_prev': None,
            'coors': [('corrs_rs', 'u', 'mtx_e_rel'),
                    ('corrs_p', 'u', None)] + update_u_by_p,
            'p': [('corrs_rs', 'p', 'mtx_e_rel'),
                ('corrs_p', 'p', None)] + update_p_by_p,
            'p1': [('corrs_eta1', 'p1', 'gdp1_0', eps0),
                ('corrs_p1', 'p1', None, eps0),
                (None, None, 'dp1_0')],
            'p2': [('corrs_eta2', 'p2', 'gdp2_0', eps0),
                ('corrs_p2', 'p2', None, eps0),
                (None, None, 'dp2_0')],
        },
        'mesh_update_variable': 'u',
        'file_format': 'vtk',
    }

    fields = {
        'displacement': ('real', 'vector', 'Y', approx_u),
        'pressure': ('real', 'scalar', 'Ym', approx_p),
    }

    functions = {
        'match_x_plane': (per.match_x_plane,),
        'match_y_plane': (per.match_y_plane,),
        'match_z_plane': (per.match_z_plane,),
        'mat_fce': (def_mat,),
    }

    integrals = {
        'i': 3,
    }

    materials = {
        'mat_he': 'mat_fce',
        'solid': ({
            'mu': {
                'Ym': 1e6,
                'Yc1': .6e6,
                'Yc2': .6e6
            },
            'K': {
                'Ym': 1e-11 * nm.eye(dim) / eps0**2,
                'Yc1': 1e-6 * nm.eye(dim),
                'Yc2': 2e-6 * nm.eye(dim),
            },
        },),
    }

    variables = {
        'u': ('unknown field', 'displacement'),
        'v': ('test field', 'displacement', 'u'),
        'Piu': ('parameter field', 'displacement', 'u'),
        'Pi1u': ('parameter field', 'displacement', '(set-to-None)'),
        'Pi2u': ('parameter field', 'displacement', '(set-to-None)'),
        'p': ('unknown field', 'pressure'),
        'q': ('test field', 'pressure', 'p'),
        'Pip': ('parameter field', 'pressure', 'p'),
        'Pi1p': ('parameter field', 'pressure', '(set-to-None)'),
        'Pi2p': ('parameter field', 'pressure', '(set-to-None)'),
        'U': ('parameter field', 'displacement', '(set-to-None)'),
        'P': ('parameter field', 'pressure', '(set-to-None)'),
    }

    regions = {
        'Y': 'all',
        'Ym': 'cells of group 1',
        'Left_': ('r.Left -v r.Corners', 'vertex'),
        'Right_': ('r.Right -v r.Corners', 'vertex'),
        'Top_': ('r.Top -v r.Corners', 'vertex'),
        'Bottom_': ('r.Bottom -v r.Corners', 'vertex'),
        'Left0': ('r.Left_ -v r.Gamma_mc', 'vertex'),
        'Right0': ('r.Right_ -v r.Gamma_mc', 'vertex'),
        'Top0': ('r.Top_ -v r.Gamma_mc', 'vertex'),
        'Bottom0': ('r.Bottom_ -v r.Gamma_mc', 'vertex'),
    }

    if dim == 3:
        regions.update({
            'Near_': ('r.Near -v r.Corners', 'vertex'),
            'Far_': ('r.Far -v r.Corners', 'vertex'),
            'Near0': ('r.Near_ -v r.Gamma_mc', 'vertex'),
            'Far0': ('r.Far_ -v r.Gamma_mc', 'vertex'),
        })

    if nch > 1:
        regions.update({
            'Gamma_mc': (' +s '.join(['r.Gamma%d' % ii for ii in chs]),
                        'facet', 'Ym')
        })
    else:
        regions.update({'Gamma_mc': ('copy r.Gamma1', 'facet')})

    regions.update(define_box_regions(dim, (0, 0, 0)[:dim], (1, 1, 1)[:dim]))

    ebcs = {
        'fixed_u': ('Corners', {'u.all': 0.0}),
        'fixed_p': ('Gamma_mc', {'p.0': 0.0}),
    }

    epbcs = {
        'periodic_ux': (['Right_', 'Left_'], {'u.all': 'u.all'}, 'match_x_plane'),
        'periodic_px': (['Right0', 'Left0'], {'p.0': 'p.0'}, 'match_x_plane'),
    }

    periodic_all = ['periodic_ux', 'periodic_uy', 'periodic_px', 'periodic_py']

    if dim == 3:
        epbcs.update({
            'periodic_uy': (['Near_', 'Far_'], {'u.all': 'u.all'}, 'match_y_plane'),
            'periodic_uz': (['Bottom_', 'Top_'], {'u.all': 'u.all'}, 'match_z_plane'),
            'periodic_py': (['Near0', 'Far0'], {'p.0': 'p.0'}, 'match_y_plane'),
            'periodic_pz': (['Bottom0', 'Top0'], {'p.0': 'p.0'}, 'match_z_plane'),
        })
        periodic_all += ['periodic_uz', 'periodic_pz']
    else:
        epbcs.update({
            'periodic_uy': (['Top_', 'Bottom_'], {'u.all': 'u.all'}, 'match_y_plane'),
            'periodic_py': (['Top0', 'Bottom0'], {'p.0': 'p.0'}, 'match_y_plane'),
        })

    lcbcs = {}

    coefs = {
        'A1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_rs'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', ('pis_u', 'corrs_rs'), 'u'),
                            ('Pi2u', ('pis_u', 'corrs_rs'), 'u')],
            'class': MyCoefNonSymNonSym,
        },
        'A2': {
            'status': 'auxiliary',
            'requires': ['corrs_rs'],
            'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
            'set_variables': [('Pi1p', 'corrs_rs', 'p'),
                            ('Pi2p', 'corrs_rs', 'p')],
            'class': cb.CoefNonSymNonSym,
        },
        'A': {  # effective viscoelastic incremental tensor, eq. (51)
            'requires': ['c.A1', 'c.A2'],
            'expression': 'c.A1 + %e * c.A2' % dt,
            'class': cb.CoefEval,
        },
        'S': {  # averaged Cauchy stress, eq. (53)
            'expression': 'ev_volume_integrate_mat.i.Y(mat_he.S, u)',
            'class': cb.CoefOne,
        },
        'Q1': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
            'set_variables': [('Pi1u', 'pis_u', 'u'),
                            ('Pi2u', 'corrs_p', 'u')],
            'class': cb.CoefNonSym,
        },
        'Q2': {
            'status': 'auxiliary',
            'requires': ['pis_u', 'corrs_p'],
            'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
            'set_variables': [('Pi1p', 'corrs_p', 'p'),
                            ('Pi1u', 'pis_u', 'u')],
            'class': cb.CoefNonSym,
        },
        'Q': {  # retardation stress, eq. (54)
            'requires': ['c.Q1', 'c.Q2'],
            'expression': 'c.Q1 - c.Q2',
            'class': cb.CoefEval,
        },
    }

    requirements = {
        'pis_u': {
            'variables': ['u'],
            'class': cb.ShapeDimDim,
        },
        'corrs_rs': {  # eq. (43)
            'requires': ['pis_u'],
            'ebcs': ['fixed_u', 'fixed_p'],
            'epbcs': periodic_all,
            'equations': {
                'balance_of_forces':
                """    dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                    - dw_biot.i.Ym(mat_he.BI, v, p)
                = - dw_nonsym_elastic.i.Y(mat_he.A, v, Piu)""",
                'mass equilibrium':
                """  - dw_biot.i.Ym(mat_he.BI, u, q)
                -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                    = dw_biot.i.Ym(mat_he.BI, Piu, q)""" % dt,
            },
            'set_variables': [('Piu', 'pis_u', 'u')],
            'class': cb.CorrDimDim,
            'save_name': 'corrs_hp_rs',
            'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
        },
        'corrs_p': {  #  particular response, eq. (45)
            'requires': ['press_m'],
            'ebcs': ['fixed_u', 'fixed_p'],
            'epbcs': periodic_all,
            'equations': {
                'balance_of_forces':
                """    dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                    - dw_biot.i.Ym(mat_he.BI, v, p)
                = - dw_lin_prestress.i.Y(mat_he.S, v)""",
                'mass equilibrium':
                """  - dw_biot.i.Ym(mat_he.BI, u, q)
                -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                = %e * dw_diffusion.i.Ym(mat_he.KH, q, Pip)
                + %e * dw_diffusion.i.Ym(mat_he.dK, q, Pip) """ % (dt, dt, dt),  # !!! d3(Pip, q)
            },
            'class': cb.CorrOne,
            'set_variables': [('Pip', 'press_m', 'p')],
            'save_name': 'corrs_hp_p',
            'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
        },
        'press_m': {
            'variable': 'p',
            'class': CorrStatePressureM,
            'save_name': 'corrs_hp_press_m',
        },
    }

    for ich in chs:
        lab = '%d' % ich
        Yc = 'Yc' + lab

        fields.update({
            'pressure' + lab: ('real', 'scalar', Yc, approx_p),
        })

        variables.update({
            'p' + lab: ('unknown field', 'pressure' + lab),
            'q' + lab: ('test field', 'pressure' + lab, 'p' + lab),
            'Pip' + lab: ('parameter field', 'pressure' + lab, 'p' + lab),
            'Pi1p' + lab: ('parameter field', 'pressure' + lab, '(set-to-None)'),
            'Pi2p' + lab: ('parameter field', 'pressure' + lab, '(set-to-None)'),
            'P' + lab: ('parameter field', 'pressure' + lab, '(set-to-None)'),
            'ls' + lab: ('unknown field', 'pressure' + lab),
            'lv' + lab: ('test field', 'pressure' + lab, 'ls' + lab),
        })

        epbcs.update({
            'periodic_px' + lab: (['Left', 'Right'],
                                {'p%s.0' % lab: 'p%s.0' % lab},
                                'match_x_plane'),
        })

        periodic_all_p = ['periodic_px' + lab, 'periodic_py' + lab]

        if dim == 3:
            epbcs.update({
                'periodic_py' + lab: (['Near', 'Far'],
                                    {'p%s.0' % lab: 'p%s.0' % lab},
                                    'match_y_plane'),
                'periodic_pz' + lab: (['Bottom', 'Top'],
                                    {'p%s.0' % lab: 'p%s.0' % lab},
                                    'match_z_plane'),
            })

            periodic_all_p += ['periodic_pz' + lab]
        else:
            epbcs.update({
                'periodic_py' + lab: (['Bottom', 'Top'],
                                    {'p%s.0' % lab: 'p%s.0' % lab},
                                    'match_y_plane'),
            })


        regions.update({
            Yc: 'cells of group %d' % (ich + 1),
            'Gamma' + lab: ('r.Yc%s *s r.Ym' % lab, 'facet', 'Ym'),
        })

        ename = 'fixed_p%s_%s_1' % (lab, lab)
        ebcs[ename] = ('Gamma' + lab, {'p.0': 1.})
        fixed_p_01 = [ename]

        chs2 = chs[:]
        chs2.remove(ich)
        for ich2 in chs2:
            lab2 = '%d' % ich2
            ename = 'fixed_p%s_%s_0' % (lab, lab2)
            ebcs[ename] = ('Gamma' + lab2, {'p.0': 0.})
            fixed_p_01.append(ename)

        lname = 'imv' + lab
        lcbcs[lname] = (Yc, {'ls%s.0' % lab: None}, None,
                        'integral_mean_value')

        coefs.update({
            'B%s_1' % lab: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_' + lab],
                'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                'set_variables': [('Pi1p', 'corrs_' + lab, 'p'),
                                ('Pi1u', 'pis_u', 'u')],
                'class': cb.CoefNonSym,
            },
            'B%s_2' % lab: {
                'status': 'auxiliary',
                'requires': ['pis_u'],
                'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
                'set_variables': [('Pi1u', 'pis_u', 'u')],
                'class': cb.CoefNonSym,
            },
            'B%s_3' % lab: {
                'status': 'auxiliary',
                'requires': ['pis_u', 'corrs_' + lab],
                'expression': 'dw_nonsym_elastic.i.Y(mat_he.A, Pi1u, Pi2u)',
                'set_variables': [('Pi1u', 'pis_u', 'u'),
                                ('Pi2u', 'corrs_' + lab, 'u')],
                'class': cb.CoefNonSym,
            },
            'B' + lab: {  # The Biot poroelasticity tensor, eq. (52)
                'requires': ['c.B%s_%d' % (lab, ii + 1) for ii in range(3)],
                'expression': 'c.B%s_1 + c.B%s_2 - c.B%s_3' % ((lab,) * 3),
                'class': cb.CoefEval,
            },
            'C' + lab: {  # channel permeability, eq. (55)
                'requires': ['pis_p' + lab, 'corrs_eta' + lab],
                'expression': 'dw_diffusion.i.Yc%s(mat_he.KH, Pi1p%s, Pi2p%s)'\
                            % ((lab,) * 3),
                'set_variables': [('Pi1p' + lab,
                                ('pis_p' + lab, 'corrs_eta' + lab), 'p' + lab),
                                ('Pi2p' + lab,
                                ('pis_p' + lab, 'corrs_eta' + lab), 'p' + lab)],
                'class': cb.CoefDimDim,
            },
            'Z%s_1' % lab: {
                'status': 'auxiliary',
                'requires': ['corrs_p'],
                'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
                'set_variables': [('Pi1u', 'corrs_p', 'u')],
                'class': cb.CoefOne,
            },
            'Z%s_2' % lab: {
                'status': 'auxiliary',
                'requires': ['corrs_' + lab, 'corrs_p'],
                'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                'set_variables': [('Pi1u', 'corrs_p', 'u'),
                                ('Pi1p', 'corrs_' + lab, 'p')],
                'class': cb.CoefOne,
            },
            'Z%s_3' % lab: {
                'status': 'auxiliary',
                'requires': ['corrs_' + lab, 'corrs_p', 'press_m'],
                'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
                'set_variables': [('Pi1p', ('corrs_p', 'press_m'), 'p'),
                                ('Pi2p', 'corrs_' + lab, 'p')],
                'class': cb.CoefOne,
            },
            'Z' + lab: {  # effective discharge, eq. (58)
                'requires': ['c.Z%s_%d' % (lab, ii + 1) for ii in range(3)],
                'expression': 'c.Z%s_1/%e + c.Z%s_2/%e + c.Z%s_3'\
                    % (lab, dt, lab, dt, lab),
                'class': cb.CoefEval,
            },
            'g%s' % lab: {  # effective discharge, eq. (58)
                'requires': ['pis_p' + lab, 'press_' + lab, 'corrs_p' + lab],
                'expression': """dw_diffusion.i.Yc%s(mat_he.KH, Pi1p%s, Pi2p%s)"""\
                            % ((lab,) * 3),
                'set_variables': [('Pi1p' + lab, ('press_' + lab, 'corrs_p' + lab),
                                'p' + lab),
                                ('Pi2p' + lab, 'pis_p' + lab, 'p' + lab)],
                'class': cb.CoefDim,
            },
        })

        requirements.update({
            'corrs_' + lab: {  # eq. (44)
                'requires': [],
                'ebcs': ['fixed_u'] + fixed_p_01,
                'epbcs': periodic_all,
                'equations': {
                    'balance_of_forces':
                    """   dw_nonsym_elastic.i.Y(mat_he.A, v, u)
                        - dw_biot.i.Ym(mat_he.BI, v, p)
                        = dw_lin_prestress.i.Yc%s(mat_he.BI, v)""" % lab,
                    'mass equilibrium':
                    """ - dw_biot.i.Ym(mat_he.BI, u, q)
                    -%e * dw_diffusion.i.Ym(mat_he.KH, q, p)
                        = 0""" % dt,
                },
                'class': cb.CorrOne,
                'save_name': 'corrs_hp_' + lab,
                'solvers': {'ls': 'ls', 'nls': 'nls1', 'ts': None},
            },
            'pis_p' + lab: {
                'variables': ['p' + lab],
                'class': cb.ShapeDim,
            },
            'corrs_eta' + lab: {  # channel flow correctors, eq. (46)
                'requires': ['pis_p' + lab],
                'epbcs': periodic_all_p,
                'ebcs': [],
                'lcbcs': [lname],
                'equations': {
                    'eq':
                    """   dw_diffusion.i.Yc%s(mat_he.KH, q%s, p%s)
                        + dw_volume_dot.i.Yc%s(q%s, ls%s)
                        =
                        - dw_diffusion.i.Yc%s(mat_he.KH, q%s, Pip%s)"""\
                        % ((lab,) * 9),
                    'eq_imv':
                        'dw_volume_dot.i.Yc%s(lv%s, p%s) = 0' % ((lab,) * 3),
                },
                'class': cb.CorrDim,
                'set_variables': [('Pip' + lab, 'pis_p' + lab, 'p' + lab)],
                'save_name': 'corrs_hp_eta_' + lab,
                'solvers': {'ls': 'ls', 'nls': 'nls2', 'ts': None},
            },
            'corrs_p' + lab: {  # particular response, eq. (47)
                'requires': ['press_' + lab],
                'ebcs': [],
                'epbcs': periodic_all_p,
                'lcbcs': [lname],
                'equations': {
                    'eq':
                    """   dw_diffusion.i.Yc%s(mat_he.KH, q%s, p%s)
                        + dw_volume_dot.i.Yc%s(q%s, ls%s)
                        =
                        dw_diffusion.i.Yc%s(mat_he.dK, q%s, Pip%s)
                        - dw_diffusion.i.Yc%s(mat_he.H, q%s, Pip%s)"""\
                        % ((lab,) * 12),
                    'eq_imv':
                        'dw_volume_dot.i.Yc%s(lv%s, p%s) = 0' % ((lab,) * 3),

                },
                'class': cb.CorrOne,
                'set_variables': [('Pip' + lab, 'press_' + lab, 'p' + lab)],
                'save_name': 'corrs_hp_p_' + lab,
                'solvers': {'ls': 'ls', 'nls': 'nls2', 'ts': None},
            },
            'press_' + lab: {
                'variable': 'p' + lab,
                'class': CorrStatePressureCh,
                'save_name': 'corrs_hp_press_' + lab,
            },
        })

        for ich2 in chs:
            lab2 = '%d' % ich2
            lab12 = lab + lab2

            coefs.update({
                'G%s_1' % lab12: {
                    'status': 'auxiliary',
                    'requires': ['corrs_' + lab2],
                    'expression': 'dw_lin_prestress.i.Yc%s(mat_he.BI, Pi1u)' % lab,
                    'set_variables': [('Pi1u', 'corrs_' + lab2, 'u')],
                    'class': cb.CoefOne,
                },
                'G%s_2' % lab12: {
                    'status': 'auxiliary',
                    'requires': ['corrs_' + lab, 'corrs_' + lab2],
                    'expression': 'dw_biot.i.Ym(mat_he.BI, Pi1u, Pi1p)',
                    'set_variables': [('Pi1u', 'corrs_' + lab, 'u'),
                                    ('Pi1p', 'corrs_' + lab2, 'p')],
                    'class': cb.CoefOne,
                },
                'G%s_3' % lab12: {
                    'status': 'auxiliary',
                    'requires': ['corrs_' + lab, 'corrs_' + lab2],
                    'expression': 'dw_diffusion.i.Ym(mat_he.KH, Pi1p, Pi2p)',
                    'set_variables': [('Pi1p', 'corrs_' + lab, 'p'),
                                    ('Pi2p', 'corrs_' + lab2, 'p')],
                    'class': cb.CoefOne,
                },
                'G%s' % lab12: {  # perfusion coefficient , eq. (57)
                    'requires': ['c.G%s_%d' % (lab12, ii + 1) for ii in range(3)],
                    'expression': 'c.G%s_1/%e + c.G%s_2/%e + c.G%s_3'\
                        % (lab12, dt, lab12, dt, lab12),
                    'class': cb.CoefEval,
                },
            })

    solvers = {
        'ls': ('ls.mumps', {
            'memory_relaxation': 50,
        }),
        'nls2': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-16,
            'eps_r': 1e-3,
            'problem': 'nonlinear',
        }),
        'nls1': ('nls.newton', {
            'i_max': 1,
            'eps_a': 1e-6,
            'eps_r': 1e-3,
            'problem': 'nonlinear',
        }),

    }

    return locals()
