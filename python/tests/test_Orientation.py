import pytest
import numpy as np
from itertools import permutations
from matplotlib import pyplot as plt
from PIL import Image

from damask import Rotation
from damask import Orientation
from damask import Table
from damask import util
from damask import grid_filters
from damask import _crystal

crystal_families = set(_crystal.lattice_symmetries.values())


@pytest.fixture
def res_path(res_path_base):
    """Directory containing testing resources."""
    return res_path_base/'Orientation'

@pytest.fixture
def set_of_rodrigues(set_of_quaternions):
    return Rotation(set_of_quaternions).as_Rodrigues_vector()


class TestOrientation:

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[None,5,(4,6)])
    def test_equal(self,family,shape):
        R = Rotation.from_random(shape)
        assert Orientation(R,family=family) == Orientation(R,family=family) if shape is None else \
              (Orientation(R,family=family) == Orientation(R,family=family)).all()


    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[None,5,(4,6)])
    def test_unequal(self,family,shape):
        R = Rotation.from_random(shape)
        assert not ( Orientation(R,family=family) != Orientation(R,family=family) if shape is None else \
                    (Orientation(R,family=family) != Orientation(R,family=family)).any())

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[None,5,(4,6)])
    def test_close(self,family,shape):
        R = Orientation.from_random(family=family,shape=shape)
        assert R.isclose(R.reduced).all() and R.allclose(R.reduced)

    @pytest.mark.parametrize('a,b',[
                                    (dict(rotation=[1,0,0,0],family='triclinic'),
                                     dict(rotation=[0.5,0.5,0.5,0.5],family='triclinic')),

                                    (dict(rotation=[1,0,0,0],family='cubic'),
                                     dict(rotation=[1,0,0,0],family='hexagonal')),
                                    ])
    def test_unequal_family(self,a,b):
        assert Orientation(**a) != Orientation(**b)

    @pytest.mark.parametrize('a,b',[
                                    (dict(rotation=[1,0,0,0],lattice='cF',a=1),
                                     dict(rotation=[1,0,0,0],lattice='cF',a=2)),
                                    ])
    def test_unequal_lattice(self,a,b):
        assert Orientation(**a) != Orientation(**b)

    @pytest.mark.parametrize('kwargs',[
                                        dict(lattice='aP',                  alpha=np.pi/4,beta=np.pi/3,             ),
                                        dict(lattice='mP',            c=1.2,alpha=np.pi/4,             gamma=np.pi/2),
                                        dict(lattice='oP',            c=1.2,alpha=np.pi/4,                          ),
                                        dict(lattice='oS',a=1.0,      c=2.0,alpha=np.pi/2,beta=np.pi/3,             ),
                                        dict(lattice='tP',a=1.0,b=1.2,                                              ),
                                        dict(lattice='tI',                  alpha=np.pi/3,                          ),
                                        dict(lattice='hP',                                             gamma=np.pi/2),
                                        dict(lattice='cI',a=1.0,      c=2.0,alpha=np.pi/2,beta=np.pi/2,             ),
                                        dict(lattice='cF',                                beta=np.pi/3,             ),
                                      ])
    def test_invalid_init(self,kwargs):
        with pytest.raises(ValueError):
            Orientation(**kwargs)

    @pytest.mark.parametrize('invalid_family',[None,'fcc','bcc','hello'])
    def test_invalid_family_init(self,invalid_family):
        with pytest.raises(KeyError):
            Orientation(family=invalid_family)

    @pytest.mark.parametrize('invalid_lattice',[None,'fcc','bcc','hello'])
    def test_invalid_lattice_init(self,invalid_lattice):
        with pytest.raises(KeyError):
            Orientation(lattice=invalid_lattice)

    @pytest.mark.parametrize('kwargs',[
                                        dict(lattice='aP',a=1.0,b=1.1,c=1.2,alpha=np.pi/4,beta=np.pi/3,gamma=np.pi/2),
                                        dict(lattice='mP',a=1.0,b=1.1,c=1.2,              beta=np.pi/3              ),
                                        dict(lattice='oS',a=1.0,b=1.1,c=1.2,                                        ),
                                        dict(lattice='tI',a=1.0,      c=1.2,                                        ),
                                        dict(lattice='hP',a=1.0                                                     ),
                                        dict(lattice='cI',a=1.0,                                                    ),
                                      ])
    def test_repr(self,kwargs):
        o = Orientation.from_random(**kwargs)
        assert isinstance(o.__repr__(),str)

    @pytest.mark.parametrize('kwargs',[
                                        dict(lattice='aP',a=1.0,b=1.1,c=1.2,alpha=np.pi/4,beta=np.pi/3,gamma=np.pi/2),
                                        dict(lattice='mP',a=1.0,b=1.1,c=1.2,              beta=np.pi/3              ),
                                        dict(lattice='oS',a=1.0,b=1.1,c=1.2,                                        ),
                                        dict(lattice='tI',a=1.0,      c=1.2,                                        ),
                                        dict(lattice='hP',a=1.0                                                     ),
                                        dict(lattice='cI',a=1.0,                                                    ),
                                      ])
    def test_copy(self,kwargs):
        o = Orientation.from_random(**kwargs)
        p = o.copy(rotation=Rotation.from_random())
        assert o != p

    def test_from_quaternion(self):
        assert np.all(Orientation.from_quaternion(q=np.array([1,0,0,0]),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_Euler_angles(self):
        assert np.all(Orientation.from_Euler_angles(phi=np.zeros(3),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_axis_angle(self):
        assert np.all(Orientation.from_axis_angle(n_omega=[1,0,0,0],family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_basis(self):
        assert np.all(Orientation.from_basis(basis=np.eye(3),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_matrix(self):
        assert np.all(Orientation.from_matrix(R=np.eye(3),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_Rodrigues_vector(self):
        assert np.all(Orientation.from_Rodrigues_vector(rho=np.array([0,0,1,0]),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_homochoric(self):
        assert np.all(Orientation.from_homochoric(h=np.zeros(3),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_cubochoric(self):
        assert np.all(Orientation.from_cubochoric(x=np.zeros(3),family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_spherical_component(self):
        assert np.all(Orientation.from_spherical_component(center=Rotation(),
                                                           sigma=0.0,shape=1,family='triclinic').as_matrix()
                   == np.eye(3))

    def test_from_fiber_component(self):
        crystal = np.random.rand(2) * [180,360]
        sample = np.random.rand(2) * [180,360]
        r = Rotation.from_fiber_component(crystal=crystal,sample=sample,
                                          sigma=0.0,shape=1,rng_seed=0)
        assert np.all(Orientation.from_fiber_component(crystal=crystal,sample=sample,
                                                       sigma=0.0,shape=None,rng_seed=0,lattice='cI').quaternion
                   == r.quaternion)

    @pytest.mark.parametrize('crystal,sample,direction,color',[([np.pi/4,0],[np.pi/2,0],[1,0,0],[0,1,0]),
                                                               ([np.arccos(3**(-.5)),np.pi/4,0],[0,0],[0,0,1],[0,0,1])])
    def test_fiber_IPF(self,crystal,sample,direction,color):
        fiber = Orientation.from_fiber_component(crystal=crystal,sample=sample,family='cubic',shape=200)
        assert np.allclose(fiber.IPF_color(direction),color)


    @pytest.mark.parametrize('kwargs',[
                                        dict(lattice='aP',a=1.0,b=1.1,c=1.2,alpha=np.pi/4.5,beta=np.pi/3.5,gamma=np.pi/2.5),
                                        dict(lattice='mP',a=1.0,b=1.1,c=1.2,                beta=np.pi/3.5),
                                        dict(lattice='oS',a=1.0,b=1.1,c=1.2,),
                                        dict(lattice='tI',a=1.0,      c=1.2,),
                                        dict(lattice='hP',a=1.0             ),
                                        dict(lattice='cI',a=1.0,            ),
                                      ])
    def test_from_directions(self,kwargs):
        for a,b in np.random.random((10,2,3)):
            c = np.cross(b,a)
            if np.allclose(c,0): continue
            o = Orientation.from_directions(uvw=a,hkl=c,**kwargs)
            x = o.to_pole(uvw=a)
            z = o.to_pole(hkl=c)
            assert np.isclose(np.dot(x,np.array([1,0,0])),1) \
               and np.isclose(np.dot(z,np.array([0,0,1])),1)

    @pytest.mark.parametrize('function',[Orientation.from_random,
                                         Orientation.from_quaternion,
                                         Orientation.from_Euler_angles,
                                         Orientation.from_axis_angle,
                                         Orientation.from_basis,
                                         Orientation.from_matrix,
                                         Orientation.from_Rodrigues_vector,
                                         Orientation.from_homochoric,
                                         Orientation.from_cubochoric,
                                         Orientation.from_spherical_component,
                                         Orientation.from_fiber_component,
                                         Orientation.from_directions])
    def test_invalid_from(self,function):
        with pytest.raises(TypeError):
            function(c=.1,degrees=True,invalid=66)

    def test_negative_angle(self):
        with pytest.raises(ValueError):
            Orientation(lattice='aP',a=1,b=2,c=3,alpha=45,beta=45,gamma=-45,degrees=True)           # noqa

    def test_excess_angle(self):
        with pytest.raises(ValueError):
            Orientation(lattice='aP',a=1,b=2,c=3,alpha=45,beta=45,gamma=90.0001,degrees=True)       # noqa

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('angle',[10,20,30,40])
    def test_average(self,angle,family):
        o = Orientation.from_axis_angle(family=family,n_omega=[[0,0,1,10],[0,0,1,angle]],degrees=True)
        avg_angle = o.average().as_axis_angle(degrees=True,pair=True)[1]
        assert np.isclose(avg_angle,10+(angle-10)/2.)

    @pytest.mark.parametrize('family',crystal_families)
    def test_reduced_equivalent(self,family):
        i = Orientation(family=family)
        o = Orientation.from_random(family=family)
        eq = o.equivalent
        FZ = np.argmin(abs(eq.misorientation(i.broadcast_to(len(eq))).as_axis_angle(pair=True)[1]))
        assert o.reduced == eq[FZ]

    @pytest.mark.parametrize('family',crystal_families)
    def test_reduced_corner_cases(self,family):
        # test whether there is always exactly one sym-eq rotation that falls into the FZ
        N = np.random.randint(10,40)
        size = np.ones(3)*np.pi**(2./3.)
        grid = grid_filters.coordinates0_node([N+1,N+1,N+1],size,-size*.5)
        evenly_distributed = Orientation.from_cubochoric(x=grid,family=family)
        assert evenly_distributed.shape == evenly_distributed.reduced.shape

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('N',[1,8,32])
    def test_disorientation(self,family,N):
        o = Orientation.from_random(family=family,shape=N)
        p = Orientation.from_random(family=family,shape=N)

        d,ops = o.disorientation(p,return_operators=True)

        for n in range(N):
            assert np.allclose(d[n].as_quaternion(),
                               o[n].equivalent[ops[n][0]]
                                   .misorientation(p[n].equivalent[ops[n][1]])
                                   .as_quaternion()) \
                or np.allclose((~d)[n].as_quaternion(),
                                  o[n].equivalent[ops[n][0]]
                                      .misorientation(p[n].equivalent[ops[n][1]])
                                      .as_quaternion())

    @pytest.mark.parametrize('family',crystal_families)
    def test_disorientation360(self,family):
        o_1 = Orientation(Rotation(),family=family)
        o_2 = Orientation.from_Euler_angles(family=family,phi=[360,0,0],degrees=True)
        assert np.allclose((o_1.disorientation(o_2)).as_matrix(),np.eye(3))

    @pytest.mark.parametrize('color',[{'label':'red',  'RGB':[1,0,0],'direction':[0,0,1]},
                                      {'label':'green','RGB':[0,1,0],'direction':[0,1,1]},
                                      {'label':'blue', 'RGB':[0,0,1],'direction':[1,1,1]}])
    @pytest.mark.parametrize('proper',[True,False])
    def test_IPF_cubic(self,color,proper):
        cube = Orientation(family='cubic')
        for direction in set(permutations(np.array(color['direction']))):
            assert np.allclose(np.array(color['RGB']),
                               cube.IPF_color(vector=np.array(direction),proper=proper))

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('proper',[True,False])
    def test_IPF_equivalent(self,set_of_quaternions,family,proper):
        direction = np.random.random(3)*2.0-1.0
        o = Orientation(rotation=set_of_quaternions,family=family).equivalent
        color = o.IPF_color(vector=direction,proper=proper)
        assert np.allclose(np.broadcast_to(color[0,...],color.shape),color)

    @pytest.mark.parametrize('relation',[None,'Peter','Paul'])
    def test_unknown_relation(self,relation):
        with pytest.raises(KeyError):
            Orientation(lattice='cF').related(relation)                                             # noqa

    @pytest.mark.parametrize('relation,lattice,a,b,c,alpha,beta,gamma',
                            [
                             ('Bain',   'aP',0.5,2.0,3.0,0.8,0.5,1.2),
                             ('KS',     'mP',1.0,2.0,3.0,np.pi/2,0.5,np.pi/2),
                             ('Pitsch', 'oI',0.5,1.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('Burgers','tP',0.5,0.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('GT',     'hP',1.0,None,1.6,np.pi/2,np.pi/2,2*np.pi/3),
                             ('Burgers','cF',1.0,1.0,None,np.pi/2,np.pi/2,np.pi/2),
                            ])
    def test_unknown_relation_lattice(self,relation,lattice,a,b,c,alpha,beta,gamma):
        with pytest.raises(KeyError):
            Orientation(lattice=lattice,
                        a=a,b=b,c=c,
                        alpha=alpha,beta=beta,gamma=gamma).related(relation)                        # noqa

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('proper',[True,False])
    def test_in_SST(self,family,proper):
        assert Orientation(family=family).in_SST(np.zeros(3),proper)

    @pytest.mark.parametrize('function',['in_SST','IPF_color'])
    def test_invalid_argument(self,function):
        o = Orientation(family='cubic')                                                             # noqa
        with pytest.raises(ValueError):
            eval(f'o.{function}(np.ones(4))')

    @pytest.mark.parametrize('model',['Bain','KS','GT','GT_prime','NW','Pitsch'])
    @pytest.mark.parametrize('lattice',['cF','cI'])
    def test_relationship_reference(self,update,res_path,model,lattice):
        reference = res_path/f'{lattice}_{model}.txt'
        o = Orientation(lattice=lattice)
        eu = o.related(model).as_Euler_angles(degrees=True)
        if update:
            coords = np.array([(1,i+1) for i,x in enumerate(eu)])
            Table({'Eulers':(3,)},eu).set('pos',coords).save(reference)
        assert np.allclose(eu,Table.load(reference).get('Eulers'))

    def test_basis_real(self):
        for gamma in np.random.random(2**8)*np.pi:
            basis = np.tril(np.random.random((3,3))+1e-6)
            basis[1,:2] = basis[1,1]*np.array([np.cos(gamma),np.sin(gamma)])
            basis[2,:2] = basis[2,:2]*2-1
            lengths = np.linalg.norm(basis,axis=-1)
            cosines = np.roll(np.einsum('ij,ij->i',basis,np.roll(basis,1,axis=0))/lengths/np.roll(lengths,1),1)
            o = Orientation.from_random(lattice='aP',
                            **dict(zip(['a','b','c'],lengths)),
                            **dict(zip(['alpha','beta','gamma'],np.arccos(cosines))),
                            )
            assert np.allclose(o.to_frame(uvw=np.eye(3)),basis), 'Lattice basis disagrees with initialization'

    @pytest.mark.parametrize('lattice,a,b,c,alpha,beta,gamma',
                            [
                             ('aP',0.5,2.0,3.0,0.8,0.5,1.2),
                             ('mP',1.0,2.0,3.0,np.pi/2,0.5,np.pi/2),
                             ('oI',0.5,1.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('tP',0.5,0.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('hP',1.0,1.0,1.6,np.pi/2,np.pi/2,2*np.pi/3),
                             ('cF',1.0,1.0,1.0,np.pi/2,np.pi/2,np.pi/2),
                            ])
    @pytest.mark.parametrize('kw',['uvw','hkl'])
    @pytest.mark.parametrize('with_symmetry',[False,True])
    @pytest.mark.parametrize('shape',[None,1,(12,24)])
    @pytest.mark.parametrize('vector',[
                                        np.random.random(     3 ),
                                        np.random.random(  (4,3)),
                                        np.random.random((4,8,3)),
                                      ])
    def test_to_pole(self,shape,lattice,a,b,c,alpha,beta,gamma,vector,kw,with_symmetry):
        o = Orientation.from_random(shape=shape,
                                    lattice=lattice,
                                    a=a,b=b,c=c,
                                    alpha=alpha,beta=beta,gamma=gamma)
        assert o.to_pole(**{kw:vector,'with_symmetry':with_symmetry}).shape \
            == util.shapeblender(o.shape,vector.shape[:-1]) \
             + (o.symmetry_operations.shape if with_symmetry else ()) \
             + vector.shape[-1:]

    @pytest.mark.parametrize('lattice',['hP','cI','cF']) #tI not included yet
    def test_Schmid(self,update,res_path,lattice):
        O = Orientation(lattice=lattice)                                                            # noqa
        for mode in ['slip','twin']:
            reference = res_path/f'{lattice}_{mode}.txt'
            P = O.Schmid(N_slip='*') if mode == 'slip' else O.Schmid(N_twin='*')
            if update:
                Table({'Schmid':(3,3,)},P.reshape(-1,9)).save(reference)
            assert np.allclose(P,Table.load(reference).get('Schmid'))

    def test_Schmid_invalid(self):
        with pytest.raises(KeyError):
            Orientation(lattice='fcc').Schmid()


### vectorization tests ###

    @pytest.mark.parametrize('lattice',['hP','cI','cF']) # tI not included yet
    def test_Schmid_vectorization(self,lattice):
        O = Orientation.from_random(shape=4,lattice=lattice)                                        # noqa
        for mode in ['slip','twin']:
            Ps = O.Schmid(N_slip='*') if mode == 'slip' else O.Schmid(N_twin='*')
            for i in range(4):
                P = O[i].Schmid(N_slip='*') if mode == 'slip' else O[i].Schmid(N_twin='*')
                assert np.allclose(P,Ps[:,i])

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[(1),(2,3),(4,3,2)])
    def test_reduced_vectorization(self,family,shape):
        o = Orientation.from_random(family=family,shape=shape)
        for r, theO in zip(o.reduced.flatten(),o.flatten()):
            assert r == theO.reduced


    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[(1),(2,3),(4,3,2)])
    @pytest.mark.parametrize('vector',np.array([[1,0,0],[1,2,3],[-1,1,-1]]))
    @pytest.mark.parametrize('proper',[True,False])
    def test_to_SST_vectorization(self,family,shape,vector,proper):
        o = Orientation.from_random(family=family,shape=shape)
        for r, theO in zip(o.to_SST(vector=vector,proper=proper).reshape((-1,3)),o.flatten()):
            assert np.allclose(r,theO.to_SST(vector=vector,proper=proper))

    @pytest.mark.parametrize('proper',[True,False])
    @pytest.mark.parametrize('family',crystal_families)
    def test_in_SST_vectorization(self,family,proper):
        vecs = np.random.rand(20,4,3)
        result = Orientation(family=family).in_SST(vecs,proper).flatten()
        for r,v in zip(result,vecs.reshape((-1,3))):
            assert np.all(r == Orientation(family=family).in_SST(v,proper))

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('shape',[(1),(2,3),(4,3,2)])
    @pytest.mark.parametrize('vector',np.array([[1,0,0],[1,2,3],[-1,1,-1]]))
    @pytest.mark.parametrize('proper',[True,False])
    @pytest.mark.parametrize('in_SST',[True,False])
    def test_IPF_color_vectorization(self,family,shape,vector,proper,in_SST):
        o = Orientation.from_random(family=family,shape=shape)
        for r, theO in zip(o.IPF_color(vector,in_SST=in_SST,proper=proper).reshape((-1,3)),o.flatten()):
            assert np.allclose(r,theO.IPF_color(vector,in_SST=in_SST,proper=proper))

    @pytest.mark.parametrize('family',crystal_families)
    def test_in_FZ_vectorization(self,set_of_rodrigues,family):
        result = Orientation.from_Rodrigues_vector(rho=set_of_rodrigues.reshape((-1,4,4)),family=family).in_FZ.reshape(-1)
        for r,rho in zip(result,set_of_rodrigues[:len(result)]):
            assert r == Orientation.from_Rodrigues_vector(rho=rho,family=family).in_FZ

    @pytest.mark.parametrize('family',crystal_families)
    def test_in_disorientation_FZ_vectorization(self,set_of_rodrigues,family):
        result = Orientation.from_Rodrigues_vector(rho=set_of_rodrigues.reshape((-1,4,4)),
                                            family=family).in_disorientation_FZ.reshape(-1)
        for r,rho in zip(result,set_of_rodrigues[:len(result)]):
            assert r == Orientation.from_Rodrigues_vector(rho=rho,family=family).in_disorientation_FZ

    @pytest.mark.parametrize('model',['Bain','KS','GT','GT_prime','NW','Pitsch'])
    @pytest.mark.parametrize('lattice',['cF','cI'])
    def test_relationship_vectorization(self,set_of_quaternions,lattice,model):
        r = Orientation(rotation=set_of_quaternions[:200].reshape((50,4,4)),lattice=lattice).related(model)
        for i in range(200):
            assert (r.reshape((-1,200))[:,i] == Orientation(set_of_quaternions[i],lattice=lattice).related(model)).all()

### blending tests ###

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('left,right',[
                                    ((2,3,2),(2,3,2)),
                                    ((2,2),(4,4)),
                                    ((3,1),(1,3)),
                                    (None,None),
                                   ])
    def test_disorientation_blending(self,family,left,right):
        o = Orientation.from_random(family=family,shape=left)
        p = Orientation.from_random(family=family,shape=right)
        blend = util.shapeblender(o.shape,p.shape)
        for loc in np.random.randint(0,blend,(10,len(blend))):
            l = () if  left is None else tuple(np.minimum(np.array(left )-1,loc[:len(left)]))
            r = () if right is None else tuple(np.minimum(np.array(right)-1,loc[-len(right):]))
            assert o[l].disorientation(p[r]).isclose(o.disorientation(p)[tuple(loc)])

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('left,right',[
                                    ((2,3,2),(2,3,2)),
                                    ((2,2),(4,4)),
                                    ((3,1),(1,3)),
                                    (None,(3,)),
                                    (None,()),
                                   ])
    def test_IPF_color_blending(self,family,left,right):
        o = Orientation.from_random(family=family,shape=left)
        v = np.random.random(right+(3,))
        blend = util.shapeblender(o.shape,v.shape[:-1])
        for loc in np.random.randint(0,blend,(10,len(blend))):
            l = () if  left is None else tuple(np.minimum(np.array(left )-1,loc[:len(left)]))
            r = () if right is None else tuple(np.minimum(np.array(right)-1,loc[-len(right):]))
            assert np.allclose(o[l].IPF_color(v[r]),
                               o.IPF_color(v)[tuple(loc)])

    @pytest.mark.parametrize('family',crystal_families)
    @pytest.mark.parametrize('left,right',[
                                    ((2,3,2),(2,3,2)),
                                    ((2,2),(4,4)),
                                    ((3,1),(1,3)),
                                    (None,(3,)),
                                   ])
    def test_to_SST_blending(self,family,left,right):
        o = Orientation.from_random(family=family,shape=left)
        v = np.random.random(right+(3,))
        blend = util.shapeblender(o.shape,v.shape[:-1])
        for loc in np.random.randint(0,blend,(10,len(blend))):
            l = () if  left is None else tuple(np.minimum(np.array(left )-1,loc[:len(left)]))
            r = () if right is None else tuple(np.minimum(np.array(right)-1,loc[-len(right):]))
            assert np.allclose(o[l].to_SST(v[r]),
                               o.to_SST(v)[tuple(loc)])

    @pytest.mark.parametrize('lattice,a,b,c,alpha,beta,gamma',
                            [
                             ('aP',0.5,2.0,3.0,0.8,0.5,1.2),
                             ('mP',1.0,2.0,3.0,np.pi/2,0.5,np.pi/2),
                             ('oI',0.5,1.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('tP',0.5,0.5,3.0,np.pi/2,np.pi/2,np.pi/2),
                             ('hP',1.0,1.0,1.6,np.pi/2,np.pi/2,2*np.pi/3),
                             ('cF',1.0,1.0,1.0,np.pi/2,np.pi/2,np.pi/2),
                            ])
    @pytest.mark.parametrize('left,right',[
                                    ((2,3,2),(2,3,2)),
                                    ((2,2),(4,4)),
                                    ((3,1),(1,3)),
                                    (None,(3,)),
                                   ])
    def test_to_pole_blending(self,lattice,a,b,c,alpha,beta,gamma,left,right):
        o = Orientation.from_random(shape=left,
                                    lattice=lattice,
                                    a=a,b=b,c=c,
                                    alpha=alpha,beta=beta,gamma=gamma)
        v = np.random.random(right+(3,))
        blend = util.shapeblender(o.shape,v.shape[:-1])
        for loc in np.random.randint(0,blend,(10,len(blend))):
            l = () if  left is None else tuple(np.minimum(np.array(left )-1,loc[:len(left)]))
            r = () if right is None else tuple(np.minimum(np.array(right)-1,loc[-len(right):]))
        assert np.allclose(o[l].to_pole(uvw=v[r]),
                           o.to_pole(uvw=v)[tuple(loc)])

    def test_mul_invalid(self):
        with pytest.raises(TypeError):
            Orientation.from_random(lattice='cF')*np.ones(3)

    @pytest.mark.parametrize('OR',['KS','NW','GT','GT_prime','Bain','Pitsch'])
    @pytest.mark.parametrize('pole',[[0,0,1],[0,1,1],[1,1,1]])
    def test_OR_plot(self,update,res_path,tmp_path,OR,pole):
        # https://doi.org/10.3390/cryst13040663 for comparison
        O = Orientation(lattice='cF')
        poles = O.related(OR).to_pole(uvw=pole,with_symmetry=True).reshape(-1,3)
        points = util.project_equal_area(poles,'z')

        fig, ax = plt.subplots()
        c = plt.Circle((0,0),1, color='k',fill=False)
        ax.add_patch(c)
        ax.scatter(points[:,0],points[:,1])
        ax.set_aspect('equal', 'box')
        fname=f'{OR}-{"".join(map(str,pole))}.png'
        plt.axis('off')
        plt.savefig(tmp_path/fname)
        if update: plt.savefig(res_path/fname)
        current = np.array(Image.open(tmp_path/fname))
        reference = np.array(Image.open(res_path/fname))
        assert np.allclose(current,reference)

