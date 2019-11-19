
%% image/projection geometry setting
nx = 128; % number of voxel in x direction 
ny = 128; % number of voxel in y direction 
nz = 100; % number of voxel in z direction 
dx = 4; % voxel size in x direction
dz = 4; % voxel size in z direction
na = 168; % number of projection angle

%% setting 
ig = image_geom('nx', nx, 'ny', ny, 'nz', nz, 'dx', dx, 'dz', dz); 
ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-4)) > 0;
sg = sino_geom('par', 'nb', ig.nx, 'na', na * ig.nx / nx, ...
    'dr', ig.dx, 'strip_width', 2*ig.dx);

%% system model
f.dir = test_dir;
f.dsc = [test_dir 't.dsc'];
f.wtr = strrep(f.dsc, 'dsc', 'wtr');
f.mask = [test_dir 'mask.fld'];
fld_write(f.mask, ig.mask)

tmp = Gtomo2_wtmex(sg, ig, 'mask', ig.mask_or);
[tmp dum dum dum dum is_transpose] = ...
    wtfmex('asp:mat', tmp.arg.buff, int32(0));
if is_transpose
    tmp = tmp'; % because row grouped
end
delete(f.wtr)
wtf_write(f.wtr, tmp, ig.nx, ig.ny, sg.nb, sg.na, 'row_grouped', 1)

f.sys_type = sprintf('2z@%s@-', f.wtr);

G = Gtomo3(f.sys_type, ig.mask, ig.nx, ig.ny, ig.nz, ...
    'chat', 0, 'view2d', 1, 'nthread', jf('ncore'));