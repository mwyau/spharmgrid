; SPDX-FileCopyrightText: 2026 Albert M. W. Yau
;
; SPDX-License-Identifier: BSD-3-Clause

; Generate one NCL/SPHEREPACK comparison output. The input file is written by
; generate_inputs.py and contains ascending-latitude, non-cyclic,
; double-precision arrays. NCL's Gaussian routines are used for GL and its
; fixed-grid routines are used for CC.
;
; The spectral masks below use degree/order loops.
; They are mathematical definitions, independent of spharmgrid's Python
; TransformSpec implementation.  In particular, T42x10 is not approximated
; by a triangular helper: it retains 0 <= l <= 42 and 0 <= m <= min(l,10).

procedure mask_scalar(a, b, domain, l_upper, m_upper, tapered, taper_mode)
local shape_a, nlat_a, ncoef_a, l, m, keep
begin
  shape_a = dimsizes(a)
  nlat_a = shape_a(0)
  ncoef_a = shape_a(1)
  do l = 0, nlat_a - 1
    do m = 0, ncoef_a - 1
      keep = False
      if (m.le.l .and. l.le.l_upper .and. m.le.m_upper) then
        if (domain.eq."full") then
          ; Full-resolution baseline: the caller supplies the common
          ; transform bandwidth for the source/target geometry pair.
          keep = True
        elseif (domain.eq."T42") then
          ; Triangular T42: 0 <= m <= l <= 42.
          keep = True
        elseif (domain.eq."T5_42") then
          ; Degree band: 5 <= l <= 42 and 0 <= m <= l.
          keep = l.ge.5
        elseif (domain.eq."T42x10") then
          ; Trapezoidal T42x10: 0 <= l <= 42 and 0 <= m <= min(l,10).
          keep = m.le.10
        elseif (domain.eq."R21") then
          ; Rhomboidal R21: 0 <= m <= 21 and 0 <= l-m <= 21.
          keep = m.le.21 .and. (l - m).le.21
        end if
      end if
      if (.not.keep) then
        a(l,m) = 0d0
        b(l,m) = 0d0
      end if
    end do
  end do
  if (tapered) then
    ; exp_tapersh applies S(l) to coefficients indexed by total degree l and
    ; zonal order m.
    exp_tapersh(a, b, taper_mode, 2)
  end if
end

procedure mask_vector(br, bi, cr, ci, domain, l_upper, m_upper, tapered, taper_mode)
local shape_b, nlat_b, ncoef_b, l, m, keep
begin
  shape_b = dimsizes(br)
  nlat_b = shape_b(0)
  ncoef_b = shape_b(1)
  do l = 0, nlat_b - 1
    do m = 0, ncoef_b - 1
      keep = False
      if (m.le.l .and. l.le.l_upper .and. m.le.m_upper) then
        if (domain.eq."full") then
          keep = True
        elseif (domain.eq."T42") then
          keep = True
        elseif (domain.eq."T5_42") then
          keep = l.ge.5
        elseif (domain.eq."T42x10") then
          keep = m.le.10
        elseif (domain.eq."R21") then
          keep = m.le.21 .and. (l - m).le.21
        end if
      end if
      if (.not.keep) then
        br(l,m) = 0d0
        bi(l,m) = 0d0
        cr(l,m) = 0d0
        ci(l,m) = 0d0
      end if
    end do
  end do
  if (tapered) then
    exp_tapersh(br, bi, taper_mode, 2)
    exp_tapersh(cr, ci, taper_mode, 2)
  end if
end

procedure copy_scalar(source_a, source_b, target_a, target_b)
local source_shape, target_shape, source_nlat, source_ncoef, target_nlat, target_ncoef, l, m
begin
  source_shape = dimsizes(source_a)
  target_shape = dimsizes(target_a)
  source_nlat = source_shape(0)
  source_ncoef = source_shape(1)
  target_nlat = target_shape(0)
  target_ncoef = target_shape(1)
  target_a = 0d0
  target_b = 0d0
  do l = 0, min((/source_nlat,target_nlat/)) - 1
    do m = 0, min((/source_ncoef,target_ncoef/)) - 1
      target_a(l,m) = source_a(l,m)
      target_b(l,m) = source_b(l,m)
    end do
  end do
end

procedure copy_vector(source_br, source_bi, source_cr, source_ci, target_br, target_bi, target_cr, target_ci)
local source_shape, target_shape, source_nlat, source_ncoef, target_nlat, target_ncoef, l, m
begin
  source_shape = dimsizes(source_br)
  target_shape = dimsizes(target_br)
  source_nlat = source_shape(0)
  source_ncoef = source_shape(1)
  target_nlat = target_shape(0)
  target_ncoef = target_shape(1)
  target_br = 0d0
  target_bi = 0d0
  target_cr = 0d0
  target_ci = 0d0
  do l = 0, min((/source_nlat,target_nlat/)) - 1
    do m = 0, min((/source_ncoef,target_ncoef/)) - 1
      target_br(l,m) = source_br(l,m)
      target_bi(l,m) = source_bi(l,m)
      target_cr(l,m) = source_cr(l,m)
      target_ci(l,m) = source_ci(l,m)
    end do
  end do
end

procedure write_regrid_case(output_file, target_kind, target_lat, target_lon, source_a, source_b, source_br, source_bi, source_cr, source_ci, domain, taper_suffix)
local target_lat_shape, target_lon_shape, target_nlat, target_nlon, target_ncoef, target_a, target_b, target_br, target_bi, target_cr, target_ci, scalar_target, vector_target_u, vector_target_v, output_name
begin
  target_lat_shape = dimsizes(target_lat)
  target_lon_shape = dimsizes(target_lon)
  target_nlat = target_lat_shape(0)
  target_nlon = target_lon_shape(0)
  target_ncoef = min((/target_nlat, (target_nlon + 2) / 2/))

  target_a = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  target_b = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  copy_scalar(source_a, source_b, target_a, target_b)
  scalar_target = new((/target_nlat,target_nlon/), double, "No_FillValue")
  scalar_target = 0d0
  if (target_kind.eq."gl") then
    shsgc(target_a, target_b, scalar_target)
  else
    shsec(target_a, target_b, scalar_target)
  end if
  scalar_target!0 = "target_lat_" + target_kind
  scalar_target!1 = "target_lon_" + target_kind
  output_name = "regrid_scalar_" + target_kind + "_" + domain + "_" + taper_suffix
  output_file->$output_name$ = (/ scalar_target /)

  target_br = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  target_bi = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  target_cr = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  target_ci = new((/target_nlat,target_ncoef/), double, "No_FillValue")
  copy_vector(source_br, source_bi, source_cr, source_ci, target_br, target_bi, target_cr, target_ci)
  vector_target_u = new((/target_nlat,target_nlon/), double, "No_FillValue")
  vector_target_v = new((/target_nlat,target_nlon/), double, "No_FillValue")
  vector_target_u = 0d0
  vector_target_v = 0d0
  if (target_kind.eq."gl") then
    vhsgc(target_br, target_bi, target_cr, target_ci, vector_target_u, vector_target_v)
  else
    vhsec(target_br, target_bi, target_cr, target_ci, vector_target_u, vector_target_v)
  end if
  vector_target_u!0 = "target_lat_" + target_kind
  vector_target_u!1 = "target_lon_" + target_kind
  vector_target_v!0 = "target_lat_" + target_kind
  vector_target_v!1 = "target_lon_" + target_kind
  output_name = "regrid_vector_u_" + target_kind + "_" + domain + "_" + taper_suffix
  output_file->$output_name$ = (/ vector_target_u /)
  output_name = "regrid_vector_v_" + target_kind + "_" + domain + "_" + taper_suffix
  output_file->$output_name$ = (/ vector_target_v /)
end

begin
  input_path = getenv("SPHARMGRID_NCL_INPUT")
  output_path = getenv("SPHARMGRID_NCL_OUTPUT")
  grid_kind = getenv("SPHARMGRID_NCL_GRID")
  family = getenv("SPHARMGRID_NCL_FAMILY")

  if (input_path.eq."" .or. output_path.eq."" .or. family.eq."") then
    print("SPHARMGRID_NCL_INPUT, OUTPUT, and FAMILY are required")
    status_exit(2)
  end if
  if (grid_kind.ne."gl" .and. grid_kind.ne."cc") then
    print("SPHARMGRID_NCL_GRID must be gl or cc")
    status_exit(2)
  end if

  input_file = addfile(input_path, "r")
  scalar = todouble(input_file->scalar)
  u = todouble(input_file->u)
  v = todouble(input_file->v)
  lat = todouble(input_file->lat)
  lon = todouble(input_file->lon)
  target_lat_gl = todouble(input_file->target_lat_gl)
  target_lon_gl = todouble(input_file->target_lon_gl)
  target_lat_cc = todouble(input_file->target_lat_cc)
  target_lon_cc = todouble(input_file->target_lon_cc)

  if (.not.isdouble(scalar) .or. .not.isdouble(u) .or. .not.isdouble(v)) then
    print("NCL parity input arrays must be double")
    status_exit(2)
  end if
  print("NCL parity input precision: " + typeof(scalar))

  setfileoption("nc", "Format", "NetCDF4Classic")
  output_file = addfile(output_path, "c")
  output_file@schema_version = "1"
  output_file@ncl_version = "6.6.2"
  output_file@reference = "NCL 6.6.2 built-in SPHEREPACK"
  output_file@grid_kind = grid_kind
  output_file@input_family = family
  output_file@input_schema_version = "1"
  output_file@random_seed = input_file@random_seed
  output_file@latitude_order = "ascending"
  output_file@longitude_convention = "0_to_360_noncyclic"
  output_file@earth_radius_m = 6371220d0
  output_file@taper = 0.1d0
  output_file@taper_mode = (-1d0 + sqrt(1d0 + 4d0 * 42d0 * 43d0 / sqrt(-log(0.1d0)))) / 2d0

  ; Declare the canonical dimensions and coordinates before any dynamic
  ; output variable is written.  This prevents NCL from inventing anonymous
  ; dimensions while creating the reference file.
  lat!0 = "lat"
  lon!0 = "lon"
  target_lat_gl!0 = "target_lat_gl"
  target_lon_gl!0 = "target_lon_gl"
  target_lat_cc!0 = "target_lat_cc"
  target_lon_cc!0 = "target_lon_cc"
  scalar!0 = "lat"
  scalar!1 = "lon"
  u!0 = "lat"
  u!1 = "lon"
  v!0 = "lat"
  v!1 = "lon"
  output_file->lat = (/ lat /)
  output_file->lon = (/ lon /)
  output_file->target_lat_gl = (/ target_lat_gl /)
  output_file->target_lon_gl = (/ target_lon_gl /)
  output_file->target_lat_cc = (/ target_lat_cc /)
  output_file->target_lon_cc = (/ target_lon_cc /)

  source_shape = dimsizes(scalar)
  source_nlat = source_shape(0)
  source_nlon = source_shape(1)
  source_ncoef = min((/source_nlat, (source_nlon + 2) / 2/))

  if (grid_kind.eq."gl") then
    source_lmax = source_nlat - 1
  else
    ; spharmgrid's CC transform leaves the final pole-resolving degree out.
    source_lmax = source_nlat - 2
  end if
  source_mmax = min((/source_lmax, (source_nlon - 1) / 2/))
  set_sphere_radius(6371220d0)

  ; spharmgrid taper=0.1 sets the response at l=42. NCL exp_tapersh uses
  ; exp(-1) at N(N+1), so solve exp(-(42*43/(N*(N+1)))^2)=0.1 for N.
  taper_mode = (-1d0 + sqrt(1d0 + 4d0 * 42d0 * 43d0 / sqrt(-log(0.1d0)))) / 2d0
  taper_weights = new(43, double, "No_FillValue")
  taper_weights!0 = "degree"
  taper_weights(0) = 1d0
  taper_weights(1:42) = exp_tapersh_wgts(42, taper_mode, 2)

  ; Analyze the source scalar and vector fields once and reuse the coefficients
  ; for filtering and regridding. These are NCL/SPHEREPACK transforms, not
  ; spharmgrid transforms.
  scalar_a_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  scalar_b_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  if (grid_kind.eq."gl") then
    shagc(scalar, scalar_a_full, scalar_b_full)
  else
    shaec(scalar, scalar_a_full, scalar_b_full)
  end if

  vector_br_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  vector_bi_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  vector_cr_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  vector_ci_full = new((/source_nlat,source_ncoef/), double, "No_FillValue")
  if (grid_kind.eq."gl") then
    vhagc(u, v, vector_br_full, vector_bi_full, vector_cr_full, vector_ci_full)
  else
    vhaec(u, v, vector_br_full, vector_bi_full, vector_cr_full, vector_ci_full)
  end if

  ; Common untruncated scalar and vector operators.  Each comment names the
  ; direct NCL/SPHEREPACK analogue used for that spharmgrid public function.
  gradient_eastward = scalar
  gradient_northward = scalar
  if (grid_kind.eq."gl") then
    gradsg(scalar, gradient_eastward, gradient_northward)       ; gradient
  else
    gradsf(scalar, gradient_eastward, gradient_northward)       ; gradient
  end if

  inverse_gradient = scalar
  if (grid_kind.eq."gl") then
    igradsg(gradient_eastward, gradient_northward, inverse_gradient)
  else
    igradsf(gradient_eastward, gradient_northward, inverse_gradient)
  end if

  laplacian = scalar
  inverse_laplacian = scalar
  if (grid_kind.eq."gl") then
    lapsg(scalar, laplacian)                                    ; laplacian
    ilapsg(scalar, 0d0, inverse_laplacian)                     ; inverse_laplacian
  else
    lapsf(scalar, laplacian)
    ilapsf(scalar, 0d0, inverse_laplacian)
  end if

  vorticity = u
  divergence = v
  if (grid_kind.eq."gl") then
    uv2vrg(u, v, vorticity)                                     ; vorticity
    uv2dvg(u, v, divergence)                                    ; divergence
  else
    uv2vrf(u, v, vorticity)
    uv2dvf(u, v, divergence)
  end if

  kinematic_vorticity = u
  kinematic_divergence = v
  streamfunction = u
  velocity_potential = v
  if (grid_kind.eq."gl") then
    uv2vrdvg(u, v, kinematic_vorticity, kinematic_divergence)    ; kinematics
    uv2sfvpg(u, v, streamfunction, velocity_potential)           ; potentials
  else
    uv2vrdvf(u, v, kinematic_vorticity, kinematic_divergence)
    uv2sfvpf(u, v, streamfunction, velocity_potential)
  end if

  rotational_u = u
  rotational_v = v
  divergent_u = u
  divergent_v = v
  wind_vo_d_u = u
  wind_vo_d_v = v
  wind_potentials_u = u
  wind_potentials_v = v
  helmholtz_u_rotational = u
  helmholtz_v_rotational = v
  helmholtz_u_divergent = u
  helmholtz_v_divergent = v
  if (grid_kind.eq."gl") then
    vr2uvg(vorticity, rotational_u, rotational_v)      ; rotational_wind
    dv2uvg(divergence, divergent_u, divergent_v)       ; divergent_wind
    vrdv2uvg(vorticity, divergence, wind_vo_d_u, wind_vo_d_v)
    sfvp2uvg(streamfunction, velocity_potential, wind_potentials_u, wind_potentials_v)
    vr2uvg(kinematic_vorticity, helmholtz_u_rotational, helmholtz_v_rotational)
    dv2uvg(kinematic_divergence, helmholtz_u_divergent, helmholtz_v_divergent)
  else
    vr2uvf(vorticity, rotational_u, rotational_v)
    dv2uvf(divergence, divergent_u, divergent_v)
    vrdv2uvf(vorticity, divergence, wind_vo_d_u, wind_vo_d_v)
    sfvp2uvf(streamfunction, velocity_potential, wind_potentials_u, wind_potentials_v)
    vr2uvf(kinematic_vorticity, helmholtz_u_rotational, helmholtz_v_rotational)
    dv2uvf(kinematic_divergence, helmholtz_u_divergent, helmholtz_v_divergent)
  end if

  vector_laplacian_u = u
  vector_laplacian_v = v
  inverse_vector_laplacian_u = u
  inverse_vector_laplacian_v = v
  if (grid_kind.eq."gl") then
    lapvg(u, v, vector_laplacian_u, vector_laplacian_v)         ; vector_laplacian
    ilapvg(u, v, inverse_vector_laplacian_u, inverse_vector_laplacian_v)
  else
    lapvf(u, v, vector_laplacian_u, vector_laplacian_v)
    ilapvf(u, v, inverse_vector_laplacian_u, inverse_vector_laplacian_v)
  end if

  ; Scalar filters.  The explicit l/m domains are applied to NCL coefficients
  ; before synthesis, including the independent T42x10 and R21 definitions.
  domain_names = (/"T42", "T5_42", "T42x10", "R21"/)
  do domain_index = 0, 3
    domain = domain_names(domain_index)
    scalar_a = scalar_a_full
    scalar_b = scalar_b_full
    mask_scalar(scalar_a, scalar_b, domain, 42, 42, False, taper_mode)
    scalar_filtered = scalar
    if (grid_kind.eq."gl") then
      shsgc(scalar_a, scalar_b, scalar_filtered)
    else
      shsec(scalar_a, scalar_b, scalar_filtered)
    end if
    output_name = "filter_" + domain + "_hard"
    output_file->$output_name$ = (/ scalar_filtered /)

    scalar_a = scalar_a_full
    scalar_b = scalar_b_full
    mask_scalar(scalar_a, scalar_b, domain, 42, 42, True, taper_mode)
    scalar_filtered = scalar
    if (grid_kind.eq."gl") then
      shsgc(scalar_a, scalar_b, scalar_filtered)
    else
      shsec(scalar_a, scalar_b, scalar_filtered)
    end if
    output_name = "filter_" + domain + "_taper"
    output_file->$output_name$ = (/ scalar_filtered /)
  end do

  ; The full-resolution, taper-disabled baseline is an analysis/synthesis
  ; operation with spharmgrid's actual geometry bandwidth.
  scalar_a = scalar_a_full
  scalar_b = scalar_b_full
  mask_scalar(scalar_a, scalar_b, "full", source_lmax, source_mmax, False, taper_mode)
  scalar_filtered = scalar
  if (grid_kind.eq."gl") then
    shsgc(scalar_a, scalar_b, scalar_filtered)
  else
    shsec(scalar_a, scalar_b, scalar_filtered)
  end if
  output_file->filter_full_hard = (/ scalar_filtered /)

  ; Regridding analyzes on the source grid and synthesizes on the target grid.
  ; Explicit coefficient masks allow the spharmgrid truncation domains and
  ; taper to be compared without NCL's regridding wrappers.
  do domain_index = -1, 3
    if (domain_index.eq.-1) then
      domain = "full"
      case_lmax = min((/source_lmax,63/))
      case_mmax = min((/source_mmax,63/))
      tapered = False
      taper_suffix = "hard"

      ; GL supports at most l=63 on the requested 64 x 128 target.
      scalar_a = scalar_a_full
      scalar_b = scalar_b_full
      mask_scalar(scalar_a, scalar_b, domain, case_lmax, case_mmax, tapered, taper_mode)
      vector_br = vector_br_full
      vector_bi = vector_bi_full
      vector_cr = vector_cr_full
      vector_ci = vector_ci_full
      mask_vector(vector_br, vector_bi, vector_cr, vector_ci, domain, case_lmax, case_mmax, tapered, taper_mode)
      write_regrid_case(output_file, "gl", target_lat_gl, target_lon_gl, scalar_a, scalar_b, vector_br, vector_bi, vector_cr, vector_ci, domain, taper_suffix)

      ; CC supports at most l=71 on the requested 73 x 144 target.  Retain
      ; those additional source degrees when the source is also CC.
      case_lmax = min((/source_lmax,71/))
      case_mmax = min((/source_mmax,71/))
      scalar_a = scalar_a_full
      scalar_b = scalar_b_full
      mask_scalar(scalar_a, scalar_b, domain, case_lmax, case_mmax, tapered, taper_mode)
      vector_br = vector_br_full
      vector_bi = vector_bi_full
      vector_cr = vector_cr_full
      vector_ci = vector_ci_full
      mask_vector(vector_br, vector_bi, vector_cr, vector_ci, domain, case_lmax, case_mmax, tapered, taper_mode)
      write_regrid_case(output_file, "cc", target_lat_cc, target_lon_cc, scalar_a, scalar_b, vector_br, vector_bi, vector_cr, vector_ci, domain, taper_suffix)
    else
      domain = domain_names(domain_index)
      case_lmax = 42
      case_mmax = 42
      do taper_index = 0, 1
        tapered = taper_index.eq.1
        taper_suffix = "hard"
        if (tapered) then
          taper_suffix = "taper"
        end if

        scalar_a = scalar_a_full
        scalar_b = scalar_b_full
        mask_scalar(scalar_a, scalar_b, domain, case_lmax, case_mmax, tapered, taper_mode)
        vector_br = vector_br_full
        vector_bi = vector_bi_full
        vector_cr = vector_cr_full
        vector_ci = vector_ci_full
        mask_vector(vector_br, vector_bi, vector_cr, vector_ci, domain, case_lmax, case_mmax, tapered, taper_mode)
        write_regrid_case(output_file, "gl", target_lat_gl, target_lon_gl, scalar_a, scalar_b, vector_br, vector_bi, vector_cr, vector_ci, domain, taper_suffix)
        write_regrid_case(output_file, "cc", target_lat_cc, target_lon_cc, scalar_a, scalar_b, vector_br, vector_bi, vector_cr, vector_ci, domain, taper_suffix)
      end do
    end if
  end do

  output_file->scalar_input = (/ scalar /)
  output_file->u_input = (/ u /)
  output_file->v_input = (/ v /)
  output_file->gradient_eastward = (/ gradient_eastward /)
  output_file->gradient_northward = (/ gradient_northward /)
  output_file->inverse_gradient = (/ inverse_gradient /)
  output_file->laplacian = (/ laplacian /)
  output_file->inverse_laplacian = (/ inverse_laplacian /)
  output_file->vorticity = (/ vorticity /)
  output_file->divergence = (/ divergence /)
  output_file->kinematic_vorticity = (/ kinematic_vorticity /)
  output_file->kinematic_divergence = (/ kinematic_divergence /)
  output_file->streamfunction = (/ streamfunction /)
  output_file->velocity_potential = (/ velocity_potential /)
  output_file->rotational_u = (/ rotational_u /)
  output_file->rotational_v = (/ rotational_v /)
  output_file->divergent_u = (/ divergent_u /)
  output_file->divergent_v = (/ divergent_v /)
  output_file->wind_vo_d_u = (/ wind_vo_d_u /)
  output_file->wind_vo_d_v = (/ wind_vo_d_v /)
  output_file->wind_potentials_u = (/ wind_potentials_u /)
  output_file->wind_potentials_v = (/ wind_potentials_v /)
  output_file->helmholtz_u_rotational = (/ helmholtz_u_rotational /)
  output_file->helmholtz_v_rotational = (/ helmholtz_v_rotational /)
  output_file->helmholtz_u_divergent = (/ helmholtz_u_divergent /)
  output_file->helmholtz_v_divergent = (/ helmholtz_v_divergent /)
  output_file->vector_laplacian_u = (/ vector_laplacian_u /)
  output_file->vector_laplacian_v = (/ vector_laplacian_v /)
  output_file->inverse_vector_laplacian_u = (/ inverse_vector_laplacian_u /)
  output_file->inverse_vector_laplacian_v = (/ inverse_vector_laplacian_v /)
  output_file->taper_weights = (/ taper_weights /)
  print("NCL parity output precision: " + typeof(gradient_eastward))
end
