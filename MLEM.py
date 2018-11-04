import numpy as np
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

def calcPETgeom(det_diameter, det_cnts, det_arclengths):
    tol = 1e-13
    # Calculate differential angles (in radians) of each detector size
    det_circumference = pi * det_diameter
    det_angles = 2 * det_arclengths / det_diameter  # radians
    # Check that we have a full ring of detectors (not more or less)
    if abs(sum(det_angles * det_cnts) - 2 * pi) > tol :
        raise Exception("You must fill the detector array exactly (currently using {} of 2\pi radians".format(str(sum(det_angles*det_cnts))))
    n_det_groups = len(matrix(det_cnts))
    det_indiv_angele = []
    for i in range(n_det_groups):
        group_angle_sz = det_angles
        n_det_in_group = det_cnts
        # This is very sloppy... should preallocate, but we have plenty of
        # memory to be lazy at this point :)
        det_indiv_angle = ones((1,n_det_in_group)) * group_angle_sz
    # Calculate half angles
    det_half_angle = det_indiv_angle / 2

    # Create detector center angles
    center_angle = cumsum(det_indiv_angle) - det_half_angle
    return center_angle, det_half_angle

def calcPixGeom(im_size , bore_diameter):
    # calculate bore radius
    bore_radius = bore_diameter / 2
    pix_bord_lsp_x = linspace(-bore_radius, bore_radius, im_size +1)
    pix_bord_lsp_y = pix_bord_lsp_x
    pix_bord_x, pix_bord_y = meshgrid(pix_bord_lsp_x, pix_bord_lsp_y)
    linsp_pix_cent = pix_bord_lsp_x + (pix_bord_lsp_x[1] - pix_bord_lsp_x[0]) / 2
    linsp_pix_cent = linsp_pix_cent[:-1]
    pix_cent_x ,pix_cent_y = meshgrid(linsp_pix_cent , linsp_pix_cent)
    return pix_bord_lsp_x, pix_bord_lsp_y, pix_bord_x , pix_bord_y, pix_cent_x, pix_cent_y


def calcCoincidenceLORs(center_angle, det_half_angle, det_diameter, bore_diameter):
    det_radius = det_diameter / 2
    bore_radius = bore_diameter / 2
    det_center_x = det_radius * cos(center_angle)
    det_center_y = det_radius * sin(center_angle)
    det_edge1_x = det_radius * cos(center_angle + det_half_angle)
    det_edge1_y = det_radius * sin(center_angle + det_half_angle)
    det_edge2_x = det_radius * cos(center_angle - det_half_angle)
    det_edge2_y = det_radius * sin(center_angle - det_half_angle)

    ## transform shape
    det_center_x = det_center_x[0]
    det_center_y = det_center_y[0]
    det_edge1_x = det_edge1_x[0]
    det_edge1_y = det_edge1_y[0]
    det_edge2_x = det_edge2_x[0]
    det_edge2_y = det_edge2_y[0]

    n_det = len(det_center_x)
    k = 0

    n_rays = sum(range(1, n_det))
    LOR_x_all = zeros((2, n_rays))
    LOR_y_all = LOR_x_all
    edge_x1_all = LOR_x_all
    edge_y1_all = LOR_x_all
    edge_x2_all = LOR_x_all
    edge_y2_all = LOR_x_all

    # Prepare a friendly waitbar since the calculation takes a bit...


    # Loop through all possible pairs and only add unique rays
    for i in range(n_det):  # Starting pixel
        for j in range(n_det):  # Ending pixel
            # Only add rays that are unique and not starting/ending at the same point
            if i != j and (det_center_x[i] > det_center_x[j] or (
                    det_center_x[i] == det_center_x[j] and det_center_y[i] > det_center_y[j])):
                k = k + 1
                LOR_x_all[:, k - 1] = det_center_x[i], det_center_x[j]
                LOR_y_all[:, k - 1] = det_center_y[i], det_center_y[j]

                edge_x1_all[:, k - 1] = det_edge1_x[i], det_edge2_x[j]
                edge_y1_all[:, k - 1] = det_edge1_y[i], det_edge2_y[j]
                edge_x2_all[:, k - 1] = det_edge2_x[i], det_edge1_x[j]
                edge_y2_all[:, k - 1] = det_edge2_y[i], det_edge1_y[j]
    k = 0

    n_rays = sum(range(1, n_det))
    LOR_x_all = zeros((2, n_rays))
    LOR_y_all = zeros((2, n_rays))
    edge_x1_all = zeros((2, n_rays))
    edge_y1_all = zeros((2, n_rays))
    edge_x2_all = zeros((2, n_rays))
    edge_y2_all = zeros((2, n_rays))

    for i in range(n_det):  # Starting pixel
        for j in range(n_det):  # Ending pixel
            # Only add rays that are unique and not starting/ending at the same point
            if i != j and (det_center_x[i] > det_center_x[j] or (
                    det_center_x[i] == det_center_x[j] and det_center_y[i] > det_center_y[j])):
                k = k + 1
                LOR_x_all[:, k - 1] = det_center_x[i], det_center_x[j]
                LOR_y_all[:, k - 1] = det_center_y[i], det_center_y[j]

                edge_x1_all[:, k - 1] = det_edge1_x[i], det_edge2_x[j]
                edge_y1_all[:, k - 1] = det_edge1_y[i], det_edge2_y[j]
                edge_x2_all[:, k - 1] = det_edge2_x[i], det_edge1_x[j]
                edge_y2_all[:, k - 1] = det_edge2_y[i], det_edge1_y[j]
    k = 0
    for i in range(n_rays):
        if LOR_x_all[0, i] < -bore_radius and LOR_x_all[1, i] < -bore_radius:
            continue
        elif LOR_x_all[0, i] > bore_radius and LOR_x_all[1, i] > bore_radius:
            continue
        elif LOR_y_all[0, i] < -bore_radius and LOR_y_all[1, i] < -bore_radius:
            continue
        elif LOR_y_all[0, i] > bore_radius and LOR_y_all[1, i] > bore_radius:
            continue
        else:
            k = k + 1

    LOR_x = zeros((2, k))
    LOR_y = zeros((2, k))
    edge_x1 = zeros((2, k))
    edge_y1 = zeros((2, k))
    edge_x2 = zeros((2, k))
    edge_y2 = zeros((2, k))

    k = 0
    for i in range(n_rays):
        if LOR_x_all[0, i] < -bore_radius and LOR_x_all[1, i] < -bore_radius:
            continue
        elif LOR_x_all[0, i] > bore_radius and LOR_x_all[1, i] > bore_radius:
            continue
        elif LOR_y_all[0, i] < -bore_radius and LOR_y_all[1, i] < -bore_radius:
            continue
        elif LOR_y_all[0, i] > bore_radius and LOR_y_all[1, i] > bore_radius:
            continue
        else:
            k = k + 1
            LOR_x[:, k - 1] = LOR_x_all[0, i], LOR_x_all[1, i]
            LOR_y[:, k - 1] = LOR_y_all[0, i], LOR_y_all[1, i]
            edge_x1[:, k - 1] = edge_x1_all[0, i], edge_x1_all[1, i]
            edge_y1[:, k - 1] = edge_y1_all[0, i], edge_y1_all[1, i]
            edge_x2[:, k - 1] = edge_x2_all[0, i], edge_x2_all[1, i]
            edge_y2[:, k - 1] = edge_y2_all[0, i], edge_y2_all[1, i]
    return LOR_x, LOR_y, edge_x1, edge_y1, edge_x2, edge_y2


def calcRayProbMatrix(ray_x, ray_y,
                      edge1_x, edge1_y, edge2_x, edge2_y,
                      pix_bord_lsp_x, pix_bord_lsp_y, pix_cent_x, pix_cent_y):
    Pj = zeros(pix_cent_x.shape)
    pix_half_size = (pix_bord_lsp_x[1] - pix_bord_lsp_x[0]) / 2

    # Calculate pixel diagonal distance
    pix_diag = sqrt((pix_bord_lsp_x[0] - pix_bord_lsp_x[1]) ** 2 + (pix_bord_lsp_y[0] - pix_bord_lsp_y[1]) ** 2)

    # Project corner of each pixel onto ray
    # Corner labels: 1 2
    #                3 4

    Pt_Bet_x1 = pix_cent_x - pix_half_size  ######
    Pt_Bet_x2 = pix_cent_y - pix_half_size  ######
    Pt_Bet_x3 = pix_cent_x + pix_half_size  ######
    Pt_Bet_x4 = pix_cent_y + pix_half_size  ######

    corner1_in_bounds = calcPtBetweenRays(Pt_Bet_x1, Pt_Bet_x2,
                                          edge1_x, edge1_y,
                                          edge2_x, edge2_y)

    corner2_in_bounds = calcPtBetweenRays(Pt_Bet_x3, Pt_Bet_x2,
                                          edge1_x, edge1_y,
                                          edge2_x, edge2_y)

    corner3_in_bounds = calcPtBetweenRays(Pt_Bet_x1, Pt_Bet_x4,
                                          edge1_x, edge1_y,
                                          edge2_x, edge2_y)

    corner4_in_bounds = calcPtBetweenRays(Pt_Bet_x3, Pt_Bet_x4,
                                          edge1_x, edge1_y,
                                          edge2_x, edge2_y)

    for i in range(len(corner1_in_bounds)):
        for j in range(len(corner1_in_bounds)):
            if corner1_in_bounds[i, j] == 1 and corner2_in_bounds[i, j] == 1 and corner3_in_bounds[i, j] == 1 and \
                            corner4_in_bounds[i, j] == 1:
                Pj[i, j] = 1
    return Pj


def calcProbMatrix(LOR_x, LOR_y, edge_x1, edge_y1,
                   edge_x2, edge_y2, pix_bord_lsp_x, pix_bord_lsp_y,
                   pix_cent_x, pix_cent_y):
    n_LOR = len(LOR_x[0])
    Pij = [0] * n_LOR
    # Calculate diagonal of pixel
    pix_diag = sqrt((pix_bord_lsp_x[0] - pix_bord_lsp_x[1]) ** 2 + (pix_bord_lsp_y[0] - pix_bord_lsp_y[1]) ** 2)

    for i in range(n_LOR):
        Pij[i] = calcRayProbMatrix(LOR_x[:, i], LOR_y[:, i],
                                   edge_x1[:, i], edge_y1[:, i], edge_x2[:, i], edge_y2[:, i],
                                   pix_bord_lsp_x, pix_bord_lsp_y, pix_cent_x, pix_cent_y)
    return Pij


def project_pt(pt_x, pt_y, x1, y1, x2, y2):
    r = ((y1-pt_y) * (y1-y2) - (x1-pt_x) *(x2-x1)) / ((x2-x1)**2 + (y2-y1)**2)
    proj_x = x1 + r * (x2-x1)
    proj_y = y1 + r * (y2-y1)
    return proj_x, proj_y, r


def calcPtBetweenRays(pt_x, pt_y, ray1_x, ray1_y, ray2_x, ray2_y):
    if ray1_x[0] == ray1_x[1] and ray1_y[0] == ray1_y[1]:
        ray1_x[1] = ray1_x[1] + (ray2_x[0] - ray2_x[1])
        ray1_y[1] = ray1_y[1] + (ray2_y[0] - ray2_y[1])

    proj_x_edge, proj_y_edge, r = project_pt(pt_x, pt_y, ray1_x[0], ray1_y[0], ray1_x[1], ray1_y[1])
    edge_int_x, edge_int_y = calcIntersection(pt_x, pt_y, proj_x_edge, proj_y_edge,
                                              ray2_x[0], ray2_y[0], ray2_x[1], ray2_y[1])

    junk2, junk2, r = project_pt(pt_x, pt_y, proj_x_edge, proj_y_edge, edge_int_x, edge_int_y)

    for i in range(len(r)):
        for j in range(len(r)):
            if r[i, j] >= 0 and r[i, j] <= 1:
                r[i, j] = True
            else:
                r[i, j] = False
    in_bounds = r
    return in_bounds


def calcIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    denom = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)
    int_x = ((x1 *y2-y1 *x2) * (x3-x4)-(x1-x2) *(x3 *y4-y3 *x4)) / denom;
    int_y = ((x1 *y2-y1 *x2) * (y3-y4)-(y1-y2) *(x3 *y4-y3 *x4)) / denom;
    return int_x , int_y



def calcProjections(im, Pij):
    n_LOR = len(Pij)
    projections = zeros((n_LOR, 1))
    for j in range(n_LOR):
        projections[j] = sum(Pij[j] * im)

    return projections


def PoissonNoise(img, noise_scale):
    for i, num in enumerate(img):
        if 1e-5 < num <= 1:
            noise = random.poisson(lam=10 * num)
        elif 1 < num <= 10:
            noise = random.poisson(lam=num)
        elif 10 < num <= 100:
            noise = random.poisson(lam=num / 10)
        else:
            noise = 0
        noise = noise / noise_scale
        img[i] += noise
    return img


def MLEM(a, p, M, numItterations):
    num_LOR = len(M)
    norm_im = zeros(a.shape)
    num_pixels = norm_im.shape[0] * norm_im.shape[1]

    for j in range(num_LOR):
        norm_im = norm_im + M[j]

    for itter in range(numItterations):
        add_proj = zeros(a.shape)
        for j in range(num_LOR):
            if sum(M[j] * a) > 0:
                add_proj = add_proj + M[j] * p[j] / sum(M[j] * a)

        non_zero = zeros(norm_im.shape)
        for i in range(norm_im.shape[0]):
            for j in range(norm_im.shape[1]):
                if norm_im[i, j] != 0:
                    non_zero[i, j] = 1
                    a[i, j] = non_zero[i, j] * add_proj[i, j] / norm_im[i, j]

        a_max = np.max(a)
        esc_a = zeros(a.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                esc_a[i, j] = a[i, j] / a_max

        plt.figure(2)
        a_max = np.max(a)
        esc_a = zeros(a.shape)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                esc_a[i, j] = a[i, j] / a_max
        plt.imshow(a,cmap=plt.cm.gray)
        plt.colorbar()
        plt.title('Itteration:{}'.format(str(itter + 1)))
        plt.savefig('./images/' + 'Itteration:{}'.format(str(itter + 1)) + ".jpg")
        plt.show()
    return esc_a

# main()

display_all = True

# Setup PET system dimmensions
# Gantry dimmensions
det_diameter = 92.7 # cm
bore_diameter = 59 # cm

# Detector dimmensions
n_blocks_per_ring = 112
n_det_per_block = 6
n_norm_det = n_blocks_per_ring*n_det_per_block
det_thickness = 3 # cm
det_circumference = pi*det_diameter
ndw = det_circumference/n_norm_det  # cm
hrw = ndw/4 # cm

center_angle, det_half_angle = calcPETgeom(det_diameter,n_norm_det, ndw)

# Pixel dimmensions
im_size = 192
pix_bord_lsp_x, pix_bord_lsp_y, pix_bord_x, pix_bord_y, pix_cent_x, pix_cent_y = calcPixGeom(im_size, bore_diameter)

## Calculate Coincidence LORs
LOR_x, LOR_y, edge_x1, edge_y1, edge_x2, edge_y2 = calcCoincidenceLORs(center_angle, det_half_angle, det_diameter, bore_diameter)

# Calculate Probability Matrix
Pij = calcProbMatrix(LOR_x, LOR_y, edge_x1, edge_y1, edge_x2, edge_y2,
                     pix_bord_lsp_x, pix_bord_lsp_y,
                     pix_cent_x, pix_cent_y)

## Display det geometry (if requested)

if display_all is True:
    # get figure handle
    plt.figure(figsize=(9, 9))
    r_ang = linspace(0, 2 * pi, int(2 * pi / 0.0001) + 1)
    det_radius = det_diameter / 2
    bore_radius = bore_diameter / 2
    inner_det_edge_x = det_radius * cos(center_angle - det_half_angle);
    inner_det_edge_y = det_radius * sin(center_angle - det_half_angle);
    outer_det_edge_x = (det_radius + det_thickness) * cos(center_angle - det_half_angle);
    outer_det_edge_y = (det_radius + det_thickness) * sin(center_angle - det_half_angle);
    inner_det_circ_x = det_radius * cos(r_ang);
    inner_det_circ_y = det_radius * sin(r_ang);
    outer_det_circ_x = (det_radius + det_thickness) * cos(r_ang);
    outer_det_circ_y = (det_radius + det_thickness) * sin(r_ang);
    bore_circ_x = bore_radius * cos(r_ang);
    bore_circ_y = bore_radius * sin(r_ang);

    # Labeled lines
    plt.plot(bore_circ_x, bore_circ_y, '--b')
    plt.plot(0, 0, '+r')
    for i in range(len(inner_det_edge_x[0])):
        Y = array([inner_det_edge_y[0, i], outer_det_edge_y[0, i]])
        X = array([inner_det_edge_x[0, i], outer_det_edge_x[0, i]])
        plt.plot(X, Y, '-b')

    plt.plot(inner_det_circ_x, inner_det_circ_y, '-b');  # Inner det circle
    plt.plot(outer_det_circ_x, outer_det_circ_y, '-b');  # Outer det circle

    pix_con_X = concatenate((pix_bord_x, pix_bord_x))
    pix_y1 = bore_radius * ones(pix_bord_x.shape)
    pix_y2 = -bore_radius * ones(pix_bord_x.shape)
    pix_con_Y = concatenate((pix_y1, pix_y2))
    for i in range(pix_con_X.shape[1]):
        plt.plot(pix_con_X[:, i], pix_con_Y[:, i], '-k')

    bord_x1 = bore_radius * ones(pix_bord_y.shape)
    bord_x2 = -bore_radius * ones(pix_bord_y.shape)
    bord_X = concatenate((bord_x1, bord_x2))
    bord_Y = concatenate((pix_bord_y, pix_bord_y))
    for i in range(bord_X.shape[1]):
        plt.plot(bord_X[:, i], bord_Y[:, i], '-k')

    for i in range(LOR_x.shape[1]):
        plt.plot(LOR_x[:, i], LOR_y[:, i], '-r')

    plt.legend(['Bore', 'Isocenter', 'Detectors'], loc='upper right')
    plt.xlabel('X-Position (cm)');
    plt.ylabel('Y-Position (cm)');
    plt.show()

im_mat = pd.read_csv('im.csv',header = None)
im = im_mat.values
noise_scale = 1e4
projections = calcProjections(im, Pij)
noise_projections = PoissonNoise( projections, noise_scale )

recon_im = ones(im.shape); # Initial guess for reconstruction
recon_im = MLEM(recon_im, noise_projections, Pij, 10);

plt.figure(figsize = (8,8))
plt.imshow(recon_im,cmap=plt.cm.gray)
plt.show()