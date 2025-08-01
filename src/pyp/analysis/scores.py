import math
import multiprocessing
import os
import shutil
import sys
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import json

from pyp import merge
from pyp.analysis import statistics, plot, geometry
from pyp.inout.image import mrc
from pyp.inout.metadata import frealign_parfile, pyp_metadata , get_particles_tilt_index
from pyp.inout.metadata import cistem_star_file
from pyp.refine.frealign import frealign
from pyp.system import project_params, mpi
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path, timer, symlink_relative
from pyp.inout.utils.pyp_edit_box_files import read_boxx_file_async, write_boxx_file_async
from pyp.system.utils import get_imod_path
from pyp.streampyp.logging import TQDMLogger

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def per_frame_scoring(
    parameters, name, current_path, allboxes, allparxs, particle_filenames
):
    """In particle frame refinement step, score the Gaussian averaged particle frames."""
    # follow settings in align.align_stack_super
    if os.path.exists(
        os.path.split(parameters["refine_parfile"])[0] + "/../frealign.config"
    ):
        fparameters = project_params.load_fyp_parameters(
            os.path.split(parameters["class_par"])[0] + "/../"
        )
        maxiter = int(fparameters["maxiter"])
        low_res = float(project_params.param(fparameters["rlref"], maxiter))
        high_res_refi = high_res_eval = float(
            project_params.param(fparameters["rhref"], maxiter)
        )
        # logger.info("high_res_refine", float( project_params.param( fparameters['rhref'], maxiter ) ))
        metric = (
            project_params.param(fparameters["metric"], maxiter)
            + " -fboost "
            + project_params.param(fparameters["fboost"], maxiter)
            + " -maskth "
            + project_params.param(fparameters["maskth"], maxiter)
        )
        # metric_weights = metric
        # print 'Retrieving FREALIGN compatible FREALIGN parameters: rhlref = %.2f, rhref = %.2f, metric = %s' % ( low_res, high_res_refi, metric )
    else:
        logger.warning("Could not find FREALIGN parameters to insure consistency")

    # temporarily disable PYP_SCRATCH
    pyp_scratch_bk = os.environ["PYP_SCRATCH"]
    os.environ["PYP_SCRATCH"] = ""

    if not os.path.exists("../log"):
        os.makedirs("../log")

    particle_stacks = [
        os.path.join("frealign_" + particle_filename, particle_filename + "_stack.mrc")
        for particle_filename in particle_filenames
    ]
    particle_parfiles = [
        particle_filename + "_r01_02.par" for particle_filename in particle_filenames
    ]

    # merge the frame stacks
    mrc.merge(particle_stacks, name + "_frames_stack.mrc")
    # merge the par files
    frealign_parfile.Parameters.merge_parameters(
        particle_parfiles,
        name + "_frames_r01_02.par",
        metric=parameters["refine_metric"],
        parx=True,
    )

    # save the averaged as backup
    shutil.copy2(name + "_frames_stack.mrc", name + "_frames_stack.mrc.real")

    # write out the stack file and par file into a txt for later processing
    with open("../stacks.txt", "a") as f:
        f.write(os.path.join(name, name + "_frames_stack.mrc\n"))
    with open("../pars.txt", "a") as f:
        f.write(os.path.join(name, name + "_frames_r01_02.par\n"))
    # if the project directory file is not written
    with open("../project_dir.txt", "w") as f:
        f.write(str(current_path))

    # find film number for this micrograph to figure out particle alignments
    # TODO: write out to func
    try:
        with open(os.path.join(current_path, parameters["data_set"] + ".films")) as x:
            series = [
                num
                for num, line in enumerate(x, 1)
                if "{}".format(name.replace("_r01", "")) == line.strip()
            ][0] - 1
        # write the overall order of the files
        with open("../ordering.txt", "a") as f:
            f.write("{}\n".format(series))
    except:
        sys.stderr.write("ERROR - Cannot find film number for " + name)
        sys.exit()

    # score the frames
    # call FREALIGN directly to improve performance
    mp = parameters.copy()
    # fp = fparameters.copy()
    mp["refine_mode"] = "1"
    mp["refine_mask"] = "0,0,0,0,0"
    # mp["refine_rlref"] = "{}".format(low_res)
    # mp["refine_rhref"] = "{}".format(high_res_refi)
    mp["refine_dataset"] = name + "_frames"

    # frames = len(np.unique(film_arr[:, scanor_col]))
    # TODO: write out to func
    frames = len(
        [
            boxes
            for boxes, line in zip(allboxes, allparxs[0])
            if float(line.split()[15]) == 0
        ]
    )

    frame_weights_width = int(
        math.floor(frames * 0.4)
    )  # width of gaussian used for frame weighting
    if frame_weights_width % 2 == 0:
        frame_weights_width += 1
    frame_weights_step = False  # use step-like weights for frame weighting

    # build weights for frame averaging
    all_weights = np.zeros([frames, frames])
    for i in range(frames):
        weights = np.exp(-pow((np.arange(frames) - float(i)), 2) / frame_weights_width)
        # apply hard threshold if using simple running averages
        if frame_weights_step:
            weights = np.where(weights > 0.5, 1, 0)
        all_weights[i, :] = weights / weights.mean() / frames

    # weight each particle stack
    blurred_stack = []
    for particle_stack in particle_stacks:
        merge.weight_stack(particle_stack, particle_stack + ".blur", all_weights)
        blurred_stack.append(particle_stack + ".blur")
    mrc.merge(blurred_stack, name + "_frames_stack.mrc")

    # necessary parfiles for scoring/recon
    os.symlink(name + "_frames_r01_02.par", name + "_frames_r01_02_used.par")
    os.symlink(name + "_frames_r01_02.par", name + "_r01_02.par")
    os.symlink(name + "_r01_02.par", name + "_r01_02_used.par")
    shutil.copy2(
        os.path.join(os.getcwd(), name + "_frames_stack.mrc"),
        "../" + name + "_frames_stack.mrc",
    )

    # score the gaussian blurred frames
    header = mrc.readHeaderFromFile(name + "_frames_stack.mrc")
    total_frames = header["nz"]
    score_metric = "cc3m -fboost T"
    command = frealign.mrefine_version(
        mp, 1, total_frames, 2, 1, name + "_r01_02", "", "log.txt", "", score_metric,
    )
    run_shell_command(command)

    # move back the original stack
    shutil.copy(name + "_frames_stack.mrc.real", name + "_frames_stack.mrc")

    # copy back the scored parfile
    # use the   LOGP      SIGMA   SCORE  CHANGE from score par -- cols 12 - 15
    original_par = frealign_parfile.Parameters.from_file(
        name + "_frames_r01_02.par"
    ).data
    # scored_par = frealign_parfile.Parameters.from_file(name + "_r01_02.par_").data
    scored_par = frealign_parfile.Parameters.from_file(name + "_r01_02.par").data
    original_par[:, 12:16] = scored_par[:, 12:16]
    frealign_parfile.Parameters.write_parameter_file(
        name + "_frames_r01_02.par", original_par, parx=True
    )
    # os.rename(name + '_r01_02.par_', name + '_frames_r01_02.par')

    # save the parameters for array job
    mp = parameters.copy()
    # fp = fparameters.copy()
    # mp["refine_rlref"] = "{}".format(low_res)
    # mp["refine_rhref"] = "{}".format(high_res_refi)
    mp["refine_dataset"] = name + "_frames"

    # save fp and mp into the main slurm job folder
    logger.info("saving mp and fp parameter files")
    project_params.save_pyp_parameters(mp, "..")
    # project_params.save_fyp_parameters(fp, "..")

    os.remove("../" + name + "_frames_stack.mrc")

    # set PYP_SCRATCH back to regular
    os.environ["PYP_SCRATCH"] = pyp_scratch_bk


def assign_angular_defocus_groups(
    input, angles: int, defocuses: int
):
    """Divide particles in parameter file into a discrete number of angular and defocus groups.

    Parameters
    ----------
    parfile : str
        Frealign .par file.
    angles : int
        Number of groups used to partition the data
    defocuses : int
        Number of groups used to partition the data
    frealignx : bool, optional
        Specify if .par parameter file is in frealignx format, by default False

    Returns
    -------
    par_obj : Parameters
        Object representing par file data
    angular groups : numpy.ndarray
        Result of angular clustering
    defocus groups : numpy.ndarray
        Result defous clustering.
    """


    """
    # load .parx file
    if os.path.isfile(parfile):

        par_obj = frealign_parfile.Parameters.from_file(parfile)
        input = par_obj.data
        
        if "_used.par" in parfile:
            if frealignx:
                input = input[input[:, 12] > 0]
            else:
                input = input[input[:, 11] > 0]
            par_obj.data = input

    else:
        logger.error("{} not found.".format(parfile))
        sys.exit(0)
    """
    par_obj = cistem_star_file.Parameters()
    theta = par_obj.get_index_of_column(cistem_star_file.THETA)
    defocus_1 = par_obj.get_index_of_column(cistem_star_file.DEFOCUS_1)

    angular_group = np.floor(np.mod(input[:, theta], 180) * angles / 180)
    if input.shape[0] > 0:
        mind, maxd = (
            int(math.floor(input[:, defocus_1].min())),
            int(math.ceil(input[:, defocus_1].max())),
        )
    else:
        mind = maxd = 0
    if maxd == mind:
        defocus_group = np.zeros(angular_group.shape)
    else:
        defocus_group = np.round((input[:, defocus_1] - mind) / (maxd - mind) * (defocuses - 1))

    # return input, angular_group, defocus_group
    return angular_group, defocus_group


def generate_cluster_stacks(inputstack, parfile, angles=25, defocuses=25):
    par_obj, angular_group, defocus_group = assign_angular_defocus_groups(
        parfile, angles, defocuses
    )
    input = par_obj.data

    # create new stacks
    pool = multiprocessing.Pool()
    for g in range(angles):
        for f in range(defocuses):
            cluster = input[np.logical_and(angular_group == g, defocus_group == f)]
            if cluster.shape[0] > 0:
                cluster = cluster[cluster[:, 11].argsort()]
                indexes = (cluster[:, 0] - 1).astype("int").tolist()
                outputstack = "{0}_{1}_{2}_stack.mrc".format(
                    os.path.splitext(inputstack)[0], g, f
                )
                pool.apply_async(
                    mrc.extract_slices, args=(inputstack, indexes, outputstack)
                )
    pool.close()

    # Wait for all processes to complete
    pool.join()


def shape_phase_residuals(
    inputparfile,
    angles,
    defocuses,
    threshold,
    mindefocus,
    maxdefocus,
    firstframe,
    lastframe,
    mintilt,
    maxtilt,
    minazh,
    maxazh,
    minscore,
    maxscore,
    binning,
    reverse,
    consistency,
    scores,
    odd,
    even,
    outputparfile
):

    cistem_par = cistem_star_file.Parameters.from_file(inputparfile)
    filmid = cistem_par.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
    field = cistem_par.get_index_of_column(cistem_star_file.SCORE)
    occ = cistem_par.get_index_of_column(cistem_star_file.OCCUPANCY)
    tind = cistem_par.get_index_of_column(cistem_star_file.TIND)
    defocus1 = cistem_par.get_index_of_column(cistem_star_file.DEFOCUS_1)
    ptlindex = cistem_par.get_index_of_column(cistem_star_file.PIND)

    input = cistem_par.get_data()
    films = np.unique(input[:, filmid])

    # read interesting part of input file
    angular_group, defocus_group = assign_angular_defocus_groups(
        input, angles, defocuses
    )

    # figure out tomo or spr by check tilt angles
    # tilts_dict = cistem_par.get_extended_data().get_tilts()
    merged_extend = inputparfile.replace(".cistem", ".json")
    with open(merged_extend, 'r') as jsonfile:
        tilts_dict = json.load(jsonfile)
        any_key = next(iter(tilts_dict.keys()))

    if any(abs(value) > 0 for value in tilts_dict[any_key].values()):
        is_tomo = True
    else:
        is_tomo = False
        
    tiltangle = []

    if is_tomo:
        for f in films:
            # contruct a tilt angle dictionary
            # tind_angle_dict = {i: tilts_dict[f][i][0].angle for i in tilts_dict[f].keys()}
            if str(int(f)) not in tilts_dict.keys():
                tind_angle_dict = {}
            else:
                tind_angle_dict = tilts_dict[str(int(f))]

            tind_angle_dict_int = {int(k): v for k, v in tind_angle_dict.items()} # json way saving dict as str but we need int
            tind_in_film = input[input[:, filmid]==f][:, tind]

            df_tind = pd.DataFrame(tind_in_film, columns=["Tind"])
            # mapped_angles = np.vectorize(tind_angle_dict.get)(tind_in_film)
            df_tind["Angle"] = df_tind["Tind"].map(tind_angle_dict_int)
            tiltangle.append(df_tind["Angle"].to_numpy())
            # for ti in input[input[:, filmid]==f][:, tind]:
            #    tiltangle.append(tilts_dict[f][ti][0].angle )
        tilt_angle_array = np.concatenate(tiltangle).reshape(-1, 1)

    else:
        tilt_angle_array = np.array([0] * input.shape[0]).reshape(-1, 1)

    input = np.hstack((input, tilt_angle_array))

    tltangle = -1 # tilt angle column as the last column temporarily

    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # additional sorting if matches available
    name = inputparfile[:-7]
    fmatch_stack = "../maps/{0}_match_unsorted.mrc".format(name)
    metric_weights = np.ones(input.shape[0])
    if os.path.exists(fmatch_stack):
        msize = int(mrc.readHeaderFromFile(fmatch_stack)["nx"]) / 2
        y, x = np.ogrid[-msize:msize, -msize:msize]
        mask = np.where(x ** 2 + y ** 2 <= msize ** 2, 1, 0)
    else:
        msize = 0
        mask = np.array([])
    
    """
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    results = manager.Queue()
    """

    # determine per-cluster threshold
    thresholds = np.empty([angles, defocuses])
    thresholds[:] = np.nan
    min_scores = np.empty([angles, defocuses])
    min_scores[:] = np.nan
    max_scores = np.empty([angles, defocuses])
    max_scores[:] = np.nan

    for g in range(angles):
        for f in range(defocuses):
            # get all images in present cluster
            cluster = np.logical_and(angular_group == g, defocus_group == f)
            size = 1

            # make sure we have enough points for computing statistics
            while (
                np.extract(cluster == 1, input[:, field]).size < 100
                and input.shape[0] > 100
            ):

                cluster = np.logical_and(
                    np.logical_and(
                        angular_group >= g - size, angular_group <= g + size
                    ),
                    np.logical_and(
                        defocus_group >= f - size, defocus_group <= f + size
                    ),
                )
                size += 1

            if cluster.size > 0:
                # find cluster threshold using either percentage of size or absolute number of images

                if threshold == 0:
                    
                    if is_tomo:
                        bool_array = np.full(input.shape[0], False, dtype=bool)
                        bool_array[cluster] = True
                        take_values = np.logical_and(bool_array, np.abs(input[:, tltangle]) <= 12)
                        used_array = input[take_values]
                        scores_used = used_array[:, [field, ptlindex]]
                        pf = pd.DataFrame(scores_used, columns=["score", "pind"])
                        prs = pf.groupby("pind")["score"].mean()
                    else:
                        prs = np.extract(cluster == 1, input[:, field])
                        
                    optimal_threshold = 1.075 * statistics.optimal_threshold(samples=prs, criteria="optimal")
                    if prs.size > 1:
                        mythreshold = optimal_threshold
                        cutoff = (
                            1.0 - 1.0 * np.argmin(np.fabs(prs - mythreshold)) / prs.size
                        )
                        # logger.info('Bi-modal distributions detected with means: {0}, {1}'.format( gmix.means_[0][0], gmix.means_[1][0] ))
                        logger.info(f'Using optimal threshold from bimodal distribution = {mythreshold:.2f}')
                        if prs.size > 20:
                            thresholds[g, f] = mythreshold
                        else:
                            logger.warning(
                                "Not enough points for estimating statistics %d %d %d",
                                g,
                                f,
                                prs.size,
                            )

                elif threshold <= 1:
                    # cluster = input[ np.logical_and( angular_group == g, defocus_group == f ) ]
                    if scores:
                        # thresholds[g,f] = cluster[ cluster[:,field].argsort() ][ int( (cluster.shape[0]-1) * (1-threshold) ), field ]
                        if is_tomo:
                            bool_array = np.full(input.shape[0], False, dtype=bool)
                            bool_array[cluster] = True
                            take_values = np.logical_and(bool_array, np.abs(input[:, tltangle]) <= 12)
                            used_array = input[take_values]
                            scores_used = used_array[:, [field, ptlindex]]
                            pf = pd.DataFrame(scores_used, columns=["score", "pind"])
                            meanscore = pf.groupby("pind")["score"].mean()
                            
                            # take_mean = []
                            # for i in np.unique(scores_used[:, 1]):
                            #     take_mean.append(np.mean(scores_used[:, 0], where=scores_used[:,1]==i))

                            # meanscore = np.array(take_mean)
                            if len(meanscore) > 0:
                                thresholds[g, f] = np.sort(meanscore)[
                                    int((meanscore.shape[0] - 1) * (1 - threshold))
                                ]
                            
                        else:
                            thresholds[g, f] = np.sort(input[cluster, field])[
                                int((cluster.shape[0] - 1) * (1 - threshold))
                            ]

                        logger.info(f"Minimum score used for reconstruction = {thresholds[g, f]:.2f}")
                    else:
                        # thresholds[g,f] = cluster[ cluster[:,field].argsort() ][ int( (cluster.shape[0]-1) * threshold ), field ]
                        thresholds[g, f] = np.sort(input[cluster, field])[
                            int((cluster.shape[0] - 1) * threshold)
                        ]

                elif cluster.ndim == 2:
                    # cluster = input[ np.logical_and( angular_group == g, defocus_group == f ) ]
                    if scores:
                        thresholds[g, f] = cluster[cluster[:, field].argsort()][
                            min(cluster.shape[0] - threshold, cluster.shape[0]) - 1,
                            field,
                        ]
                    else:
                        thresholds[g, f] = cluster[cluster[:, field].argsort()][
                            min(threshold, cluster.shape[0]) - 1, field
                        ]

                prs = np.extract(cluster == 1, input[:, field])
                if minscore < 1:
                    min_scores[g, f] = prs.min() + minscore * (prs.max() - prs.min())
                else:
                    min_scores[g, f] = minscore

                if maxscore <= 1:
                    max_scores[g, f] = prs.max() - (1 - maxscore) * (
                        prs.max() - prs.min()
                    )
                else:
                    max_scores[g, f] = maxscore

                # print thresholds[g,f], min_scores[g,f], max_scores[g,f]

            else:
                logger.warning("No points in angular/defocus group.")
    
    """
    pool.close()
    pool.join()

    # Collate periodogram averages
    bad_particles = 0
    total_particles = 0
    while results.empty() == False:
        current = results.get()
        thresholds[current[0], current[1]] = current[2]
        metric_weights[current[3]] = current[4]
        if current[0] + current[1] == 0:
            logger.info(
                "Processing group (%d,%d) containing %d particles and eliminating %d",
                current[0],
                current[1],
                len(current[3]),
                np.where(current[4] == 0, 1, 0).sum(),
            )
        bad_particles += np.where(current[4] == 0, 1, 0).sum()
        total_particles += len(current[3])
    """

    from scipy.ndimage.filters import gaussian_filter

    thresholds = gaussian_filter(thresholds, sigma=1)

    if angles + defocuses > 2:
        plt.clf()
        cax = plt.imshow(thresholds, interpolation="nearest", cmap=cm.jet)
        plt.title("Thresholds per orientation\n and defocus group")
        plt.xlabel("Defocus Group")
        plt.ylabel("Orientation Group")
        plt.colorbar(cax, ticks=[np.nanmin(thresholds), np.nanmax(thresholds)])
        plt.savefig("../maps/%s_thresholds.png" % os.path.splitext(inputparfile)[0])

    # apply new PR threshold
    for g in range(angles):
        for f in range(defocuses):
            # if threshold != 1:
            if scores:
                # input[:,field] = np.where( np.logical_and( np.logical_and( angular_group == g, defocus_group == f ), input[:,field] < thresholds[g,f] ), np.nan, input[:,field] )
                # input[:,occ] = np.where( np.logical_and( np.logical_and( angular_group == g, defocus_group == f ), input[:,field] < thresholds[g,f] ), 0, input[:,occ] )
                if is_tomo and thresholds[g, f] > 0:
                    group_mask = np.logical_and(angular_group == g, defocus_group == f)
                    input_group = input[group_mask]

                    crop_array = input_group[:, [occ, field, ptlindex, tltangle]]
                    crop_by_tltangle = crop_array[np.abs(crop_array[:, -1]) < 10]
                    ptl_index = np.unique(crop_by_tltangle[:, 2])
                    df = pd.DataFrame(crop_by_tltangle, columns=["occ", "score", "pind", "tltangle"])
                    meanscore = df.groupby("pind")["score"].mean().to_numpy()
                    above_threshold = meanscore >= thresholds[g, f]
                    if threshold == 1:
                        above_threshold = meanscore == meanscore
                    discarded = ptl_index[above_threshold == False].size
                    if discarded > 0:
                        logger.info(f"{discarded} particles scores are below the threshold and being removed from reconstruction")
                    modification_mask = np.isin(input_group[:, ptlindex], ptl_index[above_threshold == False])

                    group_mask[group_mask == True] = modification_mask
                    input[group_mask, occ] = 0 
                    
                    # enable the score range threshold
                    input[:, occ] = np.where(
                        np.logical_and(
                            np.logical_and(angular_group == g, defocus_group == f),
                            np.logical_or(
                                input[:, field] < min_scores[g, f],
                                input[:, field] > max_scores[g, f],
                            ),
                        ),
                        0,
                        input[:, occ],
                    )

                else:
                    input[:, occ] = np.where(
                        np.logical_and(
                            np.logical_and(angular_group == g, defocus_group == f),
                            np.logical_or(
                                np.logical_or(
                                    input[:, field] < thresholds[g, f],
                                    input[:, field] < min_scores[g, f],
                                ),
                                input[:, field] > max_scores[g, f],
                            ),
                        ),
                        0,
                        input[:, occ],
                    )  

    if os.path.exists(fmatch_stack):
        logger.info(
            "Removing %d bad particles by distance sorting (%d)",
            bad_particles,
            np.where(metric_weights == 0, 1, 0).sum(),
        )
        logger.info("Total particles = %d", total_particles)
        input[:, occ] = np.where(metric_weights == 0, 0, input[:, occ])
        fmatch_stack_removed = "../maps/%s_match_removed.mrc" % (
            os.path.splitext(inputparfile)[0]
        )
        mrc.extract_slices(
            fmatch_stack,
            np.nonzero(input[:, occ] == 0)[0].tolist(),
            fmatch_stack_removed,
        )

    # ignore if defocus outside permissible range
    if scores:
        input[:, occ] = np.where(
            np.logical_or(input[:, defocus1] < mindefocus, input[:, defocus1] > maxdefocus),
            0,
            input[:, occ],
        )


    # shape accorging to assigned top/side view orientations using mintilt and maxtilt values
    if maxazh < 180 or minazh > 0:
        if scores:
            input[:, occ] = np.where(
                np.logical_or(
                    np.mod(input[:, 2], 180) < minazh, np.mod(input[:, 2], 180) > maxazh
                ),
                0,
                input[:, occ],
            )

    if scores:

        # shape based on exposure sequence
        if lastframe > -1:
            if scores:
                input[:, occ] = np.where(
                    np.logical_or(input[:, tind] < firstframe, input[:, tind] > lastframe),
                    0,
                    input[:, occ],
                )

        # shape based on tilt-angle
        if scores:
            input[:, occ] = np.where(
                np.logical_or(input[:, -1] < mintilt, input[:, -1] > maxtilt),
                0,
                input[:, occ],
            )

    # revert phase residual polarity so that lowest PR become highest and viceversa
    if reverse:
        min_pr = np.extract(np.isfinite(input[:, field]), input[:, field]).min()
        max_pr = np.extract(np.isfinite(input[:, field]), input[:, field]).max()
        input[:, field] = np.where(
            np.isfinite(input[:, field]),
            max_pr - input[:, field] + min_pr,
            input[:, field],
        )

    # apply binning to image shifts (FREALIGN 9 measures shifts in Angstroms so we don't need this)
    if not scores:
        input[:, 4:6] *= binning

    ## particle selection based on consistency of angles/shifts determination
    if consistency:

        prevparfile = "%s%02d.par" % (
            inputparfile.split(".")[0][:-2],
            int(inputparfile.split(".")[0][-2:]) - 1,
        )

        if os.path.isfile(prevparfile):

            # read parameters from previous iteration
            previous = np.array(
                [
                    line.split()
                    for line in open(prevparfile)
                    if not line.startswith("C")
                ],
                dtype=float,
            )

            # detect euler angle jump
            anglejumps = np.mod(abs(input[:, 2] - previous[:, 2]), 360)
            anglejumps_sorted = anglejumps[anglejumps.argsort()]
            maxanglejump = anglejumps_sorted[
                min(int((anglejumps.shape[0] - 1) * threshold), anglejumps.shape[0] - 1)
            ]

            # detect differential shift
            shiftjumps = abs(
                np.hypot(input[:, 4] - previous[:, 4], input[:, 5] - previous[:, 5])
            )
            shiftjumps_sorted = shiftjumps[shiftjumps.argsort()]
            maxshiftjump = shiftjumps_sorted[
                min(int((shiftjumps.shape[0] - 1) * threshold), shiftjumps.shape[0] - 1)
            ]

            # keep only particles with jumps below thresholds
            input[:, field] = np.where(
                np.logical_or(anglejumps > maxanglejump, shiftjumps > maxshiftjump),
                np.nan,
                input[:, field],
            )

    if odd:
        if scores:
            input[::2, occ] = 0
        else:
            input[::2, field] = np.nan

    if even:
        if scores:
            input[1::2, occ] = 0
        else:
            input[1::2, field] = np.nan

    number = input[input[:, occ]==0].shape[0]
    logger.info(f"Number of particles with zero occupancy = {number:,} out of {input.shape[0]:,} ({number/input.shape[0]*100:.2f}%)")

    cistem_par.set_data(input[:, :-1]) # not including the tilt angle column
    
    # set unique particle projection IDs if using for reconstruction
    if outputparfile.endswith("_used.cistem"):
        input[:,0] = np.arange(0,input.shape[0]) + 1
    
    cistem_par.to_binary(outputparfile)

@timer.Timer(
    "call_shape_phase_residuals", text="Shaping scores took: {}", logger=logger.info
)
def call_shape_phase_residuals(
    input_par_file, output_par_file, fp, iteration
):

    mindefocus = float(project_params.param(fp["reconstruct_mindef"], iteration))
    maxdefocus = float(project_params.param(fp["reconstruct_maxdef"], iteration))
    firstframe = int(project_params.param(fp["reconstruct_firstframe"], iteration))
    lastframe = int(project_params.param(fp["reconstruct_lastframe"], iteration))
    mintilt = float(project_params.param(fp["reconstruct_mintilt"], iteration))
    maxtilt = float(project_params.param(fp["reconstruct_maxtilt"], iteration))
    minazh = float(project_params.param(fp["reconstruct_minazh"], iteration))
    maxazh = float(project_params.param(fp["reconstruct_maxazh"], iteration))
    minscore = float(project_params.param(fp["reconstruct_minscore"], iteration))
    maxscore = float(project_params.param(fp["reconstruct_maxscore"], iteration))

    shapr = project_params.param(fp["reconstruct_shapr"], iteration)
    reverse = False
    consistency = False
    if "reverse" in shapr.lower():
        reverse = True
    if "consistency" in shapr.lower():
        consistency = True

    # use NO cutoff if we are using multiple references
    if int(project_params.param(fp["class_num"], iteration)) > 1:
        cutoff = 1
    else:
        cutoff = project_params.param(fp["reconstruct_cutoff"], iteration)

    angle_groups = int(project_params.param(fp["reconstruct_agroups"], iteration))
    defocus_groups = int(project_params.param(fp["reconstruct_dgroups"], iteration))
    cutoff = float(cutoff)
    binning = 1.0
    odd = False
    even = False
    scores = True

    shape_phase_residuals(
        input_par_file,
        angle_groups,
        defocus_groups,
        cutoff,
        mindefocus,
        maxdefocus,
        firstframe,
        lastframe,
        mintilt,
        maxtilt,
        minazh,
        maxazh,
        minscore,
        maxscore,
        binning,
        reverse,
        consistency,
        scores,  # since we are always using scores (not phase residuals)
        odd,
        even,
        output_par_file
    )
    

def eval_phase_residual(
    defocus, mparameters, fparameters, input, name, film, scanor, tolerance
):

    if math.fabs(defocus) > tolerance:
        logger.info("Evaluating %f = %d", defocus, np.nan)
        return np.nan

    particles = input.shape[0]

    eval = np.copy(input)
    eval[:, 8:10] += defocus
    # input[:,10] += 100000 * defocus

    # print input[:,8].mean(), input[:,9].mean()

    # substitue new defocus value and write new parfile
    frealign_parameter_file = "scratch/" + name + "_r01_02.par"

    if "frealignx" in fparameters["metric"]:
        version = "frealignx"
    else:
        version = "new"

    par_obj = frealign_parfile.Parameters(version, data=input)
    par_obj.write_file(frealign_parameter_file)
    # with open(frealign_parameter_file, "w") as f:
    #     # write out header (including resolution table)
    #     # [ f.write(line) for line in open( parfile ) if line.startswith('C') ]
    #     for i in range(particles):
    #         if "frealignx" in fparameters["metric"]:
    #             f.write(
    #                 frealign_parfile.FREALIGNX_PAR_STRING_TEMPLATE % (tuple(eval[i, :16]))
    #             )
    #         else:
    #             f.write(frealign_parfile.NEW_PAR_STRING_TEMPLATE % (tuple(eval[i, :16])))
    #         f.write("\n")

    stack = "data/%s_stack.mrc" % (name).replace("_short", "")
    local_stack = "%s_stack.mrc" % (name)
    if not os.path.exists(local_stack):
        try:
            symlink_relative(os.path.join(os.getcwd(),stack), local_stack)
        except:
            logger.info("symlink failed %s %s", local_stack, stack)
            pass

    # call FREALIGN directly to improve performance
    mp = mparameters.copy()
    fp = fparameters.copy()
    fp["mode"] = "1"
    fp["mask"] = "0,0,0,0,0"
    fp["dataset"] = name

    local_model = os.getcwd() + "/scratch/%s_r01_01.mrc" % (name)
    if not os.path.exists(local_model):
        symlink_relative(mparameters["class_ref"], local_model)

    os.chdir("scratch")

    command = frealign.mrefine_version(
        mp, fp, 1, particles, 2, 1, name + "_r01_02", "", "/dev/null", "", fp["metric"]
    )
    run_shell_command(command)

    output_parfile = name + "_r01_02.par_"

    # open output .par file and average scores
    if "frealignx" in fp["metric"]:
        scores = np.array(
            [
                float(line[129:136])
                for line in open(output_parfile)
                if not line.startswith("C")
            ],
            dtype=float,
        )
    else:
        scores = np.array(
            [
                float(line[121:128])
                for line in open(output_parfile)
                if not line.startswith("C")
            ],
            dtype=float,
        )

    logger.info("Evaluating %f = %f", defocus, scores.mean())

    os.chdir("..")

    return -scores.mean()


def filter_particles(parameter_file: str, mintilt: float, maxtilt: float, dist: float, threshold: float, pixel_size: float):
    """ Compute scores of sub-volume from their corresponding projections in the parfile
        Scores will be updated in box3d files, which will be later used for CSP
    """        
    tiltseries = str(Path(parameter_file).name).split("_r")[0]
    alignment_parameters = cistem_star_file.Parameters.from_file(input_file=parameter_file)
    data = alignment_parameters.get_data()
    alignment_parameters.update_particle_score(tind_range=[], tiltang_range=[mintilt, maxtilt])

    extended_parameters = alignment_parameters.get_extended_data()
    tilt_parameters = extended_parameters.get_tilts()
    particle_parameters = extended_parameters.get_particles()
    
    if dist < 0.0:
        logger.error(
            "Distance cutoff has to be greater than 0. Dist of 0 to keep ALL particles,"
        )
        sys.exit()

    # sort based on scores
    particles = [particle_parameters[pind] for pind in particle_parameters.keys()]
    particles.sort(key=lambda x: x.score, reverse=True)

    # save valid points in 3D
    best_particle = particles[0]
    valid_particles = np.array([best_particle.x_position_3d - (best_particle.shift_x/pixel_size), 
                                best_particle.y_position_3d - (best_particle.shift_y/pixel_size), 
                                best_particle.z_position_3d - (best_particle.shift_z/pixel_size)
                                ], ndmin=2)
    
    for idx, particle in enumerate(particles):
        pind = particle.particle_index
        if idx == 0:
            particle_parameters[pind].occ = particle_parameters[pind].occ if particle.score >= threshold else 0.0
            continue
        
        posx_with_shift = particle.x_position_3d - (particle.shift_x/pixel_size)
        posy_with_shift = particle.y_position_3d - (particle.shift_y/pixel_size)
        posz_with_shift = particle.z_position_3d - (particle.shift_z/pixel_size)

        # check if the point is close to previous evaluated points
        dmin = scipy.spatial.distance.cdist(
            np.array([posx_with_shift, posy_with_shift, posz_with_shift], ndmin=2), valid_particles
        ).min()
        if particle.score < threshold or dmin <= dist:
            particle_parameters[pind].occ = 0.0
        else:
            valid_particles = np.vstack((valid_particles, np.array([posx_with_shift, posy_with_shift, posz_with_shift])))

    scores = [particle_parameters[pind].score for pind in particle_parameters.keys()]

    extended_parameters.set_data(particles=particle_parameters, tilts=tilt_parameters)
    alignment_parameters.set_data(data=data, extended_parameters=extended_parameters)
    alignment_parameters.sync_particle_occ()
    alignment_parameters.to_binary(output=parameter_file)

    plot.histogram_particle_tomo(scores, threshold, tiltseries, "csp")


def particle_cleaning(parameters: dict):
    """ Particle cleaning (called from pyp_main) 

    Parameters
    ----------

    Returns
    ----------
        list: return box3d lines that stores ptlidx, 3d coord, scores and keep
    """
    
    try:
        filmlist_file = "{}.films".format(parameters["data_set"])
        films = np.loadtxt(filmlist_file, dtype='str', ndmin=1)
        # films = [film.strip() for film in f.readlines()]
    except:
        raise Exception(
            "{} does not exists".format("{}.films".format(parameters["data_set"]))
        )

    parameter_folder = project_params.resolve_path(parameters["clean_parfile"])
    
    assert Path(parameter_folder).exists(), f"{parameter_folder} does not exists."

    # decompress the parametere file (if needed), and copy it over to the particle filtering block
    parameter_folder_init = f"{parameters['data_set']}_r01_01"

    parameter_folder_current = Path("frealign", "maps", parameter_folder_init) 

    # just get the class function 
    p_obj = cistem_star_file.Parameters()

    # first check class selection, and generate one parfile with occ=0 to mark the discarded particles
    if parameters["clean_class_selection"]:
        sel = parameters["clean_class_selection"]
        selist = sel.split(",")
        selection = [int(x) for x in selist]
        # merge_align = parameters["clean_class_merge_alignment"]
        
        output_parfile = parameter_folder_current 
        os.makedirs(output_parfile,exist_ok=True)

        all_zero_list = pyp_metadata.merge_par_selection(parameter_folder, output_parfile, films, selection, parameters)
        parameter_folder = output_parfile
        parameters["refine_parfile"] = output_parfile

    else:
        if os.path.exists(parameter_folder) and parameter_folder.endswith(".bz2"):
            ref = int(parameter_folder.split("_r")[-1].split("_")[0]) 
            logger.info(f"Decompressing parameter files from {Path(parameter_folder).name}")
            frealign_parfile.Parameters.decompress_parameter_file_and_move(file=Path(parameter_folder), 
                                                                        new_file=parameter_folder_current, 
                                                                        micrograph_list=[f"{f}_r{ref:02d}" for f in films],
                                                                        threads=parameters["slurm_tasks"])
            parameter_folder = str(parameter_folder_current)  
            parameters["clean_parfile"] = Path(parameter_folder).absolute()

    # single class cleaning regard to box files
    if "spr" in parameters["data_mode"]:
        all_binary = os.listdir(parameter_folder)
        binary_list = [os.path.join(parameter_folder,filename) for image in films for filename in all_binary if image in filename if "_stat.cistem" not in filename and "_extended.cistem" not in filename]

        par_data = cistem_star_file.merge_all_binary_with_filmid(binary_list)

        if parameters["clean_spr_auto"]:
            # figure out optimal score threshold
            score_col = p_obj.get_index_of_column(column_code=cistem_star_file.SCORE)
            samples = np.sort( par_data[:, score_col] )
            # samples = pardata[:, 14].astype("f")

            threshold = 1.075 * statistics.optimal_threshold(
                samples=samples, criteria="optimal"
            )
            logger.info(f"Using {threshold:.2f} as optimal threshold")
        else:
            threshold = parameters["clean_threshold"]

        parameters, new_pardata = clean_particle_sprbox(par_data, threshold, parameters, metapath="./pkl")

        if parameters["clean_discard"]:
            film_col = p_obj.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
            film_ids = np.unique(new_pardata[:, film_col])
            if parameters["clean_check_reconstruction"]:
                clean_output_folder = str(parameter_folder_current)[:-3] + "_02_clean"
            else:
                clean_output_folder = str(parameter_folder_current) + "_clean"

            if os.path.exists(clean_output_folder):
                shutil.rmtree(clean_output_folder)
            
            os.makedirs(clean_output_folder)
            logger.info(f"Generating clean parameter file")
            with tqdm(desc="Progress", total=len(films), file=TQDMLogger()) as pbar:
                for film_id, image_name in enumerate(films):
                    if film_id in film_ids:
                        # get external data by reading the input binary files
                        input_image_binary = os.path.join(str(parameter_folder_current), image_name + "_r01.cistem")
                        output_binary = os.path.join(clean_output_folder, image_name + "_r01.cistem" )
                        image_par_obj = cistem_star_file.Parameters.from_file(input_image_binary)
                        image_array = new_pardata[new_pardata[:,film_col]==film_id]
                        # renumber the pid
                        id = np.arange(1, image_array.shape[0] + 1 )
                        image_array[:, 0] = id
                        image_par_obj.set_data(image_array)
                        image_par_obj.sync_particle_occ(ptl_to_prj=False)
                        image_par_obj.sync_particle_ptlid()
                        image_par_obj.to_binary(output_binary, extended_output=output_binary.replace(".cistem", "_extended.cistem"))
                    pbar.update(1)

            # update extract selection after cleaning
            parameters["extract_cls"] += 1

            if parameters.get("clean_export_clean"):
                generate_clean_spk(clean_output_folder, is_tomo=False)

            current_dir = Path().cwd()
            os.chdir(Path("frealign", "maps"))
            logger.info("Compressing parameter file")
            frealign_parfile.Parameters.compress_parameter_file(os.path.basename(clean_output_folder), 
                                                                os.path.basename(clean_output_folder) + ".bz2", 
                                                                parameters["slurm_merge_tasks"])
            os.chdir(current_dir)

    # we don't have box3d in spr, so this only works for tomo so far
    elif "tomo" in parameters["data_mode"]:

        parameter_files = [str(f) for f in Path(parameter_folder).glob("*.cistem") if "_extended.cistem" not in str(f) and "_stat.cistem" not in str(f)]

        mpi_args = []

        for parameter_file in parameter_files:
            mpi_args.append((parameter_file, 
                            parameters["clean_mintilt"], 
                            parameters["clean_maxtilt"], 
                            parameters["clean_dist"], 
                            parameters["clean_threshold"],
                            parameters["scope_pixel"]))
        if len(mpi_args) > 0:
            mpi.submit_function_to_workers(filter_particles, mpi_args, verbose=parameters["slurm_verbose"])

        # Statistics             
        clean_particle_count = 0
        all_particle_count = 0

        for parameter_file in parameter_files:
            extended_parameters = cistem_star_file.Parameters.from_file(parameter_file).get_extended_data()
            clean_particle_count += extended_parameters.get_num_clean_particles()
            all_particle_count += extended_parameters.get_num_particles()

        assert all_particle_count > 0, f"No particles left after filtering!"
            
        logger.warning(
            "{:,} particles ({:.1f}%) from {} tilt-series will be kept".format(
                clean_particle_count,
                (float(clean_particle_count) / all_particle_count * 100),
                len(parameter_files),
            )
        )

        if parameters["clean_discard"]:
            if parameters["clean_check_reconstruction"]:
                clean_parameter_folder = Path(str(parameter_folder_current)[:-3] + "_02_clean")
            else:
                clean_parameter_folder = Path(str(parameter_folder_current) + "_clean")
            if clean_parameter_folder.exists():
                shutil.rmtree(clean_parameter_folder)
            os.mkdir(clean_parameter_folder)
            
            mpi_args = []

            # completely remove particle projections from parfile and allboxes coordinate
            for parameter_file in parameter_files:
                mpi_args.append((parameter_file, clean_parameter_folder))
            
            if len(mpi_args) > 0:
                mpi.submit_function_to_workers(deep_clean_particles, mpi_args, verbose=parameters["slurm_verbose"])

            clean_micrograph_list = np.array([str(f.name).split("_r")[0] for f in Path(clean_parameter_folder).glob("*.cistem") if "_extended.cistem" not in str(f)])
            
            binning = parameters["tomo_rec_binning"]
            thickness = parameters["tomo_rec_thickness"]
            
            if parameters.get("clean_export_clean"):
                generate_clean_spk(clean_parameter_folder, binning=binning, thickness=thickness)

            current_dir = Path().cwd()
            os.chdir(Path("frealign", "maps"))
            frealign_parfile.Parameters.compress_parameter_file(str(clean_parameter_folder.name), 
                                                                str(clean_parameter_folder.name) + ".bz2", 
                                                                parameters["slurm_merge_tasks"])
            # cleanup
            shutil.rmtree(clean_parameter_folder.name)
            os.chdir(current_dir)

            # update film file
            if len(clean_micrograph_list) < len(parameter_files):
                os.rename(filmlist_file, filmlist_file.replace(".films", ".films_original"))
                np.savetxt(filmlist_file, clean_micrograph_list, fmt="%s")
                shutil.copy2(filmlist_file, filmlist_file.replace(".films", ".micrographs"))

        # cleanup, if needed
        if not parameters["clean_check_reconstruction"]:
            shutil.rmtree(parameter_folder, ignore_errors=True)

    return parameters 


def deep_clean_particles(parameter_file: str, clean_parameter_folder: Path):

    alignment_parameters = cistem_star_file.Parameters.from_file(input_file=parameter_file)
    alignment_parameters.sync_particle_occ()

    data = alignment_parameters.get_data()
    extended_parameters = alignment_parameters.get_extended_data()
    particle_parameters = extended_parameters.get_particles()
    tilt_parameters = extended_parameters.get_tilts()

    # remove projections that have 0 occ
    data = np.delete(
        data, np.argwhere(data[:, alignment_parameters.get_index_of_column(cistem_star_file.OCCUPANCY)] == 0.0), axis=0
    )

    # remove both binary files if there is no particle left
    if data.shape[0] == 0:
        return 
    
    data[:, alignment_parameters.get_index_of_column(cistem_star_file.POSITION_IN_STACK)] = np.array(
        [(_ + 1) for _ in range(data.shape[0])]
    )

    # remove particles that have 0 occ (occ should be sync between projections and particles)
    for pind in extended_parameters.get_particle_list():
        particle = particle_parameters[pind]
        if particle.occ == 0.0:
            particle_parameters.pop(pind)
        
    extended_parameters.set_data(particles=particle_parameters, tilts=tilt_parameters)
    alignment_parameters.set_data(data=data, extended_parameters=extended_parameters)
    alignment_parameters.to_binary(output=str(clean_parameter_folder / Path(parameter_file).name))
    return 


def clean_particle_sprbox(pardata, thresh, parameters, isfrealignx=False, metapath="./pkl"):

     # select particles based on Frealign score distribution 
     # modify pkl files to reflect the coordinates change
    p_obj = cistem_star_file.Parameters()
    field = p_obj.get_index_of_column(cistem_star_file.SCORE)
    occ_field = p_obj.get_index_of_column(cistem_star_file.OCCUPANCY)

    classification_pass = parameters["extract_cls"] + 1 
    occ_thresh = parameters["reconstruct_min_occ"]

    # filter particles that are too close to their neighbors
    pardata = remove_duplicates(pardata, field, occ_field, parameters)

    # discard_mask = np.logical_or(pardata[:, field] < thresh, pardata[:, 11] < occ_thresh)
    discard_mask = np.ravel(np.logical_or(pardata[:, field] < thresh, pardata[:, occ_field] < occ_thresh))
    logger.info(f"Score range [{min(pardata[:, field]):.2f},{max(pardata[:, field]):.2f}], threshold = {thresh:.2f}")
    logger.info(f"Occupancy range [{min(pardata[:, occ_field])},{max(pardata[:, occ_field])}], threshold = {occ_thresh}")

    discard = pardata[discard_mask]
    newinput_keep = pardata[np.logical_not(discard_mask)]
    global_indexes_to_remove = (discard[:, 0] - 1).astype("int").tolist()
    global_indexes_to_keep = (newinput_keep[:, 0] - 1).astype("int").tolist()

    logger.info(f"Particles to remove = {len(global_indexes_to_remove):,}")
    logger.info(f"Particles to keep = {len(global_indexes_to_keep):,}")

    thresh_ratio = len(global_indexes_to_keep) / pardata.shape[0]

    if not parameters["clean_discard"]: 
        if parameters["clean_class_selection"]:
            # class selection using exiting occ
            parameters["reconstruct_cutoff"] = "1"
            return parameters, newinput_keep
        else:
            if thresh_ratio >= 0.0001:
                parameters["reconstruct_cutoff"] = "%.4f" % thresh_ratio
                logger.info(f"Reconstruction cutoff changed to {thresh_ratio:.4f}")
                return parameters, newinput_keep
            else:
                Exception(f"Only {thresh_ratio * 100} percent of the particles will be used, which is too low, Abort.")
            
    else:
        # indexes are in base-0
        film_id = p_obj.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
        discard_filmid = np.unique(discard[:,film_id].astype(int))
        filmlist_file = "{}.films".format(parameters["data_set"])
        film_list = np.loadtxt(filmlist_file, dtype='str')
        discard_filmlist = film_list[discard_filmid]
        unchanged_filmlist = np.setdiff1d(film_list, discard_filmlist)
        
        emptyfilm = [] # record films with no particles left 

        # updating metadata (updating meta box somehow maybe not necessary after using cistem binary)
        is_spr=True
        current_dir = os.getcwd()
        os.chdir(metapath)
        box_header = ["x", "y", "Xsize", "Ysize", "inside", "selection"]

        # in case some images have no particle removed, update selection column only
        if len(unchanged_filmlist) > 0:
            logger.info("Updating particle metadata")
            with tqdm(desc="Progress", total=len(unchanged_filmlist), file=TQDMLogger()) as pbar:
                for film in unchanged_filmlist:
                    metadata = pyp_metadata.LocalMetadata(film + ".pkl", is_spr=is_spr)
                    boxx = metadata.data["box"].to_numpy()
                    boxx[:, 5] = classification_pass
                    df = pd.DataFrame(boxx, columns=box_header)
                    metadata.updateData({'box':df})    
                    metadata.write()
                    pbar.update(1)

        for id, film in enumerate(discard_filmlist):
            filmid = discard_filmid[id]
            
            metadata = pyp_metadata.LocalMetadata(film + ".pkl", is_spr=is_spr)
            if "box" in metadata.data.keys():
            
                boxx = metadata.data["box"].to_numpy()
                # current valid particles
                boxx_valid = np.argwhere(np.logical_and(boxx[:, 4] >= 1, boxx[:, 5] >= classification_pass - 1))
                boxx_valid = boxx_valid.ravel()
                
                ptls_infilm = pardata[pardata[:, film_id] == filmid] 
                assert len(boxx_valid) == ptls_infilm.shape[0], f"Valid particles in box {len(boxx_valid)} not equal to particles in parfile {ptls_infilm.shape[0]}"
                
                # the index want to keep from the current valid particles.
                ptl_to_keep = np.argwhere(np.logical_and(ptls_infilm[:, field] >= thresh, ptls_infilm[:, 11] >= occ_thresh)).ravel()
                
                if ptl_to_keep.shape[0] > 0:
                    boxx_keep_index = boxx_valid[ptl_to_keep]
                    # set clean particles to pass id
                    boxx[boxx_keep_index, 5] = classification_pass
                    # set other particles to lower level
                    all_indices = np.arange(boxx.shape[0])
                    complementary_indices = np.setdiff1d(all_indices, boxx_keep_index)
                    boxx[complementary_indices, 5] = classification_pass - 1 

                    # passed = boxx[np.logical_and(boxx[:, 4] >=1, boxx[:, 5]>= classification_pass)]
                    # logger.info(f"Particles before clean is {len(boxx_valid)}, particles after clean is {len(passed)}")
                else:
                    emptyfilm.append(film)
                
                df = pd.DataFrame(boxx, columns=box_header)
            
                metadata.updateData({'box':df})    
                metadata.write()

            else:
                Exception("No box info from Pickle file.")

        os.chdir(current_dir)

        # remove empty film from original film list
        if len(emptyfilm) > 0:
            indices = np.where(np.isin(film_list, emptyfilm))
            newfilms = np.delete(film_list, indices)
            
            os.rename(filmlist_file, filmlist_file.replace(".films", ".films_original"))
            np.savetxt(filmlist_file, newfilms, fmt="%s")
            shutil.copy2(filmlist_file, filmlist_file.replace(".films", ".micrographs"))

        # produce corresponding .par file
        # reorder index
        # Do this when split the new par data
        # newinput_keep[:, 0] = list(range(newinput_keep.shape[0]))
        # newinput_keep[:, 0] += 1

        # the new cistem binary will ignore the box, so this is not necessary. 
        if newinput_keep.shape[0] != len(global_indexes_to_keep):
            logger.error(
                "Number of clean particles does not match number of particles to keep: {0} != {1}".format(
                    newinput_keep.shape[0], len(global_indexes_to_keep)
                )
            )
            sys.exit()
        
        
        # re-number films start from 0
        # new_film_ids = newinput_keep[:, 7]
        # uniquefilm = np.unique(new_film_ids)
        # for i, old_id in enumerate(uniquefilm):
        #     film_mask = newinput_keep[:, 7] == old_id
        #     newinput_keep[film_mask, 7] = i

        """
        current_film = newinput_keep[0, 7]
        current_film = 0
        new_film_number = 0
        for i in range(newinput_keep.shape[0]):
            if newinput_keep[i, 7] != current_film:
                current_film = newinput_keep[i, 7]
                new_film_number += 1
            newinput_keep[i, 7] = new_film_number
        """
        # set occupancy to 100
        newinput_keep[:, 11] = 100.0

        # return clean_parfile.replace(".par", ".par.bz2")
        return parameters, newinput_keep


def update_boxx_files(global_indexes_to_remove, parameters, classification_pass, shifts=[]):

    pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])

    micrographs = "{}.micrographs".format(parameters["data_set"])
    inputlist = [line.strip() for line in open(micrographs, "r")]

    current_global_counter = previous_global_counter = micrograph_counter = 0

    boxx = np.array([1])

    global_indexes_to_remove.sort()

    threads = 12

    # read all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Reading box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(read_boxx_file_async, args=(micrograph, results))
    pool.close()

    # Wait for all processes to complete
    pool.join()

    boxx_dbase = dict()
    box_dbase = dict()
    while results.empty() == False:
        current = results.get()
        boxx_dbase[current[0]] = current[1]
        box_dbase[current[0]] = current[2]

    local_counter = 0

    # add infinity to force processing of micrograph list to the end
    import sys

    # global_indexes_to_remove.append( sys.maxint )

    logger.info("Updating particle database")

    # set last field in .boxx files to 0 for all bad particles
    for i in global_indexes_to_remove:

        while current_global_counter <= i:

            try:
                name = inputlist[micrograph_counter]
                boxxfile = "box/{}.boxx".format(name)
                boxfile = "box/{}.box".format(name)
            except:
                logger.exception(
                    "%d outside bounds %d", micrograph_counter, len(inputlist)
                )
                logger.exception("%d %d", current_global_counter, i)
                sys.exit()

            if os.path.exists(boxxfile):
                # boxx = numpy.loadtxt( boxxfile, ndmin=2 )
                boxx = boxx_dbase[name]
                box = box_dbase[name]

                if boxx.size > 0:

                    # only count particles that survived previous pass
                    # valid_particles = boxx[:,4].sum()
                    valid_particles = np.where(
                        np.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        1,
                        0,
                    ).sum()

                    previous_global_counter = current_global_counter
                    current_global_counter = (
                        current_global_counter + valid_particles
                    )  # 5th column contains actually extracted particles

                    # increment class membership pass for all particles of previous classification pass
                    boxxx = np.where(
                        np.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        classification_pass,
                        boxx[:, 5],
                    )

                    # alignments
                    counter = previous_global_counter
                    for j in range(boxx.shape[0]):
                        if boxx[j, 4] > 0 and boxx[j, 5] >= classification_pass - 1:
                            if len(shifts) > 0:
                                # boxx_dbase[name][j,0:2] = box[j,0:2] - boxx[j,2:4] / 2 - numpy.round( input[ counter, 4:6 ] / pixel )
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2]
                                    - boxx[j, 2:4] / 2
                                    + np.round(shifts[local_counter, :] / pixel)
                                )
                                local_counter += 1
                            else:
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2] - boxx[j, 2:4] / 2
                                )

                            counter += 1

                    boxx_dbase[name][:, 5] = boxxx
            else:
                logger.info("%s does not exist", boxxfile)

            if micrograph_counter < len(inputlist) - 1:
                micrograph_counter += 1
            # if micrograph_counter == len(inputlist):
            else:
                logger.info("Reached the end")
                break

        if micrograph_counter < len(inputlist):

            index_in_micrograph = 0
            global_counter = -1

            # print i, previous_global_counter
            while global_counter < i - previous_global_counter:
                if (
                    boxx[index_in_micrograph, 4] == 1
                    and boxx[index_in_micrograph, 5] >= classification_pass - 1
                ):
                    global_counter += 1
                index_in_micrograph += 1
            # boxx[index-1,5] = 0
            boxxx[index_in_micrograph - 1] = classification_pass - 1
            boxx_dbase[name][:, 5] = boxxx

    logger.info("Current global count %d", current_global_counter)
    # save all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Saving box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(
            write_boxx_file_async, args=(micrograph, boxx_dbase[micrograph])
        )
        # write_boxx_file_async( micrograph, boxx_dbase[micrograph] )
    pool.close()

    # Wait for all processes to complete
    pool.join()
    return boxx_dbase


def remove_duplicates(pardata: np.ndarray, field: int, occ_field: int, parameters: dict) -> np.ndarray:
    """remove_duplicates Remove particles in SPA that are too close to their neighbors after alignment by setting their score to -1

    Parameters
    ----------
    pardata : np.ndarray
        Refined parfile
    field : int
        index where the score column is
    """
    p_obj = cistem_star_file.Parameters()
    FILM_COL = p_obj.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
    shiftx = p_obj.get_index_of_column(cistem_star_file.X_SHIFT)
    shifty = p_obj.get_index_of_column(cistem_star_file.Y_SHIFT)
    x_coord = p_obj.get_index_of_column(cistem_star_file.ORIGINAL_X_POSITION)
    y_coord = p_obj.get_index_of_column(cistem_star_file.ORIGINAL_Y_POSITION)
    pixel_size = parameters["scope_pixel"]

    # filmlist_file = "{}.films".format(parameters["data_set"])
    # film_list = np.loadtxt(filmlist_file, dtype='str')

    films = np.unique(pardata[:, FILM_COL].astype("int"))
    
    total_removed = 0
    
    with tqdm(desc="Progress", total=len(films), file=TQDMLogger()) as pbar:
        for film in films:
            micrograph = pardata[pardata[:, FILM_COL] == film]
            # metadata = pyp_metadata.LocalMetadata("pkl/" + film_list[film] + ".pkl", is_spr=True)
            # box = metadata.data["box"].to_numpy()

            # micrograph[:, -2:] = box[:, :2]
            sort_pardata = micrograph[np.argsort(micrograph[:, field])][::-1]

            valid_points = np.array(
                [sort_pardata[0][x_coord] + (sort_pardata[0][shiftx]/pixel_size), sort_pardata[0][y_coord]] + (sort_pardata[0][shifty]/pixel_size), 
                ndmin=2
                )

            film_offset = np.where(pardata[:,FILM_COL]==film)[0][0]
    
            for idx, line in enumerate(sort_pardata):
                if idx == 0:
                    continue

                coordinate = np.array([line[x_coord] + (line[shiftx]/pixel_size), line[y_coord] + (line[shifty]/pixel_size)], ndmin=2)
                dmin = scipy.spatial.distance.cdist(coordinate, valid_points).min()
                if dmin <= parameters["clean_dist"]:
                    pardata[ film_offset + int(line[0]) - 1, occ_field] = 0.0
                    total_removed += 1
                else:
                    valid_points = np.vstack((valid_points, coordinate))
            
            pbar.update(1)

    logger.info(f"Removed {total_removed:,} duplicates closer than {parameters['clean_dist']} pixels")

    return pardata


def generate_clean_spk(input_parameter_folder, binning=1, output_path="./frealign/selected_particles", is_tomo=True, thickness=2048):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if is_tomo:
        cistem_binary = "*_extended.cistem"
    else:
        cistem_binary = "*_r01.cistem"

    inputfiles = glob.glob(os.path.join(input_parameter_folder, cistem_binary))

    if is_tomo:

        # clean previously exported coordinates
        try:
            [ os.remove(f) for f in glob.glob(os.path.join(output_path, "*.spk")) ]
        except:
            pass

        logger.info(f"Exporting clean particle coordinates for {len(inputfiles)} tomograms")
        with tqdm(desc="Progress", total=len(inputfiles), file=TQDMLogger()) as pbar:
            for file in inputfiles:
                coords = []
                parameter_obj = cistem_star_file.ExtendedParameters.from_file(file)

                particle_data = parameter_obj.get_particles()

                for i in particle_data.keys():
                    coords.append([particle_data[i].x_position_3d, particle_data[i].y_position_3d, particle_data[i].z_position_3d])

                clean_array = np.array(coords)

                if clean_array.size > 0:
                    clean_array[:, -1] = thickness - clean_array[:, -1]
                    clean_array = clean_array / binning

                    np.savetxt(file.replace("_extended.cistem", ".box"), clean_array, fmt='%.1f')

                    outfile = os.path.join(output_path, os.path.basename(file).replace("_extended.cistem", ".mod"))
                    command = f"{get_imod_path()}/bin/point2model -scat -sphere 5 {file.replace('_extended.cistem', '.box')} {outfile}"
                    run_shell_command(command, verbose=False)

                    run_shell_command("{0}/bin/imodtrans -T {1} {2}".format(get_imod_path(), outfile, outfile.replace('.mod', '.spk')),verbose=False)

                    os.remove(outfile)

                pbar.update(1)
        spk_files = len(glob.glob(os.path.join(output_path, "*.spk")))
        if spk_files > 0:
            logger.info(f"Clean particle coordinates in .spk format exported to {os.path.abspath(output_path)}")
        else:
            logger.warning("Unable to export clean particle coordiantes")
    else:
        p_obj = cistem_star_file.Parameters()
        x = p_obj.get_index_of_column(cistem_star_file.ORIGINAL_X_POSITION)
        y = p_obj.get_index_of_column(cistem_star_file.ORIGINAL_Y_POSITION)
        logger.info(f"Exporting clean particle coordinates for {len(inputfiles)} micrographs")
        with tqdm(desc="Progress", total=len(inputfiles), file=TQDMLogger()) as pbar:
            for file in inputfiles:
                parameter_obj = cistem_star_file.Parameters.from_file(file)
                coords = parameter_obj.get_data()[:, [x, y]]     
                outfile = os.path.join(output_path, os.path.basename(file).replace('_r01.cistem', '.box'))
                np.savetxt(outfile, coords, fmt='%.1f')
                pbar.update(1)
        box_files = len(glob.glob(os.path.join(output_path, "*.box")))
        if box_files > 0:
            logger.info(f"Clean particle coordinates in .box format exported to {os.path.abspath(output_path)}")
        else:
            logger.warning("Unable to export clean particle coordiantes")
