import datetime
import fcntl
import glob
import json
import math
import os
import copy
import shutil
import time
from ast import Or
from pathlib import Path
from re import L, T
from xml.sax import make_parser
from tqdm import tqdm

import numpy as np
import pickle
from typing import Tuple, Dict, List, Optional, Union

from pyp.analysis import plot, statistics
from pyp.analysis.geometry import divide2regions, findSpecimenBounds, get_tomo_binning
from pyp.analysis.geometry.pyp_convert_coord import read_3dbox
from pyp.analysis.occupancies import occupancy_extended
from pyp.analysis.scores import call_shape_phase_residuals
from pyp.analysis.plot import pyp_frealign_plot_weights
from pyp.inout.image import mrc, img2webp
from pyp.inout.metadata import frealign_parfile, isfrealignx, pyp_metadata, generate_ministar
from pyp.inout.metadata.cistem_star_file import *
from pyp.refine.frealign import frealign
from pyp.streampyp.web import Web
from pyp.streampyp.logging import TQDMLogger
from pyp.system import local_run, mpi, project_params, slurm
from pyp.system.db_comm import save_reconstruction_to_website, save_refinement_bundle_to_website
from pyp.system.logging import initialize_pyp_logger
from pyp.system.set_up import prepare_frealign_dir
from pyp.system.singularity import standalone_mode
from pyp.utils import get_relative_path, symlink_force, timer, symlink_relative
from pyp_main import csp_split

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def sort_particles_regions(
    particle_parameters, corners_squares, squaresize, per_particle=False
):
    """ Sort particles by sub-regions 

    Parameters
    ----------
    particle_coordinates : list[str]
        Particle 3D coordinates from 3dboxes file 
    corners_squares : list[list[float]]
        List that stores 3D coordinate of the bottom-left corner of every divided squares
    squaresize : list[float]
        List stores the size of squares

    Returns
    -------
    list[ list[str] ]
        List that stores the sorted particle indexes in squares
    """
    # let's have one more region to store particle completely out of bound (should be first checked by csp_tomo_swarm)
    ret = []
    if not per_particle:
        ret = [[] for i in range(len(corners_squares) + 1)]

    for particle_index in particle_parameters:

        particle = particle_parameters[particle_index]
        x, y, z = particle.x_position_3d, particle.y_position_3d, particle.z_position_3d

        if per_particle:
            ret.append([particle_index])
        else:
            find_square = False
            for idx_square, square in enumerate(corners_squares):

                # find which squares the particle belongs to
                if (
                    x >= square[0]
                    and y >= square[1]
                    and z >= square[2]
                    and x <= square[0] + squaresize[0]
                    and y <= square[1] + squaresize[1]
                    and z <= square[2] + squaresize[2]
                ):
                    # add particle index to the list
                    ret[idx_square].append(particle_index)
                    find_square = True
                    break

            if not find_square:
                ret[-1].append(particle_index)
                logger.debug(
                    "Particle [x = %f, y = %f, z = %f] is possibly out of bound."
                    % (x, y, z)
                )

    # sort the list based on the number of particles
    ret = sorted(ret, key=lambda x: len(x))

    return ret


def merge_alignment_parameters(
    parameter_file: str, mode: int, output_pattern: str = "_region????_????_????"
):
    """ Merge the splitted outputs from CSP into a main parfile

    Parameters
    ----------
    alignment_parameters : str
        The path of the main parfile
    output_pattern : str, optional
        The pattern used to search CSP results, by default "_region????_????_????"

    Returns
    -------
    numpy array
        Particle metadata in numpy array (before written to parfile)
    """
    extended_parameter_file = parameter_file.replace(".cistem", "_extended.cistem")
    # The alignment for projections has to be ordered 
    outputs = sorted(
        [
            file
            for file in glob.glob(
                parameter_file.strip(".cistem") + output_pattern + ".cistem"
            )
        ]
    )
    # original extended parameter file has to be the first one as the starting template, 
    # otherwise, the updated parameters will be overwritten by the old ones
    outputs_extended = [extended_parameter_file] + [file for file in 
                                                    glob.glob(
                                                        extended_parameter_file.strip("_extended.cistem") + output_pattern + "_extended.cistem"
                                                        )
                                                    ]

    assert (len(outputs) > 0), "No output parameter file is generated."

    if mode == 5 or mode == 4:
        assert (len(outputs_extended) > 0), "No extended parameter file is generated." 
    elif mode == 7 or mode == 2:
        assert (len(outputs_extended) > 0), "No extended parameter file is generated." 

    return Parameters.merge(input_files=outputs, input_extended_files=outputs_extended)    


def split_parameter_file(alignment_parameters, parameter_file, regions_list):
    """ Before frame refinement (CSP mode 5 & 6), this function splits the main parfile, which contains all particles in a tilt-series,
    into several sub-parfiles based on their 3D locations

    """

    parinfo_regions = []
    projection_data: np.ndarray = alignment_parameters.get_data()
    particle_parameters: dict = alignment_parameters.get_extended_data().get_particles()
    tilt_parameters: dict = alignment_parameters.get_extended_data().get_tilts()
    # have a copy of parameter data structure, in case we modify the original data
    template = copy.deepcopy(alignment_parameters)

    # go through each square to find parlines based on the particle index
    for region in regions_list:
        # if this square is not empty
        if len(region) > 0:
            # filter based on the list of particle index in this region
            parlines_filter = np.isin(projection_data[:, alignment_parameters.get_index_of_column(PIND)], region)
            parlines_region = projection_data[:][parlines_filter]

            if parlines_region.size != 0:
                parinfo_regions.append(parlines_region)

    split_files_list = []

    # write out split binary file for each region
    # re-assign each region with a new region index (for both data)
    for new_rind, region_projection_data in enumerate(parinfo_regions):
        split_parameter_file = parameter_file.replace(".cistem", "_region%04d.cistem" % (new_rind))

        # get unique pair of (tind, rind) in the projection data
        tind_rind = np.unique(region_projection_data[:, [template.get_index_of_column(TIND), template.get_index_of_column(RIND)]].astype("int"), axis=0)

        # add the tilt parameters to the region extended data and assign new region index
        region_tilt_parameters = dict()
        
        for pair in tind_rind:
            tind, rind = pair[0], pair[1]

            if tind not in region_tilt_parameters:
                region_tilt_parameters[tind] = dict()
            
            # update region_index in the dictionary and Tilt object
            tilt_object = tilt_parameters[tind][rind]
            tilt_object.region_index = new_rind
            region_tilt_parameters[tind][new_rind] = tilt_object

        extended_parameters = ExtendedParameters()
        extended_parameters.set_data(particles=particle_parameters, 
                                     tilts=region_tilt_parameters) 

        # modify RIND
        region_projection_data[:, template.get_index_of_column(RIND)] = new_rind

        template.set_data(data=region_projection_data, extended_parameters=extended_parameters)
        template.to_binary(split_parameter_file)

        # add (filename, list_of_pind, list_of_tind)
        split_files_list.append(
                (
                    split_parameter_file,
                    np.unique(region_projection_data[:, template.get_index_of_column(PIND)].astype("int")), 
                    np.unique(region_projection_data[:, template.get_index_of_column(TIND)].astype("int"))
                )
        )

    return split_files_list


def prepare_particle_cspt(
    name, parameter_file, alignment_parameters, parameters, grids=[1,1,1], use_frames=False
):
    """ This function prepares stuffs for frame refinement (CSP mode 5 & 6)
        1. Compute specimen bounds in xyz
        2. Divide specimen into multiple (overlapped) grids
        3. Sort particles into different grids 
        4. Prepare stack files for CSP 
        5. Split parfile by grids

    Parameters
    ----------
    name : str
        The name of the tilt-series
    dataset : str
        The name of the dataset
    main_parxfile : str
        The path of the main parfile
    main_stackfile : str
        The path of the main stack file
    parx_object : Parameters
        Frealign parameter object
    cpus : int
        The number of cpus/threads 
        
    Returns
    -------
    list
        list containing the names of splitted sub-parfiles that each will be independently read by CSP processes
    """
    
    metafile = "{}.pkl".format(name)
    if not os.path.exists(metafile):
        raise Exception(f"Metadata is required to run patch-based local refinement")
    metaobj= pyp_metadata.LocalMetadata(metafile)
    metadata = metaobj.data  

    particle_parameters: dict = alignment_parameters.get_extended_data().get_particles()

    # New flag: allow running without 3D coordinates by splitting per particle using 2D tracks
    use_2d_tracks_only = bool(parameters.get("csp_no_3d_points", False))

    if "tomo" in parameters["data_mode"].lower():

        if use_2d_tracks_only:
            # Bypass 3D bounds; split per particle. Corners/sizes are ignored when per_particle=True
            ptlidx_regions_list = sort_particles_regions(
                particle_parameters, corners_squares=[], squaresize=[], per_particle=True
            )
        else:
            binning = parameters["tomo_rec_binning"]
            tomox, tomoy, tomoz = metadata["tomo"].at[0, "x"] * binning, metadata["tomo"].at[0, "y"] * binning, metadata["tomo"].at[0, "z"] * binning

            # find out the bounds of the specimen (where particles are actually located)
            bottom_left_corner, top_right_corner = findSpecimenBounds(
                particle_parameters, [tomox, tomoy, tomoz]
            )

            # divide the specimen into several sub-regions
            corners, size_region = divide2regions(
                bottom_left_corner, top_right_corner, split_x=grids[0], split_y=grids[1], split_z=grids[2],
            )

            # sort particles into sub-regions
            if parameters["csp_frame_refinement"] and use_frames:
                per_particle = True
            else: 
                per_particle = False
            
            ptlidx_regions_list = sort_particles_regions(
                particle_parameters, corners, size_region, per_particle
            )

    else:
        """
        ptlind_col = 16
        
        # boxes = np.loadtxt(f"{name}.allboxes", ndmin=2)
        parfile = parx_object.data
        parfile = parfile[
                    np.unique(parfile[:, ptlind_col].astype("int"), return_index=True)[1], :
                    ]
        
        # compose cooridate list to [[ptlind, x, y, z],...[]] like tomo 
        boxes = [[int(parline[ptlind_col])] + list(box[:2]) + [0.0] 
                    for box, parline in zip(boxes, parfile)]
        """

        imagex, imagey = metadata["image"].at[0, "x"], metadata["image"].at[0, "y"]

        corners, size_region = divide2regions(
            bottom_left_corner=[0,0,0], 
            top_right_corner=[imagex, imagey, 1], 
            split_x=grids[0], 
            split_y=grids[1], 
            split_z=1,
        )
            
        ptlidx_regions_list = sort_particles_regions(
            particle_parameters, corners, size_region, per_particle=use_frames # do per particle only for frame refinement 
        )

    # split the parameter file based on regions
    split_parx_list = split_parameter_file(
        alignment_parameters, parameter_file, ptlidx_regions_list
    )

    return split_parx_list



@timer.Timer(
    "csp_local_merge", text="Total time elapsed (csp_local_merge): {}", logger=logger.info
)
def merge_movie_files_in_job_arr(
    movie_file,
    par_file,
    ordering_file,
    project_dir_file,
    output_basename,
    save_stacks=False,
):
    """
    When running multiple sprswarm runs in a single array job, merge the output
    """

    project_path = open(project_dir_file).read()
    mp = project_params.load_pyp_parameters(project_path)
    fp = mp

    iteration = fp["refine_iter"]

    if iteration == 2:
        classes = 1
        mp["class_num"] = 1 
    else:
        classes = int(project_params.param(fp["class_num"], iteration))
    
    # workaround for case when job array has only one component
    if "_" not in output_basename:
        output_basename += "_1"

    if os.path.exists(movie_file):
        with open(movie_file) as f:
            movie_list = [line.strip() for line in f]
        with open(par_file) as f:
            par_list = [line.strip() for line in f]

        # check that all files are present
        assert len(movie_list) == len(
            par_list
        ), "movie files and par files must be of equal length"
        for movie, parfile in zip(movie_list, par_list):
            not_found = "{} not found"
            if not os.path.exists(movie):
                logger.warning(not_found.format(movie))
            if not os.path.exists(parfile):
                logger.warning(not_found.format(parfile))

        if fp["extract_stacks"]:

            saved_path =os.path.join(
                        project_path, "relion", "stacks",
                    )

            if not os.path.exists(saved_path):
                os.makedirs(saved_path)

            for stack in movie_list:
                shutil.copy2( stack, os.path.join(saved_path, os.path.basename(stack).replace(".mrc", ".mrcs")) )
                logger.info(f"Stack {Path(stack).name} saved to {saved_path}")

            save_stacks = True
            global_image = os.path.join( project_path, mp["data_set"] + ".films" ) 
            global_imagelist = np.loadtxt(global_image, dtype=str).ravel()

            mpi_funcs, mpi_args = [ ], [ ]
            imagesize = mp["extract_box"]
            image_binning = mp["extract_bin"]
            doseRate = mp["scope_dose_rate"]

            for par in par_list:
                micrograph_path = os.path.join(project_path, "mrc", Path(par).stem.split("_r01")[0] + ".mrc")

                filename = os.path.join(saved_path, os.path.basename(stack).replace(".mrc", ".star")) 
                # par_obj = Parameters.from_file(par)
                # par_obj.to_star(imagesize, image_binning, micrograph_path, filename, global_imagelist, doseRate)
                mpi_args.append([(Parameters.from_file(par), imagesize, image_binning, micrograph_path, filename, global_imagelist, doseRate)])
                mpi_funcs.append(Parameters.to_star)

            mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=mp["slurm_verbose"], silent=True)

        with timer.Timer(
            "merge_stack", text="Merging particle stack took: {}", logger=logger.info
        ):
            if len(movie_list) > 1:
                logger.info(f"Merging movie files into {output_basename}_stack.mrc")
                mrc.merge_fast(movie_list,f"{output_basename}_stack.mrc",remove=True)
            else:
                try:
                    os.symlink(movie_list[0], str(output_basename) + "_stack.mrc")
                except:
                    pass

        logger.info("Merging parameter files")

    # if intermediate reconstructions are empty, raise flag
    else:
        for class_index in range(classes):
            current_class = class_index + 1
            filename = f"{output_basename}_r{current_class:02d}.empty"
            output_dir = os.path.join(project_path,"frealign","scratch")
            os.makedirs( output_dir, exist_ok=True)
            Path(os.path.join(output_dir, filename)).touch()
        # nothing else to do here
        return

    # refine3d film output = 1 but csp film output == 0
    # we just get the film id from the data
    sample_data = Parameters.from_file(par_list[0])
    film_col = sample_data.get_index_of_column(IMAGE_IS_ACTIVE)
    film_start = int(sample_data.get_data()[0, film_col])

    for class_index in range(classes):

        current_class = class_index + 1

        merged_par_file = str(output_basename) + "_r%02d.cistem" % (current_class)
        new_par_list = []
        for p in par_list:
            new_par_list.append(p.replace("_r01", "_r%02d" % current_class))

        tilt_json = merged_par_file.replace(".cistem", ".json") # a dictinary file for tilt parameters 
        tilts_dict = {}
        if len(new_par_list) == 1:
            # src_path = os.path.realpath(new_par_list[0])
            shutil.copy2(new_par_list[0], merged_par_file)
            extended_cistem = ExtendedParameters.from_file(new_par_list[0].replace(".cistem", "_extended.cistem"))
            tilt_dict = extended_cistem.get_tilts()
            tind_angle_dict = {int(t): tilt_dict[t][0].angle for t in tilt_dict.keys()}
            tilts_dict[0 + film_start] = tind_angle_dict # film id start from 0

        else:
            merged_pardata = merge_all_binary_with_filmid(new_par_list, read_extend=False)
            merge_par_obj = Parameters()
            merge_par_obj.set_data(data=merged_pardata)
            merge_par_obj.to_binary(merged_par_file)
            for i, f in enumerate(new_par_list):
                extended_cistem = ExtendedParameters.from_file(f.replace(".cistem", "_extended.cistem"))
                tilt_dict = extended_cistem.get_tilts()
                tind_angle_dict = {int(t): tilt_dict[t][0].angle for t in tilt_dict.keys()}
                tilts_dict[i + film_start] = tind_angle_dict

        with open(tilt_json, 'w') as j:
            json.dump(tilts_dict, j, indent=4)

        # TODO: for bundle size > 1, this may cause issue
        shutil.copy2(new_par_list[0].replace(".cistem", "_extended.cistem"), merged_par_file.replace(".cistem", "_extended.cistem"))

        shutil.copy2(
            par_list[0]
            .replace("/maps/", "/scratch/")
            .replace(".cistem", ".mrc")
            .replace("_r01_", "_r%02d_" % current_class),
            str(output_basename) + "_r%02d.mrc" % (current_class),
        )

    logger.info("Running reconstruction")

    # change occupancy after refinement
    if classes > 1 and not mp["refine_skip"]:

        logger.info("Updating occupancies after local merge")
        # update occupancies using LogP values
        current_path = os.getcwd()
        occupancy_extended(mp, output_basename, classes, image_list=par_list, parameter_file_folders=current_path, local=True)

    for class_index in range(classes):

        if classes > 1:
            logger.info(
                "## Running reconstruction for class {} of {} ##".format(class_index+1, classes)
            )

        current_class = class_index + 1

        par_binary = str(output_basename) + "_r%02d.cistem" % current_class
        # current_block_par_obj = Parameters.from_file(par_binary)
        
        # link statistics file
        dataset_name = mp["data_set"]
        decompressed_parameter_file_folder = os.path.join(project_path, "frealign", "maps", dataset_name + "_r%02d_%02d" % (current_class, iteration - 1))
        remote_par_stat = os.path.join(decompressed_parameter_file_folder, dataset_name + "_r%02d_stat.cistem" % (current_class))
        
        # if not exist, will use the input parameter file to calculate statistics in reconstruct3d
        if os.path.exists(remote_par_stat):
            try:
                os.symlink(remote_par_stat, par_binary.replace(".cistem", "_stat.cistem"))
            except:
                pass # file may exist
        
        new_par_list = []
        for p in par_list:
            new_par_list.append(p.replace("_r01", "_r%02d" % current_class))

        # generate tsv files for Artix display
        if "tomo" in mp["data_mode"]:
            star_output = os.path.join(project_path, "frealign", "artiax")
            binning = mp["tomo_rec_binning"]
            z_thicknes = mp["tomo_rec_thickness"]
            generate_ministar( new_par_list, z_thicknes, binning, cls=current_class, output_path=star_output)
        
        """
        if classes > 1:
            # backup the current par for later recovery
            shutil.copy(current_block_par, current_block_par.replace(".par", ".paro"))

        current_block_data = frealign_parfile.Parameters.from_file(
            current_block_par
        ).data

        # ptl_id_now = current_block_data[-1, 0]
        film_id_now = current_block_data[-1, film_col]
        film_total = np.loadtxt(os.path.join(project_path, mp["data_set"] + ".films"), dtype=str, ndmin=2)
        if classes > 1 and film_total.shape[0] > 1 and iteration > 2:

            frealign_parfile.Parameters.add_lines_with_statistics(
                current_block_par, 
                current_class, 
                project_path, 
                is_frealignx=is_frealignx, 
                )

        else:
            logger.info("Skip modifying metadata for 1 film only")
        """

        # run reconstructions
        run_reconstruction(
            output_basename,
            mp,
            "merged_recon" + "_r%02d" % current_class,
            "../output" + "_r%02d" % current_class,
            save_stacks,
            current_class,
            iteration,
        )

        t = timer.Timer(text="Saving logs and reconstruction took: {}", logger=logger.info)
        t.start()

        # save logs
        mypath = os.path.join(
            "merged_recon" + "_r%02d" % current_class, "*_0000001_*.log"
        )
        log_files = glob.glob(mypath)
        if len(log_files) > 0:
            target_name = (
                mp["data_set"]
                + "_r%02d_%02d" % (current_class, iteration)
                + "_mreconst.log"
            )
            target = os.path.join(project_path, "frealign", "log", target_name)
            shutil.copy2(log_files[0], target)
            # send output to interface
            if mp['slurm_verbose']:
                with open(log_files[0]) as f:
                    logger.info(f.read())

        # copy recon over to project dir
        if mp["refine_parfile_compress"]:
            compressed = True
        else:
            compressed = False

        save_reconstruction(
            f"{output_basename}_r{current_class:02d}", 
            project_path,
            iteration,
            output_folder="output" + "_r%02d" % current_class,
            threads=mp["slurm_tasks"],
            compress=compressed
        )
        t.stop()

    # keep track of movies that have been processed successfully
    for movie in movie_list:
        local_movie = os.path.basename(movie).replace("_stack.mrc", "")
        path = Path(
            os.path.join(project_path, "frealign", "scratch", ".{}".format(local_movie))
        )
        path.touch()

        # save metadata  
        convert = True

        if "spr" in mp["data_mode"]:
            is_spr = True
        else:
            is_spr = False

        if convert:
            cwd = os.getcwd()
            os.chdir(local_movie)
            metaname = local_movie + ".pkl"
            metadata = pyp_metadata.LocalMetadata(metaname, is_spr=is_spr)
            metadata.loadFiles()

            # copy the metadata to project folder.
            meta_path = os.path.join(project_path, "pkl")
            if not os.path.isdir(meta_path):
                os.mkdir(meta_path)
            if not os.path.isfile(os.path.join(meta_path, metaname)):  
                shutil.copy2(metaname, meta_path)
            os.chdir(cwd)

def save_ordering(project_path, basename, order_number, ordering_file="ordering.txt"):
    save_file = os.path.join(project_path, "frealign", "scratch", ordering_file)
    dest_path = os.path.dirname(save_file)
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except:
            pass
    with open(save_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write("{} {}\n".format(basename, order_number))
        fcntl.flock(f, fcntl.LOCK_UN)
    logger.info("ordering information written")


def save_reconstruction(
    output_basename, project_path, iteration, output_folder="output", threads=1, compress=True
):
    logger.info("Copying reconstruction to project folder")
    dest_path = os.path.join(project_path, "frealign", "scratch")
    log_path = os.path.join(project_path, "frealign")
    weight_path = os.path.join(project_path, "frealign", "weights")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)

    # all_files = os.listdir(output_folder)

    output_name = output_basename
    class_label = output_basename.split("_")[-1]
    # move over all the binary files (each represent on tilt-series) in a given bundle to output folder
    [os.rename(binary_file, Path(output_folder) / Path(binary_file).name) for binary_file in glob.glob(f"*/frealign/maps/*{class_label}*.cistem")]
    # move the used binary for global merge 
    [os.rename(binary_file, Path(output_folder) / Path(binary_file).name) for binary_file in glob.glob(f"*{class_label}*_used.cistem")]

    # copy only dumped mrc and .par files
    saved_path = os.getcwd()
    os.chdir(output_folder)

    for file in glob.glob("*.svgz"):
        shutil.copy(file, log_path)

    for file in glob.glob("*_weights.svgz"):
        # convert image to webp format and save to target directory: log_path
        shutil.copy(file,os.path.join(log_path, file.replace(".svgz","_local.svgz")))

        if not os.path.exists(weight_path): 
            os.makedirs(weight_path)
        try:
            shutil.copy2(file.replace("_weights.svgz", "_scores.txt"), os.path.join(weight_path, file.replace("_weights.svgz", "_scores.txt")))
        except:
            logger.warning("No scores.txt exist from the reconstruction results, probably using lblur")

        # save weights to website
        save_refinement_bundle_to_website(file.replace("_weights.svgz",""), iteration)

    # delete the broken links that have already been deleted by local merge
    local_run.run_shell_command("find . -xtype l -delete", verbose=False)
    
    file_name_suffix = "_"
    
    dumpfiles = glob.glob("*_???????_???????.mrc") + glob.glob("*_map?_n*.mrc") + glob.glob("*.cistem")
    if compress:
        compressed_file = output_name + ".bz2"
        frealign_parfile.Parameters.compress_parameter_file(
            " ".join(dumpfiles), os.path.join(dest_path, compressed_file + file_name_suffix), threads
        )
        # in case file starts to be decompressed when undergoing compression (due to live decompression)
        os.rename(os.path.join(dest_path, compressed_file + file_name_suffix), os.path.join(dest_path, compressed_file))
        os.chdir(saved_path)
        logger.info("Compressing intermediate files to scratch folder")
    else:
        for file in dumpfiles:
            shutil.copy2(file, os.path.join(dest_path, file + file_name_suffix))
        for file in dumpfiles:
            os.rename(os.path.join(dest_path, file + file_name_suffix), os.path.join(dest_path, file))
        os.chdir(saved_path)



def get_number_of_intermediate_reconstructions(mp):
    """Figure out how to manage resources for 3D reconstruction

    Parameters
    ----------
    mp : parameters
        pyp parameters

    Returns
    -------
    groups, cpus_per_group
        Number of reconstructions and threads to use for each process
    """
    frealign_cpus_per_group = 1
    # use multiple threads
    cpus_per_group = frealign_cpus_per_group
    groups = int(mp["slurm_tasks"] / cpus_per_group)
    return groups, cpus_per_group


@timer.Timer(
    "run_reconstruction", text="Reconstruction took: {}", logger=logger.info
)
def run_reconstruction(
    name,
    mp,
    merged_recon_dir="merged_recon",
    output_folder="../output",
    save_stacks=False,
    ref=1,
    iteration=2,
):
    fp = mp

    scratch = os.environ["PYP_SCRATCH"] = ""

    # remove dirs if exist
    shutil.rmtree(merged_recon_dir, ignore_errors=True)

    if not os.path.exists(merged_recon_dir):
        os.makedirs(merged_recon_dir)

    header = mrc.readHeaderFromFile(name + "_stack.mrc")
    frames = header["nz"]

    fp["refine_dataset"] = name

    parameter_file = name + "_r%02d.cistem" % (ref)

    alignment_parameters = Parameters.from_file(parameter_file)

    # create shaped _used par file
    call_shape_phase_residuals(
        parameter_file,
        parameter_file.replace(".cistem", "_used.cistem"),
        fp,
        iteration,
    )

    # if needed save the stack files
    if save_stacks:
        os.symlink(
            "../" + name + "_stack.mrc",
            os.path.join(merged_recon_dir, name + "_stack.mrc"),
        )

    curr_dir = os.getcwd()

    os.chdir(merged_recon_dir)

    # AB - Create directories if needed
    prepare_frealign_dir()

    # Split reconstruction into several jobs to run reconstruct3d in parallel
    groups, cpus_per_group = get_number_of_intermediate_reconstructions(mp)

    # get the number of frames
    # TODO this is not 100% matching the projection data, since only link one image's extended.cistem as decoy
    # But we're not actually using the num_tilts for reconstruct_3d 
    num_tilts = alignment_parameters.get_extended_data().get_num_tilts()
    frames_per_tilt = alignment_parameters.get_num_frames()

    commands, count = local_run.create_split_commands(
        mp,
        name,
        frames,
        groups,
        scratch,
        step="reconstruct3d",
        num_frames=num_tilts,
        ref=ref,
        iteration=iteration,
    )

    # make sure reconstruct3d runs in parallel
    prefix = "export OMP_NUM_THREADS={0}; export NCPUS={0}; ".format(cpus_per_group)
    for i in range(len(commands)):
        commands[i] = prefix + commands[i]

    recon_st = str(datetime.datetime.now())
    recon_S = time.perf_counter()

    mpi.submit_jobs_to_workers(commands, os.getcwd())
    recon_E = time.perf_counter()
    recon_T = recon_E - recon_S
    timer.Timer.timers.update({"reconstruct3d_splitcom" :{"elapsed_time": recon_T, "start_time": recon_st, "end_time": str(datetime.datetime.now())}})

    if mp["reconstruct_dose_weighting_enable"]:
        if os.path.exists("weights.txt"):
            pyp_frealign_plot_weights.plot_weights(name, "weights.txt", num_tilts, frames_per_tilt, mp["extract_box"], mp["scope_pixel"] * mp["extract_bin"])
        else:
            logfile = commands[0].splitlines()[0].split(" ")[-2].replace(" ","")
            if os.path.exists(logfile):
                with open(logfile) as f:
                    errors = f.read()
                    logger.warning(errors)
                    if "caught" in errors:
                        raise Exception(errors)

    # files that will be saved to /nfs
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder)

    files_to_keep = glob.glob(name + "*")
    for f in files_to_keep:
        symlink_relative(os.path.join(os.getcwd(), f), os.path.join(output_folder, f))

    # perform local merge
    if "cc" not in mp["refine_metric"].lower() and groups > 2:
        frealign.local_merge_reconstruction()

    # save the parameter files
    project_params.save_pyp_parameters(mp, output_folder)
    # project_params.save_fyp_parameters(fp, output_folder)

    # change back to working dir
    os.chdir(curr_dir)


def merge_check_err_and_resubmit(
    parameters, input_dir, micrographs, iteration=2
):
    """ Re-submit cspswarm for missing tilt-series due to slurm errors, missing files ... etc.  

    Parameters
    ----------
    parameters : dict
        PYP configuration parameters
    input_dir : str, optional
        The directory storing reconstruct3d dumped files, parfiles ...etc, by default "scratch"


    """
    # we're supposed to be in frealign/{input_dir} directory, go back to project dir
    cur_dir = os.getcwd()
    os.chdir(input_dir)
    os.chdir("../../")

    # get the number of tilt-series to be processed
    movies = [line.strip() for line in micrographs.keys()]
    num_movies = len(movies)

    # read list of processed movies
    # movies_done = [ line.strip() for line in open(orderfile, "r") ]

    # movies_resubmit = [ movie for movie in movies if movie not in movies_done ]

    movies_resubmit = []
    for movie in movies:
        if not os.path.exists(os.path.join(input_dir, "." + movie)):
            movies_resubmit.append(movie)

    if len(movies_resubmit) > 0:
        
        os.chdir("swarm")

        if not os.path.exists(".cspswarm_retry"):
            parameters["slurm_merge_only"] = False
            slurm.launch_csp(micrograph_list=movies_resubmit,
                            parameters=parameters,
                            swarm_folder=Path().cwd(),
                            )
            logger.warning(f"Successfully re-submitted {len(movies_resubmit):,} failed job(s)")

            # save flag to indicate failure
            Path(".csp_current_fail").touch()
            Path(".cspswarm_retry").touch()

        else:
            logger.error("Giving up retrying...")
            os.remove(".cspswarm_retry")

            # raise error flag
            Path(".cspswarm.error").touch()
            raise Exception(f"{len(movies_resubmit)} jobs failed to run or reached the walltime. Stopping")
    else:
        logger.info(
            "All series were successfully processed, start merging reconstructions"
        )
        os.chdir(cur_dir)


@timer.Timer(
    "run_mpi_reconstruction", text="Function with merge3d took: {}", logger=logger.info
)
def run_mpi_reconstruction(
    ref, pattern, dataset_name, iteration, mp, fp, input_dir, orderings,
):
    # $PYP_SCRATCH/frealign/scratch
    # NOTE: intermediate reconstructions and parameter files (.cistem) are all in this directory
    local_input_dir = os.getcwd()

    # check if there are any symlinks from previous iterations and remove them
    curr_files = [f for f in os.listdir(os.getcwd()) if dataset_name in f]
    [os.remove(f) for f in curr_files if os.path.islink(f)]

    metric = project_params.param(fp["refine_metric"], iteration)

    # performing symlinks and sorting
    _ = rename_csp_local_files(
        dataset_name, local_input_dir, orderings, pattern, metric
    )

    iteration = mp["refine_iter"]

    is_tomo = "tomo" in mp["data_mode"]

    if is_tomo:
        tilt_min = mp["csp_UseImagesForRefinementMin"]
        tilt_max = mp["csp_UseImagesForRefinementMax"]
    else:
        tilt_min = 0
        tilt_max = 0

    # a folder storing all the parameter files (instead of merged one)
    # NOTE: dataset_name = parameters['data_set'] + f"_r{ref:02d}"
    parameter_file_folder = f"{dataset_name}_{iteration:02d}"

    arg_scores = True

    with timer.Timer(
        "plot_used_png", text = "Plot used particles pngs took: {}", logger=logger.info
    ):

        if float(project_params.param(fp["reconstruct_cutoff"], iteration)) >= 0:
           # creat bild file from used.par file
            mpi_funcs, mpi_args = [], []

            binary_list = glob.glob(os.path.join(local_input_dir, "*_r%02d_used.cistem" % ref))
            merge_used_par_data = merge_all_binary_with_filmid(binary_list=binary_list, read_extend=False)
            merged_used_par = f"{dataset_name}_used_merged.cistem"
            # save the new merged file for the following function
            par_obj = Parameters()
            par_obj._input_file = merged_used_par
            par_obj.set_data(merge_used_par_data)
            par_obj.to_binary()
            bild_output = os.path.join(os.path.dirname(input_dir),"maps",f"{dataset_name}_{iteration:02d}.bild")
            # plot.par2bild(merged_used_par, bild_output, fp)
            mpi_funcs.append(plot.par2bild)
            mpi_args.append( [( merged_used_par, bild_output, fp)] )
            # 
            # plot using all particles
            binary_list = glob.glob(os.path.join(local_input_dir, "*_r%02d.cistem" % ref))
            merge_par_data = merge_all_binary_with_filmid(binary_list=binary_list, read_extend=False)
            # arg_input = f"{dataset_name}.par" # this should be a array of parameters 
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False

            mpi_funcs.append(plot.generate_plots)
            mpi_args.append([(
                merge_par_data,
                dataset_name + "_%02d" % iteration,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                is_tomo,
                arg_dump,
                tilt_min,
                tilt_max,
            )])

            # plot using used particles
            # arg_input = f"{dataset_name}_used.par"
            arg_angle_groups = 25
            arg_defocus_groups = 25
            arg_dump = False
            mpi_funcs.append(plot.generate_plots)
            mpi_args.append([(
                merge_used_par_data,
                dataset_name + "_%02d_used" % iteration,
                arg_angle_groups,
                arg_defocus_groups,
                arg_scores,
                is_tomo,
                arg_dump,
                tilt_min,
                tilt_max,
            )])

            mpi.submit_function_to_workers(mpi_funcs, mpi_args, verbose=fp["slurm_verbose"])

            # transfer files to maps directory
            for file in glob.glob(dataset_name + "*_prs.png"):
                try:
                    shutil.move(file, "../maps")
                except:
                    pass

            # do the statistics calculation here
            stat_array_mean = np.mean(merge_used_par_data, axis=0)
            stat_array_var = np.var(merge_used_par_data, axis=0)
            # only projecton parameters here
            stat = Parameters()
            stat.set_data(np.vstack((stat_array_mean, stat_array_var)))
            stat.to_binary( os.path.join(local_input_dir, dataset_name + "_stat.cistem" ) )
            occ_col = stat.get_index_of_column(OCCUPANCY)
            # save an occ array for ploting in final merge
            occ_data = merge_used_par_data[:, occ_col]
            saved_occ = os.path.join(input_dir, "occ_only_r%02d.npy" % ref) # save in project folder/frealign/scratch
            np.save(saved_occ, occ_data)

    # combine 2D plots from used particles and global statistics for histograms
    # read saved pickle files 
    with open(f"{dataset_name}_{iteration:02d}_temp.pkl", 'rb') as f1:
        plot_outputs = pickle.load(f1)
    with open(f"{dataset_name}_{iteration:02d}_meta_temp.pkl", 'rb') as f2:
        metadata = pickle.load(f2)
    with open(f"{dataset_name}_{iteration:02d}_used_temp.pkl", 'rb') as f3:
        plot_outputs_used = pickle.load(f3)
    with open(f"{dataset_name}_{iteration:02d}_used_meta_temp.pkl", 'rb') as f4:
        metadata_used = pickle.load(f4)

    consolidated_plot_outputs = plot_outputs.copy()
    consolidated_plot_outputs["def_rot_histogram"] = plot_outputs_used["def_rot_histogram"]
    consolidated_plot_outputs["def_rot_scores"] = plot_outputs_used["def_rot_scores"]

    consolidated_metadata = metadata.copy()
    consolidated_metadata["particles_used"] = metadata_used["particles_used"]
    consolidated_metadata["phase_residual"] = metadata_used["phase_residual"]

    temp_par_obj = Parameters()
    occ_col = temp_par_obj.get_index_of_column(OCCUPANCY)
    score_col = temp_par_obj.get_index_of_column(SCORE)
    # samples = np.array(frealign.get_phase_residuals(pardata,f"{dataset_name}.par",fp,2))
    mask = np.logical_and(merge_par_data[:, occ_col] > 0, np.isfinite(merge_par_data[:, score_col]))
    samples = merge_par_data[mask][:, score_col]
    threshold = 1.075 * statistics.optimal_threshold(
        samples=samples, criteria="optimal"
    )
    consolidated_metadata["phase_residual"] = threshold

    # perform final merge
    # hack os.environ['PYP_SCRATCH']
    local_scratch = os.environ["PYP_SCRATCH"]
    os.environ["PYP_SCRATCH"] = local_input_dir

    logger.info("Merging intermediate reconstructions")
    frealign.merge_reconstructions(mp, iteration, ref)

    os.environ["PYP_SCRATCH"] = local_scratch

    with timer.Timer(
        "output reconstruction results", text = "Final output reconstructions took: {}", logger=logger.info
    ):
        # copy log and png files

        # write compressed file to maps directory
        output_folder = os.path.join(os.path.dirname(input_dir), "maps")

        saved_path = os.getcwd()
        os.chdir(local_input_dir)
        parameter_files = glob.glob("*_r%02d.cistem" % ref) + glob.glob("*_r%02d_stat.cistem" % ref) + glob.glob("*_r%02d_extended.cistem" % ref) 
        os.mkdir(parameter_file_folder)
        [os.rename(f, Path(parameter_file_folder) / f) for f in parameter_files]

        if fp["refine_parfile_compress"]:
            compressed_file = os.path.join(
                output_folder, parameter_file_folder + ".bz2"
            )
            frealign_parfile.Parameters.compress_parameter_file(
                parameter_file_folder, compressed_file, fp["slurm_merge_tasks"]
            )
        else:
            os.rename(parameter_file_folder, Path(output_folder) / parameter_file_folder)
        
        os.chdir(saved_path)

        # append merge log
        reclogfile = "../log/%s_%02d_mreconst.log" % (dataset_name, iteration)
        outputlogfile = os.path.join(os.path.dirname(input_dir), "log/%s_%02d_mreconst.log" % (dataset_name, iteration))
        with open(outputlogfile, "a") as fw:
            with open(reclogfile) as fr:
                fw.write(fr.read())

        # save statistics file
        stats_file_name = os.path.join(output_folder, f"{dataset_name}_statistics.txt")
        res_file_name = f"{dataset_name}.res"

        # smooth part FSC curves
        """
        if project_params.param(mp["refine_metric"], iteration) == "new" and os.path.exists(stats_file_name):

            plot_name = os.path.join(
                output_folder, "%s_%02d_snr.png" % (dataset_name, iteration)
            )

            postprocess.smooth_part_fsc(str(stats_file_name), plot_name)
        """

        if os.path.exists(res_file_name):
            com = (
                """grep -A 10000 "C  NO.  RESOL  RING RAD" {0}""".format(res_file_name)
                + """ | grep -v RESOL | grep -v Average | grep -v Date | grep C | awk '{if ($2 != "") printf "%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f\\n", $2, $3, $4, $6, $7, $8, $9}' > """
                + str(stats_file_name)
            )
            local_run.run_shell_command(com, verbose=False)

        elif os.path.exists(f"{dataset_name}_statistics.txt"):
            shutil.copy2(f"{dataset_name}_statistics.txt", stats_file_name)

        # save what is worth to original frealing/maps
        for file in (
            [ f"../maps/{dataset_name}_{iteration:02d}.mrc", f"../maps/{dataset_name}_{iteration:02d}_raw.mrc"]
            + glob.glob(f"../maps/*_r{ref:02d}_???.txt")
            + glob.glob(f"./*statistics.txt")
            + glob.glob("../maps/*half*.mrc")
            + glob.glob("../maps/*crop.mrc")
            + glob.glob("../maps/*scores.svgz")
            + glob.glob("../maps/*.webp")
        ): 
            if os.path.exists(file):
                shutil.copy2(file, output_folder)

        if True:
            fsc_file = os.path.join("../maps", fp["refine_dataset"] + "_r%02d_fsc.txt" % ref)
            FSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)

            # send reconstruction to website
            save_reconstruction_to_website(
                dataset_name + "_%02d" % iteration, FSCs, consolidated_plot_outputs, consolidated_metadata
            )

@timer.Timer(
    "run_merge", text="Total time elapsed: {}", logger=logger.info
)
def run_merge(input_dir="scratch", ordering_file="ordering.txt"):

    # we are originally in the project directory
    project_dir = os.getcwd()

    # now switch to frealign/scratch directory where all the .bz2 files are
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(input_dir)

    # load pyp params from main folder
    mp = project_params.load_pyp_parameters("../..")
    fp = mp

    not_retrying = True
    if not (fp["class_num"] > 1 and fp["refine_iter"] > 2):
        # cspswarm -> cspmerge
        not_retrying = csp_class_merge(class_index=1, input_dir=input_dir)
    else: 
        # cspswarm -> classmerge -> cspmerge
        if not classmerge_succeed(fp):
            raise Exception("One or more classmerge job(s) failed")

    iteration = fp["refine_iter"]

    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(fp["class_num"], iteration))

    fp["refine_dataset"] = mp["data_set"]
    # metric = project_params.param(mp["refine_metric"], iteration).lower()

    # initialize directory structure to replicate frealign folders
    local_scratch = os.environ["PYP_SCRATCH"]
    local_frealign = os.path.join(local_scratch, "frealign")
    local_frealign_scratch = os.path.join(local_frealign, "scratch")
    for dir in ["maps", "scratch", "log"]:
        os.makedirs(os.path.join(local_frealign, dir), exist_ok=True)
    # copy frealign metadata
    for file in glob.glob(os.path.join(input_dir, "../maps/*.txt")):
        shutil.copy2(file, os.path.join(local_frealign, "maps"))

    # copy pyp metadata to scracth space
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".micrographs"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".films"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], ".pyp_config.toml"), local_frealign_scratch
    )

    with timer.Timer(
        "plot fsc and clean par", text = "Plot FSC and producing clean par file took: {}", logger=logger.info
    ):
        # collate FSC curves from all references in one plot
        if classes > 1 and not Web.exists:

            metadata = {}
            logger.info("Creating plots for visualizing classes OCCs and FSCs")
            # plot class statistics
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, figsize=(8, 8))
            ranking = np.zeros([classes])
            for ref in range(classes):
                fsc_file = "../maps/" + fp["refine_dataset"] + "_r%02d_fsc.txt" % (ref + 1)
                FSCs = np.loadtxt(fsc_file, ndmin=2, dtype=float)

                if ref == 0:
                    metadata["frequency"] = FSCs[:, 0].tolist()
                metadata["fsc_%02d" % ( ref + 1 ) ] = FSCs[:, iteration - 1].tolist()

                ax[0].plot(
                    1 / FSCs[:, 0], FSCs[:, iteration - 1], label="r%02d" % (ref + 1),
                )
                ranking[ref] = FSCs[:, iteration - 1].mean()

                occ_only_file = os.path.join(input_dir, "occ_only_r%02d.npy" % (ref + 1))
                if not os.path.exists(occ_only_file):
                    assert Exception(f"{occ_only_file} does not exist. Please check")

                input = np.load(occ_only_file)
 
                sortedocc = np.sort(input)
                ax[1].plot(sortedocc, label="r%02d" % (ref + 1))
                metadata["occ_%02d" % ( ref + 1 ) ] = sortedocc.tolist()

            ax[0].legend(loc="upper right", shadow=True)
            ax[0].set_ylim((-0.1, 1.05))
            ax[0].set_xlim((1 / FSCs[0, 0], 1 / FSCs[-1, 0]))
            dataset = fp["refine_dataset"] + "_%02d" % iteration
            ax[0].set_title("%s" % dataset, fontsize=12)
            ax[0].set_xlabel("Frequency (1/A)")
            ax[0].set_ylabel("FSC")
            ax[1].legend(loc="upper right", shadow=True)
            ax[1].set_xlim(0, input.shape[0] - 1)
            ax[1].set_xlabel("Particle Index")
            ax[1].set_ylabel("Occupancy (%)")
            plt.savefig(os.path.join(input_dir, "../maps/%s_classes.png" % dataset))
            plt.close()

            with open( "../maps/%s_classes.json" % dataset, 'w' ) as f:
                json.dump( metadata, f )

        else:
            if len(glob.glob(f"{local_frealign_scratch}/*_scores.png")) > 0:
                command = "montage {0}/*_{1}_scores.png -geometry +0+0 {3}/../maps/{2}_{1}_scores.png".format(
                    local_frealign_scratch,
                    "r%02d_%02d" % (1, iteration),
                    fp["refine_dataset"],
                    input_dir,
                )
                local_run.run_shell_command(command, verbose=mp["slurm_verbose"])

    # remove the directory
    if not_retrying:
        shutil.rmtree(input_dir)

    # go back to project directory
    os.chdir(project_dir)

    # merge individual extracted star files 
    if mp["extract_stacks"]:

        stacks_folder = os.path.join(project_dir, "relion", "stacks")
        merged_star = os.path.join(stacks_folder, fp["data_set"] + "_particles.star")

        # delete merged_star if already exists
        try:
            os.remove(os.path.join(stacks_folder,merged_star))
        except:
            pass

        individual_star_files = glob.glob( os.path.join(project_dir, "relion", "stacks", "*.star"))

        shutil.move(individual_star_files[0], merged_star)

        for star in individual_star_files[1:]:
            command = "awk 'BEGIN {{OFS=\"\\t\"}}; NF>10{{print}}' {0} >> {1}".format(star, merged_star)
            local_run.run_shell_command(command, verbose=mp["slurm_verbose"])
            os.remove(star)

        logger.info(f"Stacks saved to {stacks_folder}")

    # update iteration number
    maxiter = fp["refine_maxiter"]
    if not_retrying:
        fp["refine_iter"] = iteration + 1
    fp["refine_dataset"] = mp["data_set"]
    if "refine_skip" in fp.keys() and fp["refine_skip"] and fp["class_num"] > 1:
        fp["refine_skip"] = False
    fp["slurm_merge_only"] = False
    project_params.save_parameters(fp, ".")

    # clean-up the previous parameter file folders if they exist
    refinement_path = Path().cwd() / "frealign" / "maps"
    parameter_file_folders = refinement_path.glob(f"{fp['data_set']}_r??_??")
    # [shutil.rmtree(folder) for folder in parameter_file_folders if Path(f"{folder}.bz2").exists()]
    for folder in parameter_file_folders:
        f = Path(f"{folder}.bz2")
        class_one_file = f.parents[0] / f"{f.name[:-11]}_r01_02.bz2"
        if Path(f).exists() or Path(folder).name.endswith("r01_01") or class_one_file.exists() and f"{folder.name[-2:]}" == "02":
            try:
                shutil.rmtree(folder)
            except:
                raise Exception("Failed to copy files?")

    # export metadata in star format
    if mp["reconstruct_export_enable"] and "local" not in mp["extract_fmt"].lower():

        with timer.Timer(
            "Export to star", text = "Export metadata to .star format took: {}", logger=logger.info
        ):
            mode = mp["data_mode"].lower()
            export_iteration = mp["refine_iter"]
            micrographs = {}
            all_micrographs_file = mp["data_set"] + ".films"
            with open(all_micrographs_file) as f:
                index = 0
                for line in f.readlines():
                    micrographs[line.strip()] = index
                    index += 1

            par_input = os.path.join(project_dir, "frealign", "maps", mp["data_set"] + "_r01_%02d" % ( export_iteration - 1 ) + ".bz2")

            if not os.path.exists(par_input):
                raise Exception(f"Cannot find {par_input} to read particle alignments")

            if os.path.isdir(par_input):
                parfile = par_input
            elif par_input.endswith(".bz2"):
                parfile = par_input.replace(".bz2", "")
                frealign_parfile.Parameters.decompress_parameter_file_and_move(Path(par_input), Path(parfile), threads=mp["slurm_tasks"])
            else:
                raise Exception(f"Unknown parfile format: {par_input}")

            imagelist = list(micrographs.keys())

            globalmeta = pyp_metadata.GlobalMetadata(
                mp["data_set"],
                mp,
                imagelist=imagelist,
                mode=mode,
                getpickle=True,
                parfile=parfile,
                path="./pkl"
                )

            select = mp["extract_cls"]
            globalmeta.meta2Star(mp["data_set"] + ".star", imagelist, select=select, stack="stack.mrc", parfile=parfile)
            
            # cleanup
            shutil.rmtree(Path(parfile), ignore_errors=True)

    # launch next iteration if needed
    if iteration < maxiter:
        if not standalone_mode():
            logger.info("Now launching iteration " + str(iteration + 1))
            csp_split(fp, iteration + 1)
        else:
            logger.warning(f"Standalone mode does not support running multiple iterations. Please run csp manually again.")


def rename_csp_local_files(dataset_name, input_dir, ordering, pattern, metric):
    import re

    curr_dir = os.getcwd()
    os.chdir(input_dir)
    files = os.listdir(os.getcwd())

    if "new" in metric or "frealignx" in metric:
        # p = re.compile(r"(\w+)_r01_02_(map\d)_n1.mrc")
        p = re.compile(r"(\w+)_%s_map1_(n\d+).mrc" % pattern)

        new_files = []
        order = 1

        files = sorted([f for f in os.listdir(os.getcwd()) if p.match(f)], key=lambda x: int(x.split("_")[1]))

        for f in files:
            if p.match(f):
                match = p.match(f)
                old_name = match.group()
                old_name2 = old_name.replace("map1", "map2")
                # job_name, map_no, dump_num = match.groups()
                # job_name, dump_num = match.groups()

                # order = next((i for i, v in enumerate(ordering, 1) if v[0] == job_name), -1)
                # new_name = "{0}_{1}_n{2}.mrc".format(dataset_name, map_no, order)
                new_name = "{0}_map1_n{1}.mrc".format(dataset_name, order)
                new_name2 = "{0}_map2_n{1}.mrc".format(dataset_name, order)

                os.rename(old_name, new_name)
                os.rename(old_name2, new_name2)

                new_files.append(new_name)
                new_files.append(new_name2)

                order += 1
    elif "cc" in metric:
        p = re.compile(r"(\w+)_%s_(\w+).mrc" % pattern)

        new_files = []
        order = 1

        for f in files:
            if p.match(f):
                match = p.match(f)
                old_name = match.group()
                job_name, dump_num = match.groups()

                new_name = "{0}_n{1}.mrc".format(dataset_name, order)

                # logger.info("symlinking from {} to {}".format(old_name, new_name))
                symlink_force(old_name, new_name)

                new_files.append(new_name)

                order += 1

    os.chdir(curr_dir)
    return new_files


def rename_par_local_files(
    dataset_name,
    input_dir,
    ordering,
    old_pattern=r"(\w+)_r01_02.par",
    new_pattern="{0}_{1:04d}_r01_02.par",
):
    import re

    curr_dir = os.getcwd()
    os.chdir(input_dir)
    files = os.listdir(os.getcwd())
    p = re.compile(old_pattern)

    new_files = []
    for f in files:
        if p.match(f):
            match = p.match(f)
            old_name = match.group()
            job_name = match.groups()[0]

            # order = next((i for i, v in enumerate(ordering, 1) if v[0] == job_name), -1)
            order = [i for i in range(len(ordering)) if ordering[i] == job_name][0]
            new_name = new_pattern.format(dataset_name, order)

            # logger.info("symlinking from {} to {}".format(old_name, new_name))
            symlink_force(old_name, new_name)
            new_files.append(new_name)

    os.chdir(curr_dir)
    new_files.sort()
    return new_files


@timer.Timer(
    "live_decompress_and_merge", text="Live decompress and merge took: {}", logger=logger.info
)
def live_decompress_and_merge(class_index, input_dir, parameters, micrographs, all_jobs, merge=True):
    """ Perform live bz2 file decompression and intermediate merging once single cspswarm completes.  

    Args:
        input_dir (str): Directory where cspswarm stores its compressed files
        parameters (dict): PYP parameters
        micrographs (dict): Micorgraphs derived from .films file
        all_jobs (list): job id list that will be used for later parfile merging 
        merge (boolean): perform intermediate merge or not (now only support metrics other than cc3m, cclin)

    Raises:
        Exception: Check if number of dumpfiles is correct
    """

    # timer will be reset if we get a batch of files
    # set the timer to slurm_merge_walltime - 10 min 
    TIMEOUT =  slurm.get_total_seconds(parameters["slurm_walltime"]) - 10 * 60 
    INTERVAL = 10           # 10 s 
    start_time = time.time()

    path_to_logs = Path(input_dir).parent.parent / "log"
    iteration = parameters["refine_iter"]
    compressed = parameters["refine_parfile_compress"]
    if iteration == 2:
        classes = 1
    else:
        classes = int(project_params.param(parameters["class_num"], iteration))
    if compressed:
        num_dumpfiles_per_bundle = 1
        if standalone_mode():
            num_bundle = 1
        else:
            num_bundle = math.ceil(len(micrographs.keys()) / parameters["slurm_bundle_size"])
        num_bz2_per_bundle = 1 # classes
        total_num_bz2 = num_bz2_per_bundle * num_bundle
        
        # there should only be one intermediate reconstruction when running in standalone mode
        if standalone_mode():
            total_num_bz2 = 1

        decompression_threshold = min(parameters["slurm_merge_tasks"] - 1, total_num_bz2) 

        decompressed = set()
        decompression_queue = set()
        processed_micrographs = set()
        class_to_merge = [False for _ in range(classes)]
        dumpfiles_count_class = [1 for _ in range(classes)]
        processed = 0
        succeed = False
        
        empty_file_number = 0
        
        logger.info("Start live-merging intermediate reconstructions")
        while time.time() - start_time < TIMEOUT:

            arguments = []
            finished_micrographs = glob.glob(os.path.join(input_dir, ".*"))

            compressed_files = glob.glob(os.path.join(input_dir, f"*_r{class_index:02d}*.bz2"))

            # find un-processed bz2 files
            for f in compressed_files:
                filename = os.path.basename(f)
                if filename not in decompressed:
                    decompression_queue.add(filename)

            # decompress them if we find enough files
            if len(decompression_queue) >= decompression_threshold - empty_file_number:

                for filename in decompression_queue:

                    arguments.append((os.path.join(input_dir, filename), 1,))
                    decompressed.add(filename)

                    class_ind = int(filename.replace(".bz2", "").split("_r")[-1]) - 1
                    class_to_merge[class_ind] = True

                    jobid = filename.split("_r")[0].split("_")
                    
                    cspswarm_jobid = int(jobid[0])
                    cspswarm_arrid = int(jobid[1])
                    [all_jobs.append([cspswarm_jobid, cspswarm_arrid]) for _ in range(num_dumpfiles_per_bundle)]

                decompression_threshold = min(parameters["slurm_merge_tasks"] - 1, total_num_bz2 - len(decompressed))
                decompression_queue.clear()

            if len(arguments) > 0:
                mpi.submit_function_to_workers(
                    frealign_parfile.Parameters.decompress_file, arguments, verbose=parameters["slurm_verbose"]
                )
                # reset if we get a batch
                start_time = time.time()

                # if they're successfully decompressed we think they're complete
                [processed_micrographs.add(micrograph.split(".")[-1]) for micrograph in finished_micrographs]
                processed += len(arguments)

            if merge:
                # perform intermediate merge on files we just decompressed
                class_ind = class_index - 1
                # only merge decompressed intermediate reconstructions
                if class_to_merge[class_ind]:
                    pattern = "r%02d" % (class_index)
                    num_dumpfiles = frealign.local_merge_reconstruction(name=pattern)
                    class_to_merge[class_ind] = False
                    dumpfiles_count_class[class_ind] += num_dumpfiles - 1   # the 1 is output dumpfile

            # done processing all micrographs
            # if len(set(micrographs.keys()) - processed_micrographs) == 0:

            # check for empty micrographs/tilt-series
            empty_files = glob.glob(os.path.join(input_dir,"*.empty"))
            empty_file_number = len(empty_files)
            # clear flags and revise total processed count
            if total_num_bz2 - processed - empty_file_number == 0:
                # check the number of dumpfiles is correct
                if merge:
                    assert (dumpfiles_count_class[class_index-1] + empty_file_number == num_bundle * num_dumpfiles_per_bundle), f"{dumpfiles_count_class[class_index-1]} dumpfiles in class {class_index} is not {num_bundle * num_dumpfiles_per_bundle}"
                succeed = True
                break

            # check if there's error from logs. If yes, we stop the merge job
            if csp_has_error(path_to_logs, micrographs):
                return False

            # exit loop if we are resuming from a failed run
            if parameters['slurm_merge_only'] and classes == 1:
                break

            time.sleep(INTERVAL)
        logger.info("Done live-merging intermediate reconstructions")

    else:
        # NOTE: have not yet tested
        pwd = os.getcwd()
        os.chdir(input_dir)
        num_dumpfiles_per_bundle = 1
        num_bundle = math.ceil(len(micrographs.keys()) / parameters["slurm_bundle_size"])
        total_num_dump_perclass =num_dumpfiles_per_bundle * num_bundle

        processed_micrographs = set()
        class_to_merge = [True for _ in range(classes)]
        dumpfiles_count_class = [1 for _ in range(classes)]
        processed = 0
        succeed = False
        while time.time() - start_time < TIMEOUT:
            if merge:
                # only merge decompressed intermediate reconstructions
                if class_to_merge[class_index-1]:
                    pattern = "r%02d_%02d" % (class_index, iteration)
                    num_dumpfiles = frealign.local_merge_reconstruction(name=pattern)
                    dumpfiles_count_class[class_index-1] += num_dumpfiles - 1   # the 1 is output dumpfile
            else:
                # check all the intermediate files
                if class_to_merge[class_index-1]:
                    pattern = "r%02d_%02d.par" % (class_index, iteration)
                    dump_par_num = len(glob.glob("*" + pattern))
                    dumpfiles_count_class[class_index-1] = dump_par_num

            if dumpfiles_count_class[class_index-1] == total_num_dump_perclass:
                class_to_merge[class_index-1] = False
            else:
                class_to_merge[class_index-1] = True

            if not any(class_to_merge):
                succeed = True
                os.chdir(pwd)
                break

            # check if there's error from logs. If yes, we stop the merge job
            if csp_has_error(path_to_logs, micrographs):
                return False

            time.sleep(INTERVAL)

    if not succeed:
        # result is incomplete after TIMEOUT -> need to resubmit failed cspswarm jobs
        if class_index == 1 and len(set(micrographs.keys()) - processed_micrographs) > 0:
            merge_check_err_and_resubmit(parameters, input_dir, micrographs, int(parameters["refine_iter"]))
        return False
    else:
        logger.info(f"Decompression of all micrographs/tilt-series is done, start merging reconstruction and parameter files")

    return True

def csp_has_error(path_to_logs: Path, micrographs: dict) -> bool:

    has_error = False
    ERROR_KEYWORDS = ["PYP (cspswarm) failed"]

    for micrograph in micrographs.keys():
        micrograph_log = Path(path_to_logs, f"{micrograph}_csp.log")
        if micrograph_log.exists():
            # use "grep" to check if log files contain any error message
            command = "grep -E %s '%s'" % ("'" + "|".join(ERROR_KEYWORDS) + "'", str(micrograph_log))
            [output, error] = local_run.run_shell_command(command, verbose=False)

            if len(output) > 0:
                logger.error(f"{micrograph} fails. Stopping the merge job.")
                logger.error(output)
                has_error = True
                break

    return has_error



def csp_class_merge(class_index: int, input_dir="scratch", ordering_file="ordering.txt"):

    # we are originally in the project directory
    project_dir = os.getcwd()

    # now switch to frealign/scratch directory where all the .bz2 files are
    Path(input_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(input_dir)

    # load pyp params from main folder
    mp = project_params.load_pyp_parameters("../../")
    fp = mp

    iteration = fp["refine_iter"]

    fp["refine_dataset"] = mp["data_set"]
    metric = project_params.param(mp["refine_metric"], iteration).lower()

    # initialize directory structure to replicate frealign folders
    local_scratch = os.environ["PYP_SCRATCH"]
    local_frealign = os.path.join(local_scratch, "frealign")
    local_frealign_scratch = os.path.join(local_frealign, "scratch")
    for dir in ["maps", "scratch", "log"]:
        os.makedirs(os.path.join(local_frealign, dir), exist_ok=True)
    # copy frealign metadata
    for file in glob.glob(os.path.join(input_dir, "../maps/*.txt")):
        shutil.copy2(file, os.path.join(local_frealign, "maps"))

    # copy pyp metadata to scratch space
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".micrographs"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], fp["data_set"] + ".films"),
        local_frealign_scratch,
    )
    shutil.copy2(
        os.path.join(Path(input_dir).parents[1], ".pyp_config.toml"), local_frealign_scratch
    )

    micrographs = {}
    all_micrographs_file = "../../" + mp["data_set"] + ".films"
    with open(all_micrographs_file) as f:
        index = 0
        for line in f.readlines():
            micrographs[line.strip()] = index
            index += 1

    total_micrographs = len(micrographs.keys())

    (number_of_reconstructions_per_micrograph, _) = get_number_of_intermediate_reconstructions(mp)

    # each class should have (num_dumpfiles_per_bundle * num_bundle) dumpfiles
    num_dumpfiles_per_bundle = 1 if "cc" not in metric else number_of_reconstructions_per_micrograph - 1
    num_bundle = math.ceil(total_micrographs / mp["slurm_bundle_size"])

    orderings = ["" for _ in range(num_dumpfiles_per_bundle * num_bundle)]

    # decompress all intermediate dump files in local scratch
    os.chdir(local_frealign_scratch)

    all_jobs = []

    collect_all_cspswarm = live_decompress_and_merge(class_index, input_dir, mp, micrographs, all_jobs, merge=(
        (("spr" not in mp["data_mode"] or "local" in mp["extract_fmt"]) and iteration > 2 )
        )
    )

    if collect_all_cspswarm:

        if not fp["refine_parfile_compress"]:
            # copy the mrc files and parfiles to scratch
            with timer.Timer("Copy mrc, par to scratch", text = "Copy file to merge took: {}", logger=logger.info):
                for file in glob.glob(input_dir + "/*mrc") + glob.glob(input_dir + "/*par"):
                    symlink_relative(file, os.path.join(local_frealign_scratch, os.path.basename(file)))

        if "cc" not in metric and fp["refine_parfile_compress"]:
            all_jobs = np.atleast_2d(np.array(all_jobs))
        else:
            all_jobs = np.atleast_2d(np.array(
                [
                    i.split("_r01_")[0].split("_")
                    for i in glob.glob("*_r01_%02d.par" % iteration)
                ],
                dtype=int,
            ))

        # identify the different job IDs
        if all_jobs.size > 0:
            slurm_job_ids = sorted(np.unique(all_jobs[:, 0]))
            for job in slurm_job_ids:
                # figure out number of empty slots
                zero_indexes = [i for i in range(len(orderings)) if orderings[i] == ""]
                # get the job array IDs for this job
                array_ids = sorted(all_jobs[all_jobs[:, 0] == job][:, 1])
                if len(zero_indexes) >= len(array_ids):
                    for id in array_ids:
                        orderings[zero_indexes[id - 1]] = str(job) + "_" + str(id)
                else:
                    message = "Number of missing jobs ({}) does not match the number of missing movies ({}).".format(
                        len(zero_indexes), len(array_ids)
                    )
                    raise Exception(message)

            with timer.Timer(
                "Parallel run all reconstruction", text = "Run parallel reconstruction took: {}", logger=logger.info
            ):
                ref = class_index
                pattern = "r%02d" % (ref)
                dataset_name = fp["refine_dataset"] + "_%s" % pattern

                run_mpi_reconstruction(ref, pattern, dataset_name, iteration, mp, fp, input_dir, orderings)

    os.chdir(project_dir)
    
    return collect_all_cspswarm


def classmerge_succeed(parameters: dict) -> bool: 
    """classmerge_succeed Check if classmerge jobs all succeed. If not, either terminate current cspmerge or relaunch classmerge/cspmerge

    Parameters
    ----------
    parameters : dict
        PYP parameters 

    Returns
    -------
    bool
        Succeed or not
    """
    # see if classmerge resubmits cspwarm by its logs
    # currently in frealign/scratch
    frealign_maps = Path().cwd().parent / "maps"
    swarm_folder = Path().cwd().parent.parent / "swarm"
    cspswarm_fail_tag = swarm_folder / ".csp_current_fail"

    dataset = parameters["data_set"]
    iteration = parameters["refine_iter"]
    num_classes = parameters["class_num"] if parameters["refine_iter"] > 2 else 1
    maps_classes = [f"{dataset}_r{class_idx+1:02d}_{iteration:02d}.mrc" for class_idx in range(num_classes)]
 
    TIMEOUT =  slurm.get_total_seconds(parameters["slurm_merge_walltime"]) - 10 * 60 
    INTERVAL = 10           # 10 s 
    start_time = time.time()

    error_flag = os.path.join( Path().cwd().parent.parent, ".cspswarm.error" )
    if os.path.exists(error_flag):
        try:
            os.remove(error_flag)
        except:
            pass
        raise Exception("Classmerge failed to run. Please check for errors in the logs")

    while time.time() - start_time < TIMEOUT:
        
        if cspswarm_fail_tag.exists():
            # partial cspswarm(s) & classmerge & cspmerge are all resubmitted by classmerge (one or more cspswarm(s) failed)
            # so terminate this cspmerge directly
            os.remove(cspswarm_fail_tag)
            return False

        classmerge_all_complete = True
        for map in maps_classes:
            if not (frealign_maps / map).exists():  
                classmerge_all_complete = False 

        if classmerge_all_complete:
            return True       

        time.sleep(INTERVAL)

    # part of the classmerge jobs failed, resubmit classmerge & cspmerge (w/o cspswarm)
    parameters["slurm_merge_only"] = True
    
    slurm.launch_csp(micrograph_list=[],
                    parameters=parameters,
                    swarm_folder=swarm_folder,
                    )

    return False

def create_cistem_from_2d_tracks(
    name: str,
    track_data: Dict,
    parameters: Dict,
    output_dir: str = "."
) -> Tuple[str, str]:
    """Create .cistem parameter files from 2D track data for tilt-series refinement.
    
    This function creates the required .cistem and _extended.cistem files needed for
    CSPT refinement using only 2D particle tracks across tilted micrographs.
    
    Parameters
    ----------
    name : str
        Name of the tilt-series (e.g., "ts001")
    track_data : Dict
        Dictionary containing 2D track information:
        - 'particles': dict[int, dict] - particle data keyed by particle ID
        - 'tilts': dict[int, dict] - tilt data keyed by tilt index
        - 'projections': list[dict] - list of projection data for each particle-tilt pair
    parameters : Dict
        PYP parameters dictionary
    output_dir : str, optional
        Output directory for the .cistem files, by default "."
        
    Returns
    -------
    Tuple[str, str]
        Paths to the created .cistem and _extended.cistem files
    """
    from pyp.inout.metadata.frealign_parfile import Parameters, ExtendedParameters
    from pyp.inout.metadata.cistem_star_file import Tilt, Particle
    
    # Create Parameters object for projection data
    par_obj = Parameters()
    
    # Build projection data array from track_data['projections']
    projection_rows = []
    for proj in track_data['projections']:
        pind = proj['pind']
        tind = proj['tind']
        
        # Get particle and tilt info
        particle = track_data['particles'][pind]
        tilt = track_data['tilts'][tind]
        
        # Create a projection row with required columns
        # Column order follows frealign parameter file format
        row = [
            pind,                    # PIND - particle index
            tind,                    # TIND - tilt index  
            0,                       # RIND - region index (will be set by CSPT)
            proj.get('shift_x', 0.0), # SHIFT_X - 2D shift from track
            proj.get('shift_y', 0.0), # SHIFT_Y - 2D shift from track
            0.0,                     # SHIFT_Z - not used for 2D
            0.0,                     # PSI - rotation angle
            0.0,                     # THETA - tilt angle  
            0.0,                     # PHI - rotation angle
            0.0,                     # DEFOCUS - will be set from tilt data
            0.0,                     # DEFOCUS_ANGLE
            0.0,                     # PHASE_SHIFT
            0.0,                     # FILM_ID
            0.0,                     # IMAGE_IS_ACTIVE
            proj.get('score', 0.0),  # SCORE
            proj.get('occupancy', 1.0), # OCCUPANCY
            proj.get('logp', 0.0),   # LOGP
            proj.get('sigma', 1.0),  # SIGMA
            proj.get('score_change', 0.0), # SCORE_CHANGE
            proj.get('pixel_error', 0.0),  # PIXEL_ERROR
            proj.get('defocus_change', 0.0), # DEFOCUS_CHANGE
            proj.get('psi_change', 0.0),     # PSI_CHANGE
            proj.get('theta_change', 0.0),   # THETA_CHANGE
            proj.get('phi_change', 0.0),     # PHI_CHANGE
            proj.get('shift_x_change', 0.0), # SHIFT_X_CHANGE
            proj.get('shift_y_change', 0.0), # SHIFT_Y_CHANGE
            proj.get('shift_z_change', 0.0), # SHIFT_Z_CHANGE
            proj.get('tilt_angle', tilt.get('angle', 0.0)), # TILT_ANGLE
            proj.get('tilt_axis_angle', 0.0), # TILT_AXIS_ANGLE
            proj.get('defocus_1', tilt.get('defocus', 0.0)), # DEFOCUS_1
            proj.get('defocus_2', tilt.get('defocus', 0.0)), # DEFOCUS_2
            proj.get('defocus_angle', 0.0),  # DEFOCUS_ANGLE
            proj.get('phase_shift', 0.0),    # PHASE_SHIFT
            proj.get('amplitude_contrast', 0.1), # AMPLITUDE_CONTRAST
            proj.get('micrograph_name', f"{name}_t{tind:02d}"), # MICROGRAPH_NAME
            proj.get('micrograph_path', ""), # MICROGRAPH_PATH
        ]
        projection_rows.append(row)
    
    # Convert to numpy array and set data
    projection_data = np.array(projection_rows, dtype=float)
    par_obj.set_data(projection_data)
    
    # Create ExtendedParameters object
    ext_obj = ExtendedParameters()
    
    # Build particles dictionary
    particles_dict = {}
    for pind, particle_data in track_data['particles'].items():
        particle = Particle()
        particle.particle_index = pind
        particle.x_position_3d = particle_data.get('x_3d', 0.0)
        particle.y_position_3d = particle_data.get('y_3d', 0.0) 
        particle.z_position_3d = particle_data.get('z_3d', 0.0)
        particles_dict[pind] = particle
    
    # Build tilts dictionary
    tilts_dict = {}
    for tind, tilt_data in track_data['tilts'].items():
        tilt = Tilt()
        tilt.tilt_index = tind
        tilt.angle = tilt_data.get('angle', 0.0)
        tilt.defocus = tilt_data.get('defocus', 0.0)
        tilt.region_index = 0  # Will be set by CSPT
        tilts_dict[tind] = {0: tilt}  # Nested dict structure expected by ExtendedParameters
    
    # Set extended data
    ext_obj.set_data(particles=particles_dict, tilts=tilts_dict)
    
    # Write files
    cistem_file = os.path.join(output_dir, f"{name}_r01.cistem")
    extended_file = os.path.join(output_dir, f"{name}_r01_extended.cistem")
    
    par_obj.to_binary(cistem_file)
    ext_obj.to_binary(extended_file)
    
    logger.info(f"Created .cistem files for {name}:")
    logger.info(f"  - {cistem_file} ({len(projection_rows)} projections)")
    logger.info(f"  - {extended_file} ({len(particles_dict)} particles, {len(tilts_dict)} tilts)")
    
    return cistem_file, extended_file