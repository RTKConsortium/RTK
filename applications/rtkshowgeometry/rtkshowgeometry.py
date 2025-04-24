#!/usr/bin/env python
import numpy as np
import itk
from itk import RTK as rtk
import pyvista as pv
import argparse


def main():
    parser = argparse.ArgumentParser(
        description=" Create an interactive 3D viewer for the given geometry and projections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument("--geometry", "-g", help="Geometry file name", required=True)
    parser.add_argument(
        "--show_trajectory", "-t", help="Show source trajectory", action="store_true"
    )
    parser.add_argument("--input", "-i", help="Input volume")
    parser.add_argument("--hide_volume", help="hide input volume", action="store_true")

    rtk.add_rtkinputprojections_group(parser)

    # Remove the required flag for '--path' and '--regexp' arguments
    for action in parser._actions:
        if action.dest == "path":
            action.required = False
        if action.dest == "regexp":
            action.required = False

    args_info = parser.parse_args()

    OutputPixelType = itk.F
    Dimension = 3

    OutputImageType = itk.Image[OutputPixelType, Dimension]

    if args_info.regexp:
        reader = rtk.ProjectionsReader[OutputImageType].New()
        rtk.SetProjectionsReaderFromArgParse(reader, args_info)
        reader.Update()
        projections = reader.GetOutput()
    else:
        projections = None

    if args_info.geometry:
        geometry = rtk.read_geometry(args_info.geometry)

    if args_info.input:
        volume = itk.imread(args_info.input)

    plotter = pv.Plotter()

    all_points_list = []
    source_positions = []

    # Iterate through all frames to calculate bounds and positions
    for frame in range(len(geometry.GetGantryAngles())):
        source_pos = np.asarray(geometry.GetSourcePosition(frame), dtype=float)[:3]
        source_positions.append(source_pos)

        SDD = geometry.GetSourceToDetectorDistances()[frame]
        SID = geometry.GetSourceToIsocenterDistances()[frame]

        m = geometry.GetProjectionCoordinatesToFixedSystemMatrix(frame)
        m_np = np.asarray(m)

        if projections is not None:
            corners = []
            idx = list(projections.GetLargestPossibleRegion().GetIndex())
            size = list(projections.GetLargestPossibleRegion().GetSize())

            # Calculate detector corners from projection indices
            for i in range(4):
                corner_pt = projections.TransformIndexToPhysicalPoint(idx)
                corner = np.asarray(corner_pt, dtype=float)
                corner = np.append(corner, 1.0)
                corner = m_np.dot(corner)
                corners.append(corner[:3])

                if i == 0:
                    idx[0] += size[0] - 1
                elif i == 1:
                    idx[1] += size[1] - 1
                elif i == 2:
                    idx[0] -= size[0] - 1
        else:
            # Simulate detector corners for cases without projections
            detector_size = max(abs(source_pos)) * 0.4
            half_size = detector_size / 2
            local_corners = [
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0],
            ]
            for corner in local_corners:
                corners = [m_np.dot(np.append(corner, 1.0))[:3]]

        # Handle parallel geometry (SDD = 0)
        if SDD == 0:
            rot_matrix = np.asarray(geometry.GetRotationMatrix(frame))
            direction = np.array(
                [-rot_matrix[2][0], -rot_matrix[2][1], -rot_matrix[2][2]]
            )
            direction *= 2.0 * SID
            source_corners = [corner - direction for corner in corners]
            frame_points = np.vstack([source_pos] + corners + source_corners)
        else:
            frame_points = np.vstack([source_pos] + corners)

        all_points_list.append(frame_points)

    if args_info.show_trajectory:
        source_positions = np.array(source_positions)
        source_trajectory = pv.PolyData(source_positions)
        plotter.add_mesh(source_trajectory, color="red", opacity=0.3)

    # Calculate bounds for the entire geometry
    all_points = np.vstack(all_points_list)
    max_value = np.max(all_points)
    min_value = np.min(all_points)

    # Compute "nice" tick spacing for grid
    raw_spacing = (abs(max_value) + abs(min_value)) / 4.0
    exp_val = np.floor(np.log10(raw_spacing))
    f = raw_spacing / (10 ** exp_val)
    if f < 1.5:
        nf = 1.0
    elif f < 3.0:
        nf = 2.0
    elif f < 7.0:
        nf = 5.0
    else:
        nf = 10.0
    nice_tick = nf * (10 ** exp_val)

    # Adjust bounds to align with tick spacing
    min_bounds = np.floor(min_value / nice_tick) * nice_tick
    max_bounds = np.ceil(max_value / nice_tick) * nice_tick

    # Add bounding box
    box = pv.Cube(
        bounds=[min_bounds, max_bounds, min_bounds, max_bounds, min_bounds, max_bounds]
    )
    plotter.add_mesh(box, style="wireframe", opacity=0)

    # Configure plotter view
    plotter.show_grid(minor_ticks=True)
    plotter.renderer.SetGradientBackground(True)
    plotter.enable_parallel_projection()

    actors = {}

    if not args_info.hide_volume:
        # Handle volume visualization if provided
        if args_info.input:
            volume_array = itk.GetArrayFromImage(volume)
            spacing = volume.GetSpacing()
            origin = volume.GetOrigin()
            direction = volume.GetDirection()

            # Adjust volume array for correct orientation
            volume_array = np.transpose(volume_array, (2, 1, 0))

            dims = volume_array.shape
            grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
            grid.point_data["values"] = volume_array.ravel(order="F")

            # Apply direction matrix if needed
            if not np.allclose(direction, np.eye(3)):
                transform = pv.transform_matrix(direction.flatten())
                grid.transform(transform)

            center = grid.center
            slice_x = grid.slice(normal="x", origin=center)
            slice_y = grid.slice(normal="y", origin=center)
            slice_z = grid.slice(normal="z", origin=center)

            # Add slices with opacity
            plotter.add_mesh(slice_x, cmap="gray", show_scalar_bar=False, opacity=0.5)
            plotter.add_mesh(slice_y, cmap="gray", show_scalar_bar=False, opacity=0.5)
            plotter.add_mesh(slice_z, cmap="gray", show_scalar_bar=False, opacity=0.5)
        else:
            # Add a placeholder sphere if no volume is provided
            volume_sphere = pv.Sphere(
                radius=0.1 * geometry.GetSourceToIsocenterDistances()[0],
                center=[0, 0, 0],
            )
            plotter.add_mesh(volume_sphere, color="blue", opacity=0.5)

    def update(frame):
        frame = int(frame)

        # Update geometry parameters for the current frame
        gantry_angle = np.degrees(geometry.GetGantryAngles()[frame])
        out_of_plane_angle = np.degrees(geometry.GetOutOfPlaneAngles()[frame])
        in_plane_angle = np.degrees(geometry.GetInPlaneAngles()[frame])
        SID = geometry.GetSourceToIsocenterDistances()[frame]
        offset_x = geometry.GetSourceOffsetsX()[frame]
        offset_y = geometry.GetSourceOffsetsY()[frame]
        SDD = geometry.GetSourceToDetectorDistances()[frame]
        proj_offset_x = geometry.GetProjectionOffsetsX()[frame]
        proj_offset_y = geometry.GetProjectionOffsetsY()[frame]

        info_text = (
            f"Frame index {frame}\n"
            f"Gantry: {gantry_angle:.2f}°\n"
            f"Out-of-Plane: {out_of_plane_angle:.2f}°\n"
            f"In-Plane: {in_plane_angle:.2f}°\n"
            f"SID: {SID:.2f} mm\n"
            f"Offset X: {offset_x:.2f} mm\n"
            f"Offset Y: {offset_y:.2f} mm\n"
            f"SDD: {SDD:.2f} mm\n"
            f"Proj Offset X: {proj_offset_x:.2f} mm\n"
            f"Proj Offset Y: {proj_offset_y:.2f} mm"
        )

        m = geometry.GetProjectionCoordinatesToFixedSystemMatrix(frame)
        m_np = np.asarray(m)
        detector_position = m_np[:3, 3]
        source_pos = np.asarray(geometry.GetSourcePosition(frame), dtype=float)[:3]

        # Detector setup
        if projections is not None:
            corners = []
            idx = list(projections.GetLargestPossibleRegion().GetIndex())
            size = list(projections.GetLargestPossibleRegion().GetSize())

            for i in range(4):
                corner_pt = projections.TransformIndexToPhysicalPoint(idx)
                corner = np.asarray(corner_pt, dtype=float)
                corner = np.append(corner, 1.0)
                corner = m_np.dot(corner)
                corners.append(corner[:3])

                if i == 0:
                    idx[0] += size[0] - 1
                elif i == 1:
                    idx[1] += size[1] - 1
                elif i == 2:
                    idx[0] -= size[0] - 1

            proj_array = np.asarray(itk.GetArrayFromImage(projections))
            proj_slice = proj_array[frame].astype(np.float32)
            proj_slice -= proj_slice.min()
            proj_slice /= proj_slice.max() + 1e-8
            texture_data = (proj_slice * 255).astype(np.uint8)
        else:
            detector_size = SID * 0.4
            half_size = detector_size / 2
            local_corners = [
                [-half_size, -half_size, 0],
                [half_size, -half_size, 0],
                [half_size, half_size, 0],
                [-half_size, half_size, 0],
            ]

            corners = []
            for corner in local_corners:
                corner = np.append(corner, 1.0)
                corner = m_np.dot(corner)
                corners.append(corner[:3])

            texture_data = np.zeros((2, 2), dtype=np.uint8)

        corners_array = np.asarray(corners, dtype=float)

        # Handle cylindrical detector if radius is non-zero
        radius = geometry.GetRadiusCylindricalDetector()
        if radius > 0:
            detector_width = np.linalg.norm(corners[1] - corners[0])
            detector_height = np.linalg.norm(corners[3] - corners[0])

            # Create grid points
            n_points = 100
            u = np.linspace(-detector_width / 2, detector_width / 2, n_points)
            v = np.linspace(-detector_height / 2, detector_height / 2, n_points)
            U, V = np.meshgrid(u, v)

            # Calculate cylindrical coordinates
            Theta = U / radius
            detector_center = np.mean(corners_array, axis=0)
            u_dir = (corners[1] - corners[0]) / np.linalg.norm(corners[1] - corners[0])
            v_dir = (corners[3] - corners[0]) / np.linalg.norm(corners[3] - corners[0])
            n_dir = np.cross(u_dir, v_dir)

            # Create rotation matrix for detector orientation
            rot_matrix = np.column_stack([u_dir, v_dir, n_dir])

            # Calculate curved surface points
            X = radius * np.sin(Theta)
            Z = radius * (1 - np.cos(Theta))
            Y = V
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            points = points @ rot_matrix.T + detector_center

            # Create faces for the curved surface
            curved_faces = []
            for i in range(n_points - 1):
                for j in range(n_points - 1):
                    idx00 = i * n_points + j
                    idx01 = i * n_points + (j + 1)
                    idx10 = (i + 1) * n_points + j
                    idx11 = (i + 1) * n_points + (j + 1)
                    curved_faces.extend([3, idx00, idx01, idx11])
                    curved_faces.extend([3, idx00, idx11, idx10])

            # Create texture coordinates
            tcoords = np.zeros((n_points * n_points, 2), dtype=np.float32)
            u_coords, v_coords = np.meshgrid(
                np.linspace(0, 1, n_points),
                np.linspace(1, 0, n_points),
            )
            tcoords[:, 0] = u_coords.ravel()
            tcoords[:, 1] = v_coords.ravel()

            # Get the corner points of the curved surface
            curved_corners = [
                points[0],  # Bottom left
                points[n_points - 1],  # Bottom right
                points[-1],  # Top right
                points[-n_points],  # Top left
            ]

            corners_array = points
            faces = np.array(curved_faces)
            detector_poly = pv.PolyData(corners_array, faces)
            detector_poly.active_texture_coordinates = tcoords
            corners = curved_corners

        else:
            faces = np.hstack([[4], np.arange(4)])
            detector_poly = pv.PolyData(corners_array, faces)
            tcoords = np.asarray([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
            detector_poly.active_texture_coordinates = tcoords

        texture = pv.Texture(texture_data)

        u_vec = (corners[1] - corners[0]) * 0.5
        v_vec = (corners[3] - corners[0]) * 0.5

        arrow_params = {
            "tip_length": 0.1,
            "tip_radius": 0.04,
            "shaft_radius": 0.02,
            "scale": "auto",
        }

        # Create arrows from center
        u_arrow = pv.Arrow(start=detector_position, direction=u_vec, **arrow_params)
        v_arrow = pv.Arrow(start=detector_position, direction=v_vec, **arrow_params)

        # Update label positions to be near arrow tips
        u_label_pos = detector_position + u_vec
        v_label_pos = detector_position + v_vec

        if SDD == 0:
            rot_matrix = np.asarray(geometry.GetRotationMatrix(frame))
            direction = (
                np.array([-rot_matrix[2][0], -rot_matrix[2][1], -rot_matrix[2][2]])
                * 2.0
                * SID
            )
            source = detector_position - direction
            central_line = pv.Line(source, detector_position)
        else:
            central_line = pv.Line(source_pos, detector_position)

        # Update or create visualization actors
        if not actors:
            actors["detector"] = plotter.add_mesh(
                detector_poly, texture=texture, opacity=0.5
            )
            actors["central_line"] = plotter.add_mesh(
                central_line, color="black", line_width=2
            )
            actors["text"] = plotter.add_text(
                info_text, position="upper_right", font_size=10, color="white"
            )
            actors["u_dir"] = plotter.add_mesh(u_arrow, color="green")
            actors["v_dir"] = plotter.add_mesh(v_arrow, color="green")
        else:
            actors["detector"].GetMapper().GetInput().points = corners_array
            actors["detector"].texture = texture
            actors["central_line"].GetMapper().GetInput().points = central_line.points
            actors["text"].SetText(3, info_text)
            actors["u_dir"].GetMapper().GetInput().points = u_arrow.points
            actors["v_dir"].GetMapper().GetInput().points = v_arrow.points

        # Handle parallel geometry (SDD = 0)
        if SDD == 0:
            rot_matrix = np.asarray(geometry.GetRotationMatrix(frame))
            direction = np.array(
                [-rot_matrix[2][0], -rot_matrix[2][1], -rot_matrix[2][2]]
            )
            direction *= 2.0 * SID
            source_corners = [corner - direction for corner in corners]
            source_corners_array = np.asarray(source_corners, dtype=float)
            source_faces = np.hstack([[4], np.arange(4)])
            source_poly = pv.PolyData(source_corners_array, source_faces)

            if "source_plane" not in actors:
                actors["source_plane"] = plotter.add_mesh(
                    source_poly, color="red", opacity=0.3
                )
            else:
                actors[
                    "source_plane"
                ].GetMapper().GetInput().points = source_corners_array

            if "lines" not in actors:
                actors["lines"] = []
                for src_corner, det_corner in zip(source_corners, corners):
                    line = pv.Line(src_corner, det_corner)
                    actors["lines"].append(
                        plotter.add_mesh(line, color="black", line_width=1)
                    )
            else:
                for i, (src_corner, det_corner) in enumerate(
                    zip(source_corners, corners)
                ):
                    line = pv.Line(src_corner, det_corner)
                    actors["lines"][i].GetMapper().GetInput().points = line.points

        else:
            if "lines" not in actors:
                actors["lines"] = []
                for corner in corners:
                    line = pv.Line(source_pos, corner)
                    actors["lines"].append(
                        plotter.add_mesh(line, color="black", line_width=1)
                    )
            else:
                for i, corner in enumerate(corners):
                    line = pv.Line(source_pos, corner)
                    actors["lines"][i].GetMapper().GetInput().points = line.points

        if "labels" in actors:
            plotter.remove_actor(actors["labels"])
        actors["labels"] = plotter.add_point_labels(
            points=[u_label_pos, v_label_pos],
            labels=["u", "v"],
            text_color="green",
            font_size=24,
            shape_opacity=0,
            always_visible=True,
            show_points=False,
        )

    plotter.camera.zoom(0.8)

    plotter.add_slider_widget(
        callback=update,
        rng=[0, len(geometry.GetGantryAngles()) - 1],
        value=0,
        title="Frame",
        style="modern",
        pointa=(0.02, 0.07),
        pointb=(0.98, 0.07),
        color="lightgrey",
        fmt="%.0f",
        interaction_event="always",
    )

    update(0)
    plotter.show_axes()
    plotter.show()


if __name__ == "__main__":
    main()
