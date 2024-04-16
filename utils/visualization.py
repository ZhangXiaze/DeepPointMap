from typing import List
import open3d as o3d
import numpy as np
import torch


def show_pcd(pcds: List, colors: List = None, window_name: str = "PCD",
             has_normals: bool = False, estimate_normals: bool = False, estimate_kwargs: dict = None,
             filter: bool = False) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=2880, height=1620)

    print(f'{window_name:*<30}')
    for i in range(len(pcds)):
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        if isinstance(pcds[i], np.ndarray):
            pcd_points = pcds[i][:, :3]
        elif isinstance(pcds[i], torch.Tensor):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()
        else:
            pcd_points = np.array(pcds[i][:, :3])
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)

        if has_normals:
            if pcds[i].shape[1] < 6:
                print('Normals is NOT found')
            else:
                if isinstance(pcds[i], np.ndarray):
                    pcd_normals = pcds[i][:, 3:6]
                elif isinstance(pcds[i], torch.Tensor):
                    pcd_normals = pcds[i][:, 3:6].detach().cpu().numpy()
                else:
                    pcd_normals = np.array(pcds[i][:, 3:6])
                pcd_o3d.normals = o3d.open3d.utility.Vector3dVector(pcd_normals)

        if filter:
            pcd_o3d = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=3)[0]

        if estimate_normals:
            if estimate_kwargs is None:
                radius, max_nn = 1, 30
            else:
                assert 'radius' in estimate_kwargs.keys() and 'max_nn' in estimate_kwargs.keys()
                radius, max_nn = estimate_kwargs['radius'], estimate_kwargs['max_nn']
            pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        vis.add_geometry(pcd_o3d)
        print(pcd_o3d)
    print('*' * 30)

    vis.run()
    vis.destroy_window()
