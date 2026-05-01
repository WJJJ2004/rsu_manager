from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'rsu_manager'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name,
            ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
        (os.path.join('share', package_name, 'robot_model'),
            glob('robot_model/*.urdf')),
        (os.path.join('share', package_name, 'robot_model/meshes'),
            glob('robot_model/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lwj',
    maintainer_email='dldnjswns7412@gmail.com',
    description='RSU solver and visualization manager',
    license='MIT',
    entry_points={
        'console_scripts': [
            'debug_solver_node = rsu_manager.node.debug_solver_node:main',
            'rt_solver_node = rsu_manager.node.rt_solver_node:main',
            'gamepad_rpy_node = rsu_manager.node.gamepad_rpy_node:main',
            'hw_controll_test_node = rsu_manager.node.hw_controll_test_node:main',
            'rsu_link_plotter_node = rsu_manager.node.rsu_link_plotter_node:main',
        ],
    },
)