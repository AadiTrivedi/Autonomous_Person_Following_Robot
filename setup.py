from setuptools import find_packages, setup

package_name = "follow_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sim.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aadi",
    maintainer_email="aadi@todo.com",
    description="Person-following robot with fall detection",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = follow_robot.nodes.detector_node:main",
            "reid_node = follow_robot.nodes.reid_node:main",
            "pose_node = follow_robot.nodes.pose_node:main",
            "viz_node = follow_robot.nodes.viz_node:main",
            "follow_controller = follow_robot.nodes.follow_controller:main",
            "motor_driver = follow_robot.nodes.motor_driver_node:main",
        ],
    },
)