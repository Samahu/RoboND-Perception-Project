#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

import xml.etree.ElementTree as et
import re


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(5)
    outlier_filter.set_std_dev_mul_thresh(0.4)
    cloud_filtered = outlier_filter.filter()


    # TODO: Voxel Grid Downsampling
    vox_filter = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox_filter.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox_filter.filter()

    # TODO: PassThrough Filter
    passthrough_z = cloud_filtered.make_passthrough_filter()
    passthrough_z.set_filter_field_name('z')
    passthrough_z.set_filter_limits(0.5, 1.5)
    cloud_filtered = passthrough_z.filter()

    passthrough_x = cloud_filtered.make_passthrough_filter()
    passthrough_x.set_filter_field_name('x')
    passthrough_x.set_filter_limits(0.35, 1.0)
    cloud_filtered = passthrough_x.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    MAX_DISTANCE = 0.01
    seg.set_distance_threshold(MAX_DISTANCE)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0], white_cloud[indice][1], white_cloud[indice][2], rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_table_pub.publish(ros_cloud_table)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs:

    detected_objects_labels = []
    detected_objects = []

    # Classify the clusters! (loop through each detected cluster one at a time)
    for index, indices in enumerate(cluster_indices):

        # Grab the points for the cluster
        a_cluster = pcl_to_ros(extracted_outliers.extract(indices))
        rospy.loginfo("cluster size for object {}: {}".format(index, a_cluster.width))

        # Compute the associated feature vector
        chists = compute_color_histograms(a_cluster, using_hsv=True)
        normals = get_normals(a_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))

        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[indices[0]])
        label_pos[2] += 0.4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = a_cluster
        detected_objects.append(do)

    rospy.loginfo("Detected {} objects: {} ".format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


def find_object_by_name(object_list, name):

    for obj in object_list:
        if obj.label == name:
            return obj

    return None

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    labels = []
    centroids = []
    dict_list = []

    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_filename = '/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/config/dropbox.yaml'
    dropbox_dict = None
    with open(dropbox_filename, "r") as stream:
        dropbox_dict = yaml.load(stream)

    launch_filename = '/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/launch/pick_place_project.launch'
    world_name = "test1.world"
    with open(launch_filename, "r") as stream:
        element = et.parse(stream)
        nodes = element.findall('./launch/include/arg')
        for n in nodes:
            print(n.text)
            if n.attrib('name') == 'world_name':
                world_name = n.attrib('value')
                break


    # TODO: Parse parameters into individual variables
    print("World Name: " + world_name)
    test_scene_num = re.findall(r'\d', world_name)[0]   # no error handling
    TEST_SCENE_NUM = Int32()
    TEST_SCENE_NUM.data = int(test_scene_num)

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for i in range(0, len(object_list_param)):

        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        do = find_object_by_name(object_list, object_name)
        points_arr = ros_to_pcl(do.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[0:3]
        centroid = np.array([np.asscalar(centroid[0]), np.asscalar(centroid[1]), np.asscalar(centroid[2])])
        labels.append(do.label)
        centroids.append(centroid)

        OBJECT_NAME = String()
        OBJECT_NAME.data = object_name

        PICK_POSE = Pose()
        PICK_POSE.position.x = centroid[0]
        PICK_POSE.position.y = centroid[1]
        PICK_POSE.position.z = centroid[2]


        # TODO: Create 'place_pose' for the object
        place_position = dropbox_dict['dropbox'][1]['position'] if object_group =='green' else dropbox_dict['dropbox'][0]['position']
        PLACE_POSE = Pose()     # Assuming green/right always the second element in the array
        PLACE_POSE.position.x = place_position[0]
        PLACE_POSE.position.y = place_position[1]
        PLACE_POSE.position.z = place_position[2]

        # TODO: Assign the arm to be used for pick_place
        WHICH_ARM = String()
        WHICH_ARM.data = 'right' if object_group == 'green' else 'left'

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_' + str(test_scene_num) + '.yaml', dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open("model.sav", "rb"))
    clf = model["classifier"]
    encoder = LabelEncoder()
    encoder.classes_ = model["classes"]
    rospy.loginfo("Hello World")
    rospy.loginfo("classes: {} ".format(encoder.classes_))
    rospy.loginfo("Great!!!")
    scaler = model["scaler"]

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
