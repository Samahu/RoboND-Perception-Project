## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

[//]: # (Image References)

[conf-matrix]: ./confusion-matrix.png
[scene3]: ./Screen-Shot-2018-09-25-at-11.12.23-PM.png


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

In addition to exercize 1, I have added a statistical outlier filter to reduce the noise.
Also added another passthrough filter along x axis to remove drop bins from being considered during clustering and classification phases.

```python

     # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(5)
    outlier_filter.set_std_dev_mul_thresh(0.4)
    cloud_filtered = outlier_filter.filter()


    # TODO: Voxel Grid Downsampling
    vox_filter = cloud_filtered.make_voxel_grid_filter()
    LEAF_SIZE = 0.004
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

```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.

```python

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

```

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

```python

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

```

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

the following code block shows how I have formed a PickPlace msg:

```python

    # TODO: Loop through the pick list
    for i in range(0, len(object_list_param)):

        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        do = find_object_by_name(object_list, object_name)
        points_arr = ros_to_pcl(do.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[0:3]
        centroid_np_array = np.array([np.asscalar(centroid[0]), np.asscalar(centroid[1]), np.asscalar(centroid[2])])
        labels.append(do.label)
        centroids.append(centroid_np_array)

        OBJECT_NAME = String()
        OBJECT_NAME.data = object_name

        PICK_POSE = Pose()
        PICK_POSE.position.x = np.asscalar(centroid[0])
        PICK_POSE.position.y = np.asscalar(centroid[1])
        PICK_POSE.position.z = np.asscalar(centroid[2])


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

```

The following image shows my confusion matrix which is based on 500 iterations:
![alt text][conf-matrix]

As far as the results are concerned I get 100% correct results on all scenes when testing locally.
The following image shows correct clustering and classification for 3rd scene:
![alt text][scene3]

What helped me get good results was:
- create another passthrough filter along x axis to exclude drop bins
- increase the sample iterations to 500 during training
- switch from linear kernel to sigmoid kernel (this was very significant)

